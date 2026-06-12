#!/usr/bin/env python3
"""
CORTEX v2 — Multi-Model Experiment Runner
==========================================
Runs the full filter comparison across multiple LLM providers.
Supports: OpenRouter, Grok (xAI), Google Gemini, OpenAI.

  "Multi-model study: does filtering matter more or less for stronger models?"

What this runs:
  Condition A — No filter      (baseline)
  Condition B — Naive truncation
  Condition C — BM25FileSelector (new baseline)
  Condition D — TF-IDF Filter  (new baseline)
  Condition E — SizeFilter(1MB)
  Condition F — HybridFilter   (recommended)

  × Models: any combination of OpenRouter / Grok / Gemini

Output:
  experiments/results/multimodel/results_<timestamp>.json
  experiments/results/multimodel/summary_<timestamp>.csv

────────────────────────────────────────
SETUP (one-time):

  pip install requests  # only external dependency

  # Set your API keys (pick whichever you have):
  export OPENROUTER_API_KEY="sk-or-..."
  export GROK_API_KEY="xai-..."
  export GEMINI_API_KEY="AIza..."
  export OPENAI_API_KEY="sk-..."

────────────────────────────────────────
USAGE:

  # Dry run — shows what would happen, makes NO API calls, costs $0
  python experiments/scripts/run_multimodel.py --dry-run --limit 5

  # Run with Grok only
  python experiments/scripts/run_multimodel.py --models grok --limit 10

  python experiments/scripts/run_multimodel.py --models gemini-flash --limit 20

  # Run with OpenRouter (access to 100+ models)
  python experiments/scripts/run_multimodel.py --models openrouter --limit 20

  # Run ALL available models (uses whichever API keys are set)
  python experiments/scripts/run_multimodel.py --limit 30

  # Run all models, all 6 conditions, full 18-task evaluation
  python experiments/scripts/run_multimodel.py --all-conditions

  # Point to your cloned repos
  python experiments/scripts/run_multimodel.py \
      --repos-dir /path/to/experiments/repos \
      --models gemini-flash grok \
      --limit 18

────────────────────────────────────────
COST ESTIMATES (per 18-task evaluation):

  Gemini Flash 2.0     ~$0.05  
  Grok 3 Mini          ~$0.10
  GPT-4o Mini          ~$0.30
  OpenRouter (Llama-3) ~$0.05   (if using free/cheap models)
  Gemini Pro           ~$0.50
  Grok 3               ~$1.00
  GPT-4o               ~$3.00
────────────────────────────────────────
"""

import os
import sys
import json
import time
import argparse
import csv
import datetime
from pathlib import Path
from typing import Optional

# Add src/python to path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "python"))

# ── Load .env automatically — no pip install needed ───────────────────────────
def _load_dotenv(env_path: Path) -> None:
    """Pure-stdlib .env parser. Doesn't override real env vars."""
    if not env_path.exists():
        return
    with open(env_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val

_load_dotenv(REPO_ROOT / ".env")

from filter import (
    NoFilter, SizeFilter, HybridFilter,
    build_context, estimate_tokens, DEFAULT_THRESHOLD
)
from filters_v2 import BM25FileSelector, TFIDFRelevanceFilter

REPOS_DIR   = Path(os.environ.get("CORTEX_REPOS_DIR", REPO_ROOT / "experiments" / "repos"))
OUT_DIR     = REPO_ROOT / "experiments" / "results" / "multimodel"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONTEXT_BUDGET = 32_000  # tokens per call


# ═══════════════════════════════════════════════════════════════════════════════
# LLM Provider Adapters
# Each adapter has: name, model_id, call(prompt, context, dry_run) → dict
# ═══════════════════════════════════════════════════════════════════════════════

class BaseAdapter:
    name: str = "base"
    model_id: str = "unknown"
    cost_per_1k_input:  float = 0.0
    cost_per_1k_output: float = 0.0

    def is_available(self) -> bool:
        return False

    def call(self, system_prompt: str, user_message: str,
             dry_run: bool = False) -> dict:
        raise NotImplementedError


class OpenRouterAdapter(BaseAdapter):
    """
    OpenRouter — access to 100+ models through one API.
    Set OPENROUTER_API_KEY. Choose model with --openrouter-model.
    Popular free/cheap models: meta-llama/llama-3-8b-instruct,
                                mistralai/mistral-7b-instruct
    """
    name = "OpenRouter"

    def __init__(self, model: str = "meta-llama/llama-3.1-8b-instruct"):
        self.model_id = model
        # Pricing varies by model — these are for Llama 3.1 8B
        self.cost_per_1k_input  = 0.00006
        self.cost_per_1k_output = 0.00006

    def is_available(self) -> bool:
        return bool(os.environ.get("OPENROUTER_API_KEY"))

    def call(self, system_prompt: str, user_message: str,
             dry_run: bool = False) -> dict:
        if dry_run:
            return _dry_run_response(self.name, self.model_id)
        import requests
        api_key = os.environ["OPENROUTER_API_KEY"]
        t0 = time.perf_counter()
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/shweta-mishra-research/cortex",
                    "X-Title": "CORTEX Research",
                },
                json={
                    "model": self.model_id,
                    "messages": [
                        {"role": "system",  "content": system_prompt},
                        {"role": "user",    "content": user_message},
                    ],
                    "max_tokens": 512,
                    "temperature": 0.0,
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            latency_ms = (time.perf_counter() - t0) * 1000
            text = data["choices"][0]["message"]["content"] or ""
            usage = data.get("usage", {})
            inp   = usage.get("prompt_tokens", estimate_tokens(len(system_prompt + user_message)))
            out   = usage.get("completion_tokens", estimate_tokens(len(text)))
            cost  = (inp * self.cost_per_1k_input + out * self.cost_per_1k_output) / 1000
            return {"response": text, "input_tokens": inp, "output_tokens": out,
                    "cost_usd": round(cost, 6), "latency_ms": round(latency_ms, 1), "error": None}
        except Exception as e:
            return _error_response(str(e))


class GrokAdapter(BaseAdapter):
    """
    Grok (xAI) — Set GROK_API_KEY.
    Models: grok-3, grok-3-mini, grok-2
    """
    name = "Grok"

    MODELS = {
        "grok-3":      (0.003,  0.015),
        "grok-3-mini": (0.0003, 0.0005),
        "grok-2":      (0.002,  0.010),
    }

    def __init__(self, model: str = "grok-3-mini"):
        self.model_id = model
        pricing = self.MODELS.get(model, (0.001, 0.005))
        self.cost_per_1k_input  = pricing[0]
        self.cost_per_1k_output = pricing[1]

    def is_available(self) -> bool:
        return bool(os.environ.get("GROK_API_KEY"))

    def call(self, system_prompt: str, user_message: str,
             dry_run: bool = False) -> dict:
        if dry_run:
            return _dry_run_response(self.name, self.model_id)
        import requests
        api_key = os.environ["GROK_API_KEY"]
        t0 = time.perf_counter()
        try:
            resp = requests.post(
                "https://api.x.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": self.model_id,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_message},
                    ],
                    "max_tokens": 512,
                    "temperature": 0.0,
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            latency_ms = (time.perf_counter() - t0) * 1000
            text  = data["choices"][0]["message"]["content"] or ""
            usage = data.get("usage", {})
            inp   = usage.get("prompt_tokens", 0)
            out   = usage.get("completion_tokens", 0)
            cost  = (inp * self.cost_per_1k_input + out * self.cost_per_1k_output) / 1000
            return {"response": text, "input_tokens": inp, "output_tokens": out,
                    "cost_usd": round(cost, 6), "latency_ms": round(latency_ms, 1), "error": None}
        except Exception as e:
            return _error_response(str(e))


class GeminiAdapter(BaseAdapter):
    """
    Google Gemini — Set GEMINI_API_KEY.
    Models: gemini-2.0-flash, gemini-1.5-flash, gemini-1.5-pro
    """
    name = "Gemini"

    MODELS = {
        "gemini-2.0-flash":   (0.000075, 0.0003),
        "gemini-1.5-flash":   (0.000075, 0.0003),
        "gemini-1.5-pro":     (0.00125,  0.005),
        "gemini-2.5-pro":     (0.00125,  0.010),
    }

    def __init__(self, model: str = "gemini-2.0-flash"):
        self.model_id = model
        pricing = self.MODELS.get(model, (0.000075, 0.0003))
        self.cost_per_1k_input  = pricing[0]
        self.cost_per_1k_output = pricing[1]

    def is_available(self) -> bool:
        return bool(os.environ.get("GEMINI_API_KEY"))

    def call(self, system_prompt: str, user_message: str,
             dry_run: bool = False) -> dict:
        if dry_run:
            return _dry_run_response(self.name, self.model_id)
        import requests
        api_key = os.environ["GEMINI_API_KEY"]
        t0 = time.perf_counter()
        # Gemini uses generateContent endpoint
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{self.model_id}:generateContent?key={api_key}")
        try:
            resp = requests.post(url,
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [
                        {"role": "user", "parts": [
                            {"text": f"{system_prompt}\n\n{user_message}"}
                        ]}
                    ],
                    "generationConfig": {"maxOutputTokens": 512, "temperature": 0.0},
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            latency_ms = (time.perf_counter() - t0) * 1000
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            usage = data.get("usageMetadata", {})
            inp   = usage.get("promptTokenCount", estimate_tokens(len(system_prompt + user_message)))
            out   = usage.get("candidatesTokenCount", estimate_tokens(len(text)))
            cost  = (inp * self.cost_per_1k_input + out * self.cost_per_1k_output) / 1000
            return {"response": text, "input_tokens": inp, "output_tokens": out,
                    "cost_usd": round(cost, 6), "latency_ms": round(latency_ms, 1), "error": None}
        except Exception as e:
            return _error_response(str(e))


class OllamaAdapter(BaseAdapter):
    """Local model evaluation via Ollama (no API key required).

    Parameters
    ----------
    model : str
        Ollama model name. Default: ``codellama:7b-instruct``.
    base_url : str
        Ollama API base URL. Default: ``http://localhost:11434``.
    """

    name = "Ollama"

    def __init__(self, model: str = "codellama:7b-instruct",
                 base_url: str = "http://localhost:11434") -> None:
        self.model_id = model
        self.base_url = base_url
        self.cost_per_1k_input = 0.0
        self.cost_per_1k_output = 0.0

    def is_available(self) -> bool:
        import shutil
        return bool(shutil.which("ollama"))

    def call(self, system_prompt: str, user_message: str,
             dry_run: bool = False) -> dict:
        if dry_run:
            return _dry_run_response(self.name, self.model_id)
        import urllib.request, json as _json
        url = f"{self.base_url}/api/chat"
        payload = _json.dumps({
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            "stream": False,
        }).encode()
        t0 = time.perf_counter()
        try:
            req = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = _json.loads(resp.read())
            latency_ms = (time.perf_counter() - t0) * 1000
            text = data.get("message", {}).get("content", "")
            inp = data.get("prompt_eval_count", estimate_tokens(len(system_prompt)))
            out = data.get("eval_count", estimate_tokens(len(text)))
            return {"response": text, "input_tokens": inp, "output_tokens": out,
                    "cost_usd": 0.0, "latency_ms": round(latency_ms, 1), "error": None}
        except Exception as exc:
            return _error_response(str(exc))


class OpenAIAdapter(BaseAdapter):
    """OpenAI — Set OPENAI_API_KEY."""
    name = "OpenAI"

    MODELS = {
        "gpt-4o-mini": (0.00015, 0.0006),
        "gpt-4o":      (0.005,   0.015),
        "o1-mini":     (0.003,   0.012),
    }

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model_id = model
        pricing = self.MODELS.get(model, (0.001, 0.003))
        self.cost_per_1k_input  = pricing[0]
        self.cost_per_1k_output = pricing[1]

    def is_available(self) -> bool:
        return bool(os.environ.get("OPENAI_API_KEY"))

    def call(self, system_prompt: str, user_message: str,
             dry_run: bool = False) -> dict:
        if dry_run:
            return _dry_run_response(self.name, self.model_id)
        import requests
        api_key = os.environ["OPENAI_API_KEY"]
        t0 = time.perf_counter()
        try:
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": self.model_id,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_message},
                    ],
                    "max_tokens": 512,
                    "temperature": 0.0,
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            latency_ms = (time.perf_counter() - t0) * 1000
            text  = data["choices"][0]["message"]["content"] or ""
            usage = data.get("usage", {})
            inp   = usage.get("prompt_tokens", 0)
            out   = usage.get("completion_tokens", 0)
            cost  = (inp * self.cost_per_1k_input + out * self.cost_per_1k_output) / 1000
            return {"response": text, "input_tokens": inp, "output_tokens": out,
                    "cost_usd": round(cost, 6), "latency_ms": round(latency_ms, 1), "error": None}
        except Exception as e:
            return _error_response(str(e))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dry_run_response(provider: str, model: str) -> dict:
    return {
        "response": f"[DRY RUN — {provider}/{model} — no API call made]",
        "input_tokens": 1500, "output_tokens": 150,
        "cost_usd": 0.0, "latency_ms": 0.0, "error": None,
    }

def _error_response(msg: str) -> dict:
    return {"response": "", "input_tokens": 0, "output_tokens": 0,
            "cost_usd": 0.0, "latency_ms": 0.0, "error": msg}


# ── Context builders ──────────────────────────────────────────────────────────

def build_naive_truncation(repo_path: str, budget: int) -> str:
    """Baseline: alphabetical concatenation, truncated at budget."""
    chunks, used = [], 0
    base = Path(repo_path)
    for fp in sorted(base.rglob("*")):
        if not fp.is_file(): continue
        if any(p in {".git","node_modules","__pycache__","dist","build"}
               for p in fp.parts): continue
        try:
            size = fp.stat().st_size
            tokens = estimate_tokens(size)
            if used + tokens > budget:
                remaining = int((budget - used) / TOKENS_PER_BYTE)
                if remaining > 50:
                    text = fp.read_text(encoding="utf-8", errors="ignore")[:remaining]
                    chunks.append(f"\n\n--- FILE: {fp.relative_to(base)} [truncated] ---\n{text}")
                break
            text = fp.read_text(encoding="utf-8", errors="ignore")
            chunks.append(f"\n\n--- FILE: {fp.relative_to(base)} ---\n{text}")
            used += tokens
        except OSError: continue
    return "".join(chunks)

TOKENS_PER_BYTE = 0.250


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_response(response: str, ground_truth_files: list[str]) -> dict:
    import re
    if not ground_truth_files:
        return {"top1": None, "top3": None, "any_match": None}
    mentioned = re.findall(r'[\w/\-\.]+\.(?:py|js|ts|go|rb|java|rs|cpp|c|h|kt|swift)', response)
    if not mentioned:
        return {"top1": False, "top3": False, "any_match": False}
    def matches(m, gt):
        m, gt = m.lower().strip("/"), gt.lower().strip("/")
        return gt.endswith(m) or m.endswith(gt) or m in gt or gt in m
    top1 = any(matches(mentioned[0], gt) for gt in ground_truth_files)
    top3 = any(matches(m, gt) for m in mentioned[:3] for gt in ground_truth_files)
    any_m = any(matches(m, gt) for m in mentioned for gt in ground_truth_files)
    return {"top1": top1, "top3": top3, "any_match": any_m}


# ── Task loading ──────────────────────────────────────────────────────────────

def load_tasks(limit: Optional[int] = None) -> list[dict]:
    task_file = REPO_ROOT / "experiments" / "tasks" / "task_definitions.json"
    if not task_file.exists():
        print(f"Task file not found: {task_file}")
        return _demo_tasks()[:limit or 99]
    with open(task_file) as f:
        data = json.load(f)
    tasks = []
    for repo_key, repo_tasks in data.items():
        if repo_key.startswith("_"): continue
        for t in repo_tasks:
            tasks.append({
                "repo": repo_key,
                "id": t.get("id", "?"),
                "type": t.get("type", "?"),
                "query": t.get("query", ""),
                "ground_truth_file": t.get("groundTruth", {}).get("file", ""),
            })
    return tasks[:limit] if limit else tasks

def _demo_tasks() -> list[dict]:
    return [
        {"repo": "fastapi_py", "id": "F-CR-1", "type": "code_retrieval",
         "query": "Which function handles HTTP dependency injection?",
         "ground_truth_file": "fastapi/dependencies/utils.py"},
        {"repo": "express_js", "id": "E-CR-1", "type": "code_retrieval",
         "query": "Which function handles HTTP request routing?",
         "ground_truth_file": "lib/router/index.js"},
        {"repo": "fastapi_py", "id": "F-BL-1", "type": "bug_localization",
         "query": "Pydantic ValidationError not converted to HTTP 422 in edge cases.",
         "ground_truth_file": "fastapi/exception_handlers.py"},
    ]


# ── Main experiment loop ──────────────────────────────────────────────────────

def build_adapters(
    model_args: list[str],
    openrouter_model: str,
    ollama_models: list[str] | None = None,
) -> list[BaseAdapter]:
    all_adapters = {
        "grok":            GrokAdapter("grok-3-mini"),
        "grok-3":          GrokAdapter("grok-3"),
        "grok-mini":       GrokAdapter("grok-3-mini"),
        "gemini":          GeminiAdapter("gemini-2.0-flash"),
        "gemini-flash":    GeminiAdapter("gemini-2.0-flash"),
        "gemini-pro":      GeminiAdapter("gemini-1.5-pro"),
        "gemini-2.5-pro":  GeminiAdapter("gemini-2.5-pro"),
        "openrouter":      OpenRouterAdapter(openrouter_model),
        "openai":          OpenAIAdapter("gpt-4o-mini"),
        "gpt-4o-mini":     OpenAIAdapter("gpt-4o-mini"),
        "gpt-4o":          OpenAIAdapter("gpt-4o"),
    }
    ollama_selected = [OllamaAdapter(model) for model in (ollama_models or ["codellama:7b-instruct"])]

    if not model_args or model_args == ["all"]:
        # Use all available
        selected = list(all_adapters.values())
        available = [a for a in selected if a.is_available()]
        if shutil.which("ollama"):
            available.extend(ollama_selected)
        if not available:
            print("WARN No API keys found. Running dry run.")
            return list({a.model_id: a for a in selected + ollama_selected}.values())
        return list({a.model_id: a for a in available}.values())
    result = []
    for m in model_args:
        if m == "ollama":
            result.extend(ollama_selected)
            continue
        if m in all_adapters:
            result.append(all_adapters[m])
        else:
            print(f"WARN Unknown model alias '{m}'. Valid: {list(all_adapters)}")
    return result or [GeminiAdapter()]


def run_experiment(args: argparse.Namespace) -> None:
    ts = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H-%M-%S")

    print("\n" + "="*65)
    print("  CORTEX v2 — Multi-Model Experiment Runner")
    print("="*65)

    adapters = build_adapters(
        getattr(args, "models", []),
        getattr(args, "openrouter_model", "meta-llama/llama-3.1-8b-instruct"),
        getattr(args, "ollama_models", None),
    )
    tasks = load_tasks(args.limit)

    print(f"\n  Models    : {', '.join(a.model_id for a in adapters)}")
    print(f"  Tasks     : {len(tasks)}")
    print(f"  Budget    : {CONTEXT_BUDGET:,} tokens/call")
    print(f"  Dry run   : {args.dry_run}")
    print(f"  Repos dir : {REPOS_DIR}")
    print()

    # Define filter conditions
    conditions: dict[str, callable] = {
        "A_no_filter":         lambda repo, q: build_context(repo, CONTEXT_BUDGET, "none"),
        "B_naive_truncation":  lambda repo, q: build_naive_truncation(repo, CONTEXT_BUDGET),
        "C_bm25":              lambda repo, q: BM25FileSelector(query=q, token_budget=CONTEXT_BUDGET).scan(repo).__class__.__name__,  # placeholder
        "D_size_1mb":          lambda repo, q: build_context(repo, CONTEXT_BUDGET, "size"),
        "E_hybrid":            lambda repo, q: build_context(repo, CONTEXT_BUDGET, "hybrid"),
    }

    # Proper BM25 context builder
    def bm25_context(repo, query):
        try:
            f = BM25FileSelector(query=query, token_budget=CONTEXT_BUDGET)
            result = f.scan(repo)
            base = Path(repo)
            chunks = []
            for fi in sorted(result.allowed_files, key=lambda x: x.tokens):
                try:
                    text = Path(fi.path).read_text(encoding="utf-8", errors="ignore")
                    try: rel = str(Path(fi.path).relative_to(base))
                    except ValueError: rel = fi.path
                    chunks.append(f"\n\n--- FILE: {rel} ---\n{text}")
                except OSError: pass
            return "".join(chunks)
        except Exception:
            return build_context(repo, CONTEXT_BUDGET, "size")

    conditions["C_bm25"] = bm25_context

    if not args.all_conditions:
        conditions = {k: v for k, v in conditions.items()
                      if k in ("A_no_filter", "B_naive_truncation", "E_hybrid")}

    # Results accumulator: {model_id: {condition: {metric: [values]}}}
    all_results: dict = {a.model_id: {c: {"top1": [], "top3": [], "costs": [],
                                           "tokens": [], "latencies": [], "errors": []}
                                       for c in conditions}
                         for a in adapters}

    total_calls = len(tasks) * len(adapters) * len(conditions)
    call_num = 0

    for task in tasks:
        repo_path = REPOS_DIR / task["repo"]
        query     = task["query"]
        gt_file   = [task["ground_truth_file"]] if task["ground_truth_file"] else []
        task_id   = task["id"]

        print(f"Task {task_id} ({task['repo']}): {query[:55]}...")

        if not repo_path.exists():
            print(f"  WARN Repo not found: {repo_path} - skipping")
            print(f"  -> Run: node experiments/scripts/clone_repos.js")
            continue

        for cond_name, ctx_builder in conditions.items():
            # Build context once, reuse across all models
            try:
                context = ctx_builder(str(repo_path), query)
            except Exception as e:
                context = f"[Context build failed: {e}]"

            ctx_tokens = estimate_tokens(len(context))
            system_prompt = (
                f"You are an expert software engineer. The following is the content "
                f"of a code repository, filtered and prepared for your context window.\n\n{context}"
            )
            user_msg = (
                f"Task: {query}\n\n"
                f"Identify the specific file(s) most relevant to this task. "
                f"State the file path(s) clearly on the first line of your response."
            )

            for adapter in adapters:
                call_num += 1
                llm_result = adapter.call(system_prompt, user_msg, args.dry_run)
                scores = score_response(llm_result["response"], gt_file)

                r = all_results[adapter.model_id][cond_name]
                r["top1"].append(int(scores["top1"] or 0))
                r["top3"].append(int(scores["top3"] or 0))
                r["costs"].append(llm_result["cost_usd"])
                r["tokens"].append(llm_result["input_tokens"] or ctx_tokens)
                r["latencies"].append(llm_result["latency_ms"])
                r["errors"].append(1 if llm_result.get("error") else 0)

                status = "OK" if scores["top1"] else ("?" if scores["top1"] is None else "ERR")
                err    = f" ERR:{llm_result['error'][:30]}" if llm_result.get("error") else ""
                print(f"  [{call_num:3d}/{total_calls}] {adapter.model_id:25s} "
                      f"{cond_name:20s} {status} "
                      f"{llm_result['input_tokens']:5,}tok "
                      f"${llm_result['cost_usd']:.4f}{err}")
        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print(f"{'Model':25s} {'Condition':20s} {'Top1%':>7} {'Top3%':>7} "
          f"{'AvgTok':>8} {'TotalCost':>11} {'Errors':>7}")
    print("-"*80)

    summary_rows = []
    for model_id, cond_results in all_results.items():
        for cond, r in cond_results.items():
            n = len(r["top1"])
            if n == 0: continue
            top1       = 100 * sum(r["top1"]) / n
            top3       = 100 * sum(r["top3"]) / n
            avg_t      = sum(r["tokens"]) / n
            total_cost = sum(r["costs"])
            errors     = sum(r["errors"])
            print(f"{model_id:25s} {cond:20s} {top1:6.1f}% {top3:6.1f}% "
                  f"{avg_t:8,.0f} ${total_cost:10.4f} {errors:7d}")
            summary_rows.append({
                "model": model_id, "condition": cond, "n": n,
                "top1_pct": round(top1, 1), "top3_pct": round(top3, 1),
                "avg_tokens": round(avg_t), "total_cost_usd": round(total_cost, 4),
                "error_count": errors,
            })

    print("="*80)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    json_path = OUT_DIR / f"multimodel_results_{ts}.json"
    with open(json_path, "w") as f:
        json.dump({
            "metadata": {
                "timestamp": ts,
                "models": [a.model_id for a in adapters],
                "conditions": list(conditions.keys()),
                "n_tasks": len(tasks),
                "token_budget": CONTEXT_BUDGET,
                "dry_run": args.dry_run,
                "repos_dir": str(REPOS_DIR),
            },
            "summary": summary_rows,
            "raw": all_results,
        }, f, indent=2)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = OUT_DIR / f"multimodel_summary_{ts}.csv"
    if summary_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)

    total_cost = sum(r["total_cost_usd"] for r in summary_rows)
    print(f"\n  Total API cost: ${total_cost:.4f}")
    print(f"  Results: {json_path}")
    print(f"  CSV:     {csv_path}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CORTEX v2 — Multi-Model Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Free dry run — no API calls
  python run_multimodel.py --dry-run --limit 5

  python run_multimodel.py --models gemini-flash --limit 18

  # Grok + Gemini comparison
  python run_multimodel.py --models grok gemini-flash --limit 18

  # All available models (uses whichever keys are set)
  python run_multimodel.py --limit 18

  # Full run, all conditions, all models
  python run_multimodel.py --all-conditions
        """
    )
    parser.add_argument("--dry-run",        action="store_true",
                        help="Run without making API calls (free)")
    parser.add_argument("--limit",          type=int, default=None,
                        help="Max tasks to run (default: all 18)")
    parser.add_argument("--models",         nargs="+",
                        default=["all"],
                        choices=["all","grok","grok-3","grok-mini",
                                 "gemini","gemini-flash","gemini-pro","gemini-2.5-pro",
                                 "openrouter","openai","gpt-4o-mini","gpt-4o","ollama"],
                        help="Which model(s) to use")
    parser.add_argument("--openrouter-model", default="meta-llama/llama-3.1-8b-instruct",
                        help="OpenRouter model string (default: Llama 3.1 8B)")
    parser.add_argument("--ollama-models", nargs="+", default=None,
                        help="One or more Ollama model names to evaluate")
    parser.add_argument("--all-conditions", action="store_true",
                        help="Run all 5 conditions (default: 3 core)")
    parser.add_argument("--repos-dir",      default=None,
                        help="Path to cloned repos (default: experiments/repos)")
    args = parser.parse_args()

    if args.repos_dir:
        global REPOS_DIR
        REPOS_DIR = Path(args.repos_dir)

    # Print key availability
    print("\nAPI Keys detected:")
    for name, env in [("OpenRouter","OPENROUTER_API_KEY"),("Grok","GROK_API_KEY"),
                      ("Gemini","GEMINI_API_KEY"),("OpenAI","OPENAI_API_KEY")]:
        status = "SET" if os.environ.get(env) else "NOT SET"
        print(f"  {name:12s}: {status}")

    run_experiment(args)


if __name__ == "__main__":
    main()