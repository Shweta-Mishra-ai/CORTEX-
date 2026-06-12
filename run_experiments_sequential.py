"""
CORTEX — Sequential Experiment Runner
======================================
Runs all experiments in a defined sequence and collects results.

Designed for VS Code one-click execution via the Run button (F5)
or via the VS Code task: "Run All Experiments (Sequential)".

Execution order:
    1. Environment check (keys, Ollama, repos)
    2. EXP-1 : Token reduction on 10 repositories (JS, no keys required)
    3. EXP-3 : AdaptiveSizeFilter study (JS, no keys required)
    4. EXP-OLLAMA : Local model evaluation via Ollama (no keys required)
    5. EXP-GEMINI : Gemini 2.0 Flash evaluation (free tier)
    6. EXP-OPENROUTER : OpenRouter free models evaluation
    7. EXP-GROK : Grok evaluation (if key present)
    8. EXP-ADAPTIVE : Full filter comparison on all conditions
    9. Figure generation
   10. Summary report

Free models used (no cost):
    Ollama   : llama3.2:3b, codellama:7b, deepseek-coder:6.7b, phi3:mini
    Gemini   : gemini-2.0-flash (15 RPM free tier)
    OpenRouter: meta-llama/llama-3.1-8b-instruct:free
               mistralai/mistral-7b-instruct:free
               google/gemma-2-9b-it:free
    Grok     : grok-3-mini (if API key is present)

Usage:
    python run_experiments_sequential.py              # all experiments
    python run_experiments_sequential.py --quick      # 5 tasks, validates setup
    python run_experiments_sequential.py --no-ollama  # skip local models
    python run_experiments_sequential.py --exp 1 3    # run only EXP-1 and EXP-3
"""

from __future__ import annotations

import os
import sys
import json
import time
import subprocess
import datetime
import argparse
import shutil
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "experiments" / "results"
LOGS_DIR = REPO_ROOT / "experiments" / "logs"
FIGURES_DIR = REPO_ROOT / "paper" / "figures"

for d in (RESULTS_DIR, LOGS_DIR, FIGURES_DIR,
          RESULTS_DIR / "multimodel", RESULTS_DIR / "tasks",
          RESULTS_DIR / "swebench", RESULTS_DIR / "adaptive"):
    d.mkdir(parents=True, exist_ok=True)

PYTHONPATH = str(REPO_ROOT / "src" / "python")


# ---------------------------------------------------------------------------
# Environment loader
# ---------------------------------------------------------------------------

def load_env() -> None:
    """Load .env file into os.environ without overriding existing variables."""
    env_path = REPO_ROOT / ".env"
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


load_env()


# ---------------------------------------------------------------------------
# Terminal colours
# ---------------------------------------------------------------------------

C = {
    "green":  "\033[92m",
    "yellow": "\033[93m",
    "red":    "\033[91m",
    "blue":   "\033[94m",
    "cyan":   "\033[96m",
    "bold":   "\033[1m",
    "reset":  "\033[0m",
}


def _c(color: str, text: str) -> str:
    return f"{C[color]}{text}{C['reset']}"


def header(title: str) -> None:
    print(f"\n{C['bold']}{C['cyan']}{'═' * 60}{C['reset']}")
    print(f"{C['bold']}{C['cyan']}  {title}{C['reset']}")
    print(f"{C['bold']}{C['cyan']}{'═' * 60}{C['reset']}\n")


def log_ok(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"  {_c('green', '✓')} [{ts}] {msg}")


def log_warn(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"  {_c('yellow', '⚠')} [{ts}] {msg}")


def log_err(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"  {_c('red', '✗')} [{ts}] {msg}")


def log_step(n: int, total: int, msg: str) -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"\n{_c('bold', f'[{n}/{total}]')} [{ts}] {_c('blue', msg)}")


# ---------------------------------------------------------------------------
# Environment check
# ---------------------------------------------------------------------------

def check_environment() -> dict[str, bool]:
    """Verify all dependencies and API key availability."""
    header("Environment Check")

    status: dict[str, bool] = {}

    # Node.js
    node = shutil.which("node")
    if node:
        ver = subprocess.run(
            ["node", "--version"], capture_output=True, text=True
        ).stdout.strip()
        major = int(ver.lstrip("v").split(".")[0])
        if major >= 22:
            log_ok(f"Node.js {ver}")
            status["node"] = True
        else:
            log_warn(f"Node.js {ver} — version 22+ required for tests")
            status["node"] = False
    else:
        log_warn("Node.js not found — JS experiments will be skipped")
        status["node"] = False

    # Python
    log_ok(f"Python {sys.version.split()[0]}")
    status["python"] = True

    # Repos
    repos_dir = REPO_ROOT / "experiments" / "repos"
    if repos_dir.exists():
        cloned = [d for d in repos_dir.iterdir() if d.is_dir()]
        if len(cloned) >= 5:
            log_ok(f"Repositories: {len(cloned)} cloned")
            status["repos"] = True
        else:
            log_warn(f"Only {len(cloned)} repos cloned — run: node experiments/scripts/clone_repos.js")
            status["repos"] = len(cloned) > 0
    else:
        log_warn("No repos directory — run: node experiments/scripts/clone_repos.js")
        status["repos"] = False

    # Ollama
    ollama = shutil.which("ollama")
    if ollama:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            models = [
                line.split()[0] for line in result.stdout.splitlines()[1:]
                if line.strip()
            ]
            log_ok(f"Ollama: {len(models)} model(s) available: {', '.join(models[:4])}")
            status["ollama"] = bool(models)
            status["ollama_models"] = models  # type: ignore[assignment]
        else:
            log_warn("Ollama installed but not running — start with: ollama serve")
            status["ollama"] = False
    else:
        log_warn("Ollama not installed — local model experiments will be skipped")
        log_warn("Install: https://ollama.com  then: ollama pull llama3.2:3b")
        status["ollama"] = False

    # API keys
    for name, env_var in [
        ("Gemini",      "GEMINI_API_KEY"),
        ("Grok",        "GROK_API_KEY"),
        ("OpenRouter",  "OPENROUTER_API_KEY"),
        ("OpenAI",      "OPENAI_API_KEY"),
    ]:
        val = os.environ.get(env_var, "")
        if val and len(val) > 10:
            masked = val[:8] + "..." + val[-4:]
            log_ok(f"{name}: {masked}")
            status[name.lower()] = True
        else:
            log_warn(f"{name}: not set  (add to .env)")
            status[name.lower()] = False

    return status


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

class Experiment:
    """Runs a single experiment and captures output."""

    def __init__(self, name: str, cmd: list[str], log_name: str) -> None:
        self.name = name
        self.cmd = cmd
        self.log_file = LOGS_DIR / log_name
        self.returncode: Optional[int] = None
        self.elapsed: float = 0.0
        self.output: str = ""

    def run(self, env: Optional[dict] = None) -> bool:
        run_env = {**os.environ, "PYTHONPATH": PYTHONPATH}
        if env:
            run_env.update(env)

        t0 = time.perf_counter()
        try:
            result = subprocess.run(
                self.cmd,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                env=run_env,
            )
            self.returncode = result.returncode
            self.output = result.stdout + result.stderr
        except Exception as exc:
            self.returncode = -1
            self.output = str(exc)

        self.elapsed = time.perf_counter() - t0
        self.log_file.write_text(
            f"CMD: {' '.join(self.cmd)}\n"
            f"RC:  {self.returncode}\n\n"
            f"{self.output}",
            encoding="utf-8",
        )
        return self.returncode == 0


# ---------------------------------------------------------------------------
# Individual experiment functions
# ---------------------------------------------------------------------------

def run_exp1_token_reduction(status: dict, quick: bool) -> bool:
    """EXP-1: Token reduction on 10 repositories."""
    if not status.get("node"):
        log_warn("Skipping EXP-1 — Node.js 22+ required")
        return False
    if not status.get("repos"):
        log_warn("Skipping EXP-1 — clone repos first: node experiments/scripts/clone_repos.js")
        return False

    exp = Experiment(
        name="EXP-1:TokenReduction",
        cmd=["node", "experiments/scripts/run_full.js"],
        log_name="exp1_token_reduction.log",
    )
    ok = exp.run()
    if ok:
        log_ok(f"EXP-1 complete in {exp.elapsed:.0f}s → {exp.log_file.name}")
    else:
        log_err(f"EXP-1 failed — see {exp.log_file}")
    return ok


def run_exp3_adaptive(status: dict) -> bool:
    """EXP-3: AdaptiveSizeFilter threshold study."""
    if not status.get("node") or not status.get("repos"):
        log_warn("Skipping EXP-3 — requires Node.js 22+ and cloned repos")
        return False

    exp = Experiment(
        name="EXP-3:Adaptive",
        cmd=["node", "experiments/scripts/run_adaptive.js"],
        log_name="exp3_adaptive.log",
    )
    ok = exp.run()
    if ok:
        log_ok(f"EXP-3 complete in {exp.elapsed:.0f}s → {exp.log_file.name}")
    else:
        log_err(f"EXP-3 failed — see {exp.log_file}")
    return ok


def run_exp_ollama(status: dict, quick: bool) -> bool:
    """EXP-OLLAMA: Evaluation with local models via Ollama."""
    if not status.get("ollama"):
        log_warn("Skipping Ollama experiment — Ollama not running")
        log_warn("  Install: https://ollama.com")
        log_warn("  Pull:    ollama pull llama3.2:3b codellama:7b phi3:mini")
        log_warn("  Start:   ollama serve")
        return False

    models: list[str] = status.get("ollama_models", [])  # type: ignore[assignment]
    preferred = ["codellama:7b-instruct", "llama3.2:3b", "phi3:mini",
                 "deepseek-coder:6.7b", "gemma2:2b"]
    available = [m for m in preferred if any(m.split(":")[0] in om for om in models)]

    if not available:
        log_warn(f"No preferred models found. Available: {models}")
        log_warn("  Run: ollama pull codellama:7b  or  ollama pull llama3.2:3b")
        return False

    limit = ["--limit", "5"] if quick else []
    exp = Experiment(
        name="EXP-OLLAMA",
        cmd=[
            sys.executable,
            "experiments/scripts/run_multimodel.py",
            "--models", "ollama",
            "--ollama-models", *available[:2],
            "--all-conditions",
            *limit,
        ],
        log_name="exp_ollama.log",
    )
    ok = exp.run()
    if ok:
        log_ok(f"Ollama experiment complete in {exp.elapsed:.0f}s — models: {available[:2]}")
    else:
        log_err(f"Ollama experiment failed — see {exp.log_file}")
    return ok


def run_exp_gemini(status: dict, quick: bool) -> bool:
    """EXP-GEMINI: Evaluation with Gemini 2.0 Flash (free tier)."""
    if not status.get("gemini"):
        log_warn("Skipping Gemini — add GEMINI_API_KEY to .env")
        log_warn("  Free key: https://aistudio.google.com/app/apikey")
        return False

    limit = ["--limit", "5"] if quick else []
    exp = Experiment(
        name="EXP-GEMINI",
        cmd=[
            sys.executable,
            "experiments/scripts/run_multimodel.py",
            "--models", "gemini-flash",
            "--all-conditions",
            *limit,
        ],
        log_name="exp_gemini.log",
    )
    ok = exp.run()
    if ok:
        log_ok(f"Gemini experiment complete in {exp.elapsed:.0f}s")
    else:
        log_err(f"Gemini experiment failed — see {exp.log_file}")
    return ok


def run_exp_openrouter(status: dict, quick: bool) -> bool:
    """EXP-OPENROUTER: Evaluation with free OpenRouter models."""
    if not status.get("openrouter"):
        log_warn("Skipping OpenRouter — add OPENROUTER_API_KEY to .env")
        log_warn("  Free key: https://openrouter.ai/keys")
        return False

    # OpenRouter free models (no cost)
    free_models = [
        "meta-llama/llama-3.1-8b-instruct:free",
        "mistralai/mistral-7b-instruct:free",
        "google/gemma-2-9b-it:free",
    ]
    limit = ["--limit", "5"] if quick else []
    exp = Experiment(
        name="EXP-OPENROUTER",
        cmd=[
            sys.executable,
            "experiments/scripts/run_multimodel.py",
            "--models", "openrouter",
            "--openrouter-model", free_models[0],
            "--all-conditions",
            *limit,
        ],
        log_name="exp_openrouter.log",
    )
    ok = exp.run()
    if ok:
        log_ok(f"OpenRouter experiment complete in {exp.elapsed:.0f}s")
    else:
        log_err(f"OpenRouter experiment failed — see {exp.log_file}")
    return ok


def run_exp_grok(status: dict, quick: bool) -> bool:
    """EXP-GROK: Evaluation with Grok."""
    if not status.get("grok"):
        log_warn("Skipping Grok — add GROK_API_KEY to .env")
        return False

    limit = ["--limit", "5"] if quick else []
    exp = Experiment(
        name="EXP-GROK",
        cmd=[
            sys.executable,
            "experiments/scripts/run_multimodel.py",
            "--models", "grok-mini",
            "--all-conditions",
            *limit,
        ],
        log_name="exp_grok.log",
    )
    ok = exp.run()
    if ok:
        log_ok(f"Grok experiment complete in {exp.elapsed:.0f}s")
    else:
        log_err(f"Grok experiment failed — see {exp.log_file}")
    return ok


def run_tests() -> bool:
    """Run the full test suite before experiments."""
    header("Test Suite")

    all_pass = True

    # JS tests
    if shutil.which("node"):
        exp = Experiment(
            name="JS-Tests",
            cmd=["node", "--test",
                 "tests/filters.test.js",
                 "tests/advanced.test.js",
                 "tests/research.test.js"],
            log_name="tests_js.log",
        )
        ok = exp.run()
        if ok:
            # Extract pass count
            pass_line = [l for l in exp.output.splitlines() if "# pass" in l]
            count = pass_line[-1].split()[-1] if pass_line else "?"
            log_ok(f"JS tests: {count} passed")
        else:
            log_err("JS tests failed — see experiments/logs/tests_js.log")
            all_pass = False
    else:
        log_warn("JS tests skipped — Node.js not found")

    # Python tests
    exp2 = Experiment(
        name="Python-Tests",
        cmd=[sys.executable, "-m", "pytest",
             "tests/test_filter_python.py", "-v", "--tb=short", "-q"],
        log_name="tests_python.log",
    )
    ok2 = exp2.run()
    if ok2:
        pass_line = [l for l in exp2.output.splitlines() if "passed" in l]
        count = pass_line[-1].strip() if pass_line else "passed"
        log_ok(f"Python tests: {count}")
    else:
        log_err("Python tests failed — see experiments/logs/tests_python.log")
        all_pass = False

    # Ruff lint
    if shutil.which("ruff"):
        exp3 = Experiment(
            name="Ruff",
            cmd=["ruff", "check", "src/python/", "experiments/scripts/",
                 "run_all_experiments.py", "run_experiments_sequential.py"],
            log_name="ruff.log",
        )
        ok3 = exp3.run()
        if ok3:
            log_ok("Ruff: no issues")
        else:
            log_warn("Ruff: style issues found — see experiments/logs/ruff.log")
    else:
        log_warn("Ruff not installed — run: pip install ruff")

    return all_pass


def generate_figures() -> bool:
    """Generate all paper figures from results."""
    header("Figure Generation")

    scripts = [
        "experiments/scripts/plot_figures.py",
        "experiments/scripts/plot_multimodel_figures.py",
    ]
    all_ok = True
    for script in scripts:
        if not (REPO_ROOT / script).exists():
            continue
        exp = Experiment(
            name=f"Plot:{Path(script).stem}",
            cmd=[sys.executable, script],
            log_name=f"figures_{Path(script).stem}.log",
        )
        ok = exp.run()
        if ok:
            log_ok(f"Generated: {script}")
        else:
            log_warn(f"Figure generation failed: {script}")
            all_ok = False

    figs = list(FIGURES_DIR.glob("*.png"))
    if figs:
        log_ok(f"Figures saved to paper/figures/ ({len(figs)} files)")
    return all_ok


def write_summary(results: dict[str, bool], elapsed_total: float) -> None:
    """Write EXPERIMENT_SUMMARY.md with all results."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    result_files = {
        "Token Reduction":
            next(iter(sorted((RESULTS_DIR).glob("results_*.json"),
                             key=lambda f: f.stat().st_mtime, reverse=True)), None),
        "Multi-Model":
            next(iter(sorted((RESULTS_DIR / "multimodel").glob("*.json"),
                             key=lambda f: f.stat().st_mtime, reverse=True)), None),
        "Adaptive":
            next(iter(sorted((RESULTS_DIR).glob("adaptive_*.json"),
                             key=lambda f: f.stat().st_mtime, reverse=True)), None),
    }

    lines = [
        "# CORTEX Experiment Summary",
        f"\nGenerated: {ts}  |  Total elapsed: {elapsed_total:.0f}s\n",
        "## Experiment Status\n",
        "| Experiment | Status |",
        "|-----------|--------|",
    ]
    for name, passed in results.items():
        icon = "✅" if passed else "⚠️ skipped"
        lines.append(f"| {name} | {icon} |")

    lines += ["\n## Result Files\n"]
    for name, path in result_files.items():
        if path:
            lines.append(f"- **{name}**: `{path.relative_to(REPO_ROOT)}`")

    lines += [
        "\n## Next Steps\n",
        "1. Review results in `experiments/results/`",
        "2. Open figures in `paper/figures/`",
        "3. Run `python run_experiments_sequential.py` again "
        "after cloning more repos or adding API keys",
    ]

    summary_path = REPO_ROOT / "EXPERIMENT_SUMMARY.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    log_ok(f"Summary written → {summary_path.name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CORTEX Sequential Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run 5 tasks per experiment (validates setup without full cost)",
    )
    parser.add_argument(
        "--no-ollama", action="store_true",
        help="Skip local Ollama model experiments",
    )
    parser.add_argument(
        "--skip-tests", action="store_true",
        help="Skip test suite before experiments",
    )
    parser.add_argument(
        "--skip-figures", action="store_true",
        help="Skip figure generation after experiments",
    )
    parser.add_argument(
        "--exp", nargs="+", type=int, default=None,
        metavar="N",
        help="Run only specific experiment numbers (e.g. --exp 1 3)",
    )
    args = parser.parse_args()

    total_start = time.perf_counter()

    print(f"\n{C['bold']}{C['cyan']}"
          f"CORTEX — Sequential Experiment Runner"
          f"{C['reset']}")
    print(f"  Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Mode:    {'quick (5 tasks)' if args.quick else 'full (18 tasks)'}")
    print()

    # Step 0: Tests
    if not args.skip_tests:
        tests_ok = run_tests()
        if not tests_ok:
            print(f"\n{_c('red', 'Tests failed — fix errors before running experiments.')}")
            print("  Override with --skip-tests if intentional.\n")
            sys.exit(1)
    else:
        log_warn("Tests skipped (--skip-tests)")

    # Step 1: Environment
    header("Environment")
    status = check_environment()

    # Step 2-8: Experiments
    exp_results: dict[str, bool] = {}
    selected = set(args.exp) if args.exp else None

    experiments = [
        (1, "EXP-1: Token Reduction",
         lambda: run_exp1_token_reduction(status, args.quick)),
        (3, "EXP-3: AdaptiveSizeFilter",
         lambda: run_exp3_adaptive(status)),
        (4, "EXP-OLLAMA: Local Models",
         lambda: run_exp_ollama(status, args.quick) if not args.no_ollama else False),
        (5, "EXP-GEMINI: Gemini 2.0 Flash",
         lambda: run_exp_gemini(status, args.quick)),
        (6, "EXP-OPENROUTER: Free Models",
         lambda: run_exp_openrouter(status, args.quick)),
        (7, "EXP-GROK: Grok",
         lambda: run_exp_grok(status, args.quick)),
    ]

    total = len(experiments)
    for i, (exp_num, label, fn) in enumerate(experiments, 1):
        if selected and exp_num not in selected:
            continue
        log_step(i, total, label)
        exp_results[label] = fn()

    # Figures
    if not args.skip_figures:
        generate_figures()

    # Summary
    total_elapsed = time.perf_counter() - total_start
    header("Complete")
    write_summary(exp_results, total_elapsed)

    passed = sum(1 for v in exp_results.values() if v)
    skipped = sum(1 for v in exp_results.values() if not v)

    print(f"\n  Experiments: {_c('green', str(passed))} completed, "
          f"{_c('yellow', str(skipped))} skipped")
    print(f"  Total time:  {total_elapsed:.0f}s")
    print(f"\n  Results in:  experiments/results/")
    print(f"  Figures in:  paper/figures/")
    print(f"  Logs in:     experiments/logs/\n")


if __name__ == "__main__":
    main()
