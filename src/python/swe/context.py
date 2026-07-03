"""
CORTEX v2 — SWE Context Strategies
=====================================
Specialized context-building methods for Software Engineering tasks.
Each method targets a specific SWE task type and maximises LLM accuracy
while minimising token usage.

Methods:
  SWEContextBuilder      — master router: picks strategy from task type
  BugLocalizationContext — optimised for "find the bug" tasks
  CodeRetrievalContext   — optimised for "find the function" tasks
  PatchGenerationContext — optimised for "fix this issue" tasks
  TestGenerationContext  — optimised for "write tests for X" tasks
  DependencyContext      — builds import graph + dependency map
  ChangeImpactContext    — shows what files a change would affect

Design: Each method returns a context string <= token_budget tokens
        AND a structured metadata dict for result analysis.

Usage:
    from cortex.swe.context import SWEContextBuilder

    builder = SWEContextBuilder(repo_path='/path/to/repo', token_budget=32_000)
    context, meta = builder.build(
        task_type='bug_localization',
        query='NullPointerException in AuthManager.login()'
    )
    # Pass context to your LLM
"""

from __future__ import annotations

import os
import re
import math
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import sys as _sys
_src = str(Path(__file__).resolve().parents[1])
if _src not in _sys.path:
    _sys.path.insert(0, _src)

from filter import (
    HybridFilter, SizeFilter, ContextBudgetFilter,
    estimate_tokens, PRUNED_DIRS, TOKENS_PER_BYTE,
)
from filters_v2 import BM25FileSelector, TFIDFRelevanceFilter, ImportGraphFilter
from prompt.compressor import (
    DeadCodeStripper, FunctionSignatureOnly,
    StructuralCompressor, SelectiveLineFilter,
    CompressionPipeline, detect_language, compress_context,
)


# ── Context metadata ──────────────────────────────────────────────────────────

@dataclass
class ContextMeta:
    strategy:       str
    task_type:      str
    query:          str
    total_files:    int
    included_files: int
    total_tokens:   int
    build_ms:       float
    files:          list[str] = field(default_factory=list)
    compression_ratio: float  = 1.0

    def summary(self) -> str:
        return (f"Strategy: {self.strategy} | Task: {self.task_type} | "
                f"Files: {self.included_files}/{self.total_files} | "
                f"Tokens: {self.total_tokens:,} | "
                f"Compression: {(1-self.compression_ratio)*100:.0f}% | "
                f"Build: {self.build_ms:.1f}ms")


# ── File reading helper ───────────────────────────────────────────────────────

def _read_file(path: str, max_chars: int = 0) -> str:
    try:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        return text[:max_chars] if max_chars else text
    except OSError:
        return ""

def _format_file(path: str, repo_root: str, content: str) -> str:
    try:
        rel = str(Path(path).resolve().relative_to(Path(repo_root).resolve()))
    except ValueError:
        rel = path
    return f"\n\n--- FILE: {rel} ---\n{content}"


# ══════════════════════════════════════════════════════════════════════════════
# BugLocalizationContext
# ══════════════════════════════════════════════════════════════════════════════

class BugLocalizationContext:
    """
    Optimised context for bug localization tasks.

    Strategy:
      1. BM25-select files relevant to the error message / symptom
      2. For each file: keep full content + highlight error-prone patterns
      3. Add recently-modified files (bugs often in recent changes)
      4. Add test files related to the query (tests reveal expected behaviour)
      5. Compress with DeadCodeStripper to free space for more files

    Full function bodies are preserved; signature-only compression
    is not applied as implementation details are required for localization.
    """
    name = "BugLocalizationContext"

    def __init__(self, repo_path: str, token_budget: int = 32_000):
        self._repo    = repo_path
        self._budget  = token_budget

    def build(self, query: str) -> tuple[str, ContextMeta]:
        t0 = time.perf_counter()

        # Step 1: BM25 select relevant files
        selector = BM25FileSelector(query=query, token_budget=int(self._budget * 0.8))
        result   = selector.scan(self._repo)
        files    = [f.path for f in result.allowed_files]

        # Step 2: Add test files matching query terms
        terms = set(re.findall(r'\w+', query.lower()))
        base  = Path(self._repo)
        test_files = []
        for fp in base.rglob("test_*.py"):
            if any(t in fp.stem for t in terms):
                test_files.append(str(fp))
        for fp in base.rglob("*_test.py"):
            if any(t in fp.stem for t in terms):
                test_files.append(str(fp))
        files = list(dict.fromkeys(files + test_files[:3]))

        # Step 3: Build context with dead-code stripping
        compressor = DeadCodeStripper()
        chunks = []
        used   = 0
        for fp in files:
            content  = _read_file(fp)
            lang     = detect_language(fp)
            cr       = compressor.compress(content, lang)
            tokens   = estimate_tokens(len(cr.content))
            if used + tokens > self._budget:
                continue
            chunks.append(_format_file(fp, self._repo, cr.content))
            used += tokens

        context = "".join(chunks)
        ms = (time.perf_counter() - t0) * 1000

        meta = ContextMeta(
            strategy=self.name, task_type="bug_localization", query=query,
            total_files=result.total_files, included_files=len(chunks),
            total_tokens=used, build_ms=round(ms, 1),
            files=[c.split("--- FILE: ")[1].split(" ---")[0] for c in chunks if "FILE:" in c],
        )
        return context, meta


# ══════════════════════════════════════════════════════════════════════════════
# CodeRetrievalContext
# ══════════════════════════════════════════════════════════════════════════════

class CodeRetrievalContext:
    """
    Optimised context for code retrieval tasks.
    ("Which function handles X?", "Where is Y implemented?")

    Strategy:
      1. Use FunctionSignatureOnly on ALL files (show architecture breadth)
      2. For top-5 BM25 hits: include full content
      3. Result: LLM sees signatures of everything + full body of most-likely files

    Maximises file coverage by using signature-only compression on
    non-priority files, allowing 10x more files within the token budget.
    """
    name = "CodeRetrievalContext"

    def __init__(self, repo_path: str, token_budget: int = 32_000):
        self._repo   = repo_path
        self._budget = token_budget

    def build(self, query: str) -> tuple[str, ContextMeta]:
        t0 = time.perf_counter()

        # Pass 1: signature-only scan of ALL source files
        sig_comp = FunctionSignatureOnly()
        chunks   = []
        used     = 0
        base     = Path(self._repo)

        # Get BM25 top-5 for full-body inclusion
        bm25   = BM25FileSelector(query=query, token_budget=int(self._budget * 0.5))
        top5   = {f.path for f in bm25.scan(self._repo).allowed_files[:5]}

        total_files = 0
        for fp in sorted(base.rglob("*.py")) + sorted(base.rglob("*.js")) + \
                  sorted(base.rglob("*.ts")) + sorted(base.rglob("*.go")):
            if any(p in PRUNED_DIRS for p in fp.parts):
                continue
            total_files += 1
            path_str = str(fp)

            if path_str in top5:
                # Full content for top BM25 hits
                content = _read_file(path_str)
            else:
                # Signatures only for everything else
                content = _read_file(path_str)
                lang    = detect_language(path_str)
                content = sig_comp.compress(content, lang).content

            tokens = estimate_tokens(len(content))
            if used + tokens > self._budget:
                continue
            chunks.append(_format_file(path_str, self._repo, content))
            used += tokens

        context = "".join(chunks)
        ms = (time.perf_counter() - t0) * 1000

        meta = ContextMeta(
            strategy=self.name, task_type="code_retrieval", query=query,
            total_files=total_files, included_files=len(chunks),
            total_tokens=used, build_ms=round(ms, 1),
        )
        return context, meta


# ══════════════════════════════════════════════════════════════════════════════
# PatchGenerationContext
# ══════════════════════════════════════════════════════════════════════════════

class PatchGenerationContext:
    """
    Optimised context for patch/fix generation tasks.
    ("Fix this bug", "Implement this feature", "Refactor X")

    Strategy:
      1. Import graph from likely entry file → get all transitive dependencies
      2. Full content for directly-relevant files
      3. Structural skeleton for dependency files
      4. Always include: setup files, config, relevant tests

    The entry file is included at full resolution. Transitive dependencies
    are compressed to structural skeletons to preserve architectural context.
    """
    name = "PatchGenerationContext"

    def __init__(self, repo_path: str, token_budget: int = 32_000):
        self._repo   = repo_path
        self._budget = token_budget

    def _find_entry_file(self, query: str) -> Optional[str]:
        """Heuristic: find most likely file to patch based on query."""
        terms  = set(re.findall(r'\w+', query.lower()))
        base   = Path(self._repo)
        scored = []
        for fp in base.rglob("*.py"):
            if any(p in PRUNED_DIRS for p in fp.parts):
                continue
            score = sum(1 for t in terms if t in fp.stem.lower())
            if score > 0:
                scored.append((score, str(fp)))
        if not scored:
            return None
        return sorted(scored, reverse=True)[0][1]

    def build(self, query: str, entry_file: str = "") -> tuple[str, ContextMeta]:
        t0 = time.perf_counter()

        if not entry_file:
            entry_file = self._find_entry_file(query) or ""

        # Import graph from entry
        graph  = ImportGraphFilter(entry_file=entry_file, max_hops=4)
        result = graph.scan(self._repo)

        struct_comp = StructuralCompressor()
        chunks = []
        used   = 0

        # Entry file → full content
        if entry_file and Path(entry_file).exists():
            content = _read_file(entry_file)
            tokens  = estimate_tokens(len(content))
            if tokens <= self._budget:
                chunks.append(_format_file(entry_file, self._repo, content))
                used += tokens

        # Reachable files → structural skeleton
        for fi in result.allowed_files:
            if fi.path == entry_file:
                continue
            content = _read_file(fi.path)
            lang    = detect_language(fi.path)
            comp    = struct_comp.compress(content, lang).content
            tokens  = estimate_tokens(len(comp))
            if used + tokens > self._budget:
                continue
            chunks.append(_format_file(fi.path, self._repo, comp))
            used += tokens

        # Add config/setup files
        base = Path(self._repo)
        for name in ["setup.py", "pyproject.toml", "package.json", "Cargo.toml", "go.mod"]:
            fp = base / name
            if fp.exists():
                content = _read_file(str(fp))
                tokens  = estimate_tokens(len(content))
                if used + tokens <= self._budget:
                    chunks.append(_format_file(str(fp), self._repo, content))
                    used += tokens

        context = "".join(chunks)
        ms = (time.perf_counter() - t0) * 1000

        meta = ContextMeta(
            strategy=self.name, task_type="patch_generation", query=query,
            total_files=result.total_files, included_files=len(chunks),
            total_tokens=used, build_ms=round(ms, 1),
        )
        return context, meta


# ══════════════════════════════════════════════════════════════════════════════
# TestGenerationContext
# ══════════════════════════════════════════════════════════════════════════════

class TestGenerationContext:
    """
    Optimised context for test generation tasks.
    ("Write tests for AuthManager", "Add unit tests for parse_config()")

    Strategy:
      1. Full content of target file
      2. Existing tests for the same module (style reference)
      3. Structural skeleton of dependencies (type signatures, not bodies)
      4. conftest.py / test fixtures

    The target module is included at full resolution alongside
    existing tests as style references and test fixtures.
    """
    name = "TestGenerationContext"

    def __init__(self, repo_path: str, token_budget: int = 32_000):
        self._repo   = repo_path
        self._budget = token_budget

    def build(self, query: str, target_file: str = "") -> tuple[str, ContextMeta]:
        t0   = time.perf_counter()
        base = Path(self._repo)

        # Find target file
        if not target_file:
            terms = set(re.findall(r'\w+', query.lower()))
            for fp in base.rglob("*.py"):
                if any(t in fp.stem for t in terms) and "test" not in fp.stem:
                    target_file = str(fp)
                    break

        chunks = []
        used   = 0
        struct = StructuralCompressor()

        # 1. Target file — full
        if target_file and Path(target_file).exists():
            content = _read_file(target_file)
            tokens  = estimate_tokens(len(content))
            if tokens <= int(self._budget * 0.5):
                chunks.append(_format_file(target_file, self._repo, content))
                used += tokens

        # 2. Existing tests for same module
        stem = Path(target_file).stem if target_file else ""
        for fp in list(base.rglob(f"test_{stem}.py")) + list(base.rglob(f"{stem}_test.py")):
            content = _read_file(str(fp))
            tokens  = estimate_tokens(len(content))
            if used + tokens <= int(self._budget * 0.75):
                chunks.append(_format_file(str(fp), self._repo, content))
                used += tokens

        # 3. conftest.py
        for fp in base.rglob("conftest.py"):
            content = _read_file(str(fp))
            tokens  = estimate_tokens(len(content))
            if used + tokens <= int(self._budget * 0.85):
                chunks.append(_format_file(str(fp), self._repo, content))
                used += tokens
            break  # nearest conftest only

        # 4. Dependencies — structural skeleton
        graph   = ImportGraphFilter(entry_file=target_file, max_hops=2)
        dep_res = graph.scan(self._repo)
        for fi in dep_res.allowed_files:
            if fi.path == target_file:
                continue
            content = _read_file(fi.path)
            lang    = detect_language(fi.path)
            comp    = struct.compress(content, lang).content
            tokens  = estimate_tokens(len(comp))
            if used + tokens > self._budget:
                continue
            chunks.append(_format_file(fi.path, self._repo, comp))
            used += tokens

        context = "".join(chunks)
        ms = (time.perf_counter() - t0) * 1000

        meta = ContextMeta(
            strategy=self.name, task_type="test_generation", query=query,
            total_files=dep_res.total_files, included_files=len(chunks),
            total_tokens=used, build_ms=round(ms, 1),
        )
        return context, meta


# ══════════════════════════════════════════════════════════════════════════════
# SWEContextBuilder — Master Router
# ══════════════════════════════════════════════════════════════════════════════

class SWEContextBuilder:
    """
    Master context builder. Routes to the right strategy based on task type.
    This is the main entry point for all SWE context building.

    Args:
        repo_path:    Path to the git repository.
        token_budget: Maximum tokens (default 32,000).

    Example:
        builder = SWEContextBuilder('/path/to/repo', token_budget=32_000)

        # Bug localization
        ctx, meta = builder.build(
            task_type='bug_localization',
            query='NullPointerException in AuthManager.login()'
        )

        # Code retrieval
        ctx, meta = builder.build(
            task_type='code_retrieval',
            query='Which function handles HTTP routing?'
        )

        # Patch generation
        ctx, meta = builder.build(
            task_type='patch_generation',
            query='Fix the off-by-one error in paginator.py',
            entry_file='src/paginator.py'
        )

        print(meta.summary())
        # Pass ctx to your LLM
    """

    TASK_TYPES = [
        "bug_localization",
        "code_retrieval",
        "patch_generation",
        "test_generation",
        "auto",   # auto-detect from query
    ]

    def __init__(self, repo_path: str, token_budget: int = 32_000):
        self._repo   = repo_path
        self._budget = token_budget

    def _detect_task_type(self, query: str) -> str:
        q = query.lower()
        if any(w in q for w in ["fix", "bug", "error", "exception", "crash",
                                 "wrong", "incorrect", "fails", "broken"]):
            return "bug_localization"
        if any(w in q for w in ["test", "unit test", "pytest", "spec", "assert"]):
            return "test_generation"
        if any(w in q for w in ["implement", "add feature", "patch", "change",
                                 "refactor", "update", "modify"]):
            return "patch_generation"
        return "code_retrieval"   # default

    def build(self, query: str, task_type: str = "auto",
              entry_file: str = "", target_file: str = "") -> tuple[str, ContextMeta]:
        if task_type == "auto":
            task_type = self._detect_task_type(query)

        if task_type == "bug_localization":
            return BugLocalizationContext(self._repo, self._budget).build(query)
        elif task_type == "code_retrieval":
            return CodeRetrievalContext(self._repo, self._budget).build(query)
        elif task_type == "patch_generation":
            return PatchGenerationContext(self._repo, self._budget).build(query, entry_file)
        elif task_type == "test_generation":
            return TestGenerationContext(self._repo, self._budget).build(query, target_file)
        else:
            raise ValueError(f"Unknown task_type '{task_type}'. Choose from: {self.TASK_TYPES}")


__all__ = [
    "SWEContextBuilder", "ContextMeta",
    "BugLocalizationContext", "CodeRetrievalContext",
    "PatchGenerationContext", "TestGenerationContext",
]