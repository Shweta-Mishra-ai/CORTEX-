"""
CORTEX — Python Filter Wrapper
================================
Pure-Python reimplementation of CORTEX's core filters.
Designed for integration with SWE-bench, HuggingFace, and
any Python-based LLM evaluation pipeline.

Install:
    pip install cortex-filter        # (when published)
    # OR from this repo:
    pip install -e .

Quick usage:
    from cortex.filter import SizeFilter, HybridFilter, run_filter

    results = run_filter("/path/to/repo", filter_name="hybrid")
    print(f"Token reduction: {results['token_reduction_pct']:.1f}%")

    # Get list of allowed files ready to feed into an LLM
    allowed_files = results['allowed_files']

Integration with SWE-bench / any agent:
    from cortex.filter import build_context

    context_text = build_context(
        repo_path="/path/to/repo",
        token_budget=128_000,
        filter_name="hybrid"
    )
    # Pass context_text directly to your LLM API call
"""

from __future__ import annotations

import os
import time
import math
import struct
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

# ── Constants (mirror src/filters/index.js) ─────────────────────────────────

TOKENS_PER_BYTE: float = 0.250   # Pearson r=0.997 across 2,688 files
DEFAULT_THRESHOLD: int = 1 * 1024 * 1024   # 1 MB

PRUNED_DIRS: set[str] = {
    "node_modules", ".git", "__pycache__", "dist", "build",
    ".venv", "venv", "vendor", "target", ".next", "coverage",
    ".pytest_cache", ".mypy_cache", "egg-info", ".tox",
}

NOISE_EXTENSIONS: set[str] = {
    ".log", ".sqlite", ".db", ".sqlite3", ".csv", ".tsv",
    ".pkl", ".pickle", ".h5", ".hdf5", ".pt", ".pth",
    ".parquet", ".arrow", ".feather", ".npy", ".npz",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z",
    ".exe", ".dll", ".so", ".dylib", ".a", ".lib",
    ".mp4", ".mp3", ".avi", ".mov", ".wav", ".flac",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp",
    ".bin", ".map", ".lock",
}

SOURCE_EXTENSIONS: set[str] = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".mjs",
    ".go", ".rb", ".rs", ".java", ".kt", ".scala",
    ".c", ".cpp", ".cc", ".h", ".hpp", ".cs",
    ".sh", ".bash", ".swift", ".php",
}

# Magic bytes for binary detection (offset, bytes)
MAGIC_BYTES: list[tuple[int, bytes]] = [
    (0, b"\x89PNG"),           # PNG
    (0, b"\xff\xd8\xff"),      # JPEG
    (0, b"GIF"),               # GIF
    (0, b"PK\x03\x04"),        # ZIP
    (0, b"\x1f\x8b"),          # GZIP
    (0, b"BZh"),               # BZIP2
    (0, b"MZ"),                # EXE/DLL
    (0, b"\x7fELF"),           # ELF binary
    (0, b"\xca\xfe\xba\xbe"),  # Java class / Mach-O
    (0, b"\x89HDF"),           # HDF5
    (0, b"PAR1"),              # Parquet
]

SEMANTIC_KEYWORDS: list[str] = [
    "def ", "class ", "function ", "const ", "let ", "var ",
    "import ", "export ", "return ", "async ", "await ",
    "module", "require(", "interface ", "struct ", "enum ",
    "public ", "private ", "protected ", "static ", "void ",
    "package ", "namespace ", "type ", "impl ", "fn ",
]


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class FileInfo:
    path: str
    size: int
    tokens: int


@dataclass
class FilterResult:
    filter_name: str
    repo_path: str
    total_files: int
    allowed_files: list[FileInfo]
    blocked_files: int
    total_tokens: int
    allowed_tokens: int
    blocked_tokens: int
    token_reduction_pct: float
    processing_ms: float
    overflows_128k: bool

    def summary(self) -> str:
        lines = [
            f"\nCORTEX — {self.filter_name}",
            f"Repository:      {self.repo_path}",
            f"Total files:     {self.total_files:,}",
            f"Allowed files:   {len(self.allowed_files):,}",
            f"Blocked files:   {self.blocked_files:,}",
            f"Token reduction: {self.token_reduction_pct:.1f}%",
            f"Latency:         {self.processing_ms:.2f}ms",
            f"Overflows 128K:  {'YES ⚠' if self.overflows_128k else 'No ✓'}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "filter": self.filter_name,
            "repo_path": self.repo_path,
            "total_files": self.total_files,
            "allowed_count": len(self.allowed_files),
            "blocked_files": self.blocked_files,
            "total_tokens": self.total_tokens,
            "allowed_tokens": self.allowed_tokens,
            "blocked_tokens": self.blocked_tokens,
            "token_reduction_pct": round(self.token_reduction_pct, 2),
            "processing_ms": round(self.processing_ms, 3),
            "overflows_128k": self.overflows_128k,
            "allowed_files": [
                {"path": f.path, "size": f.size, "tokens": f.tokens}
                for f in self.allowed_files
            ],
        }


# ── Token estimation ─────────────────────────────────────────────────────────

def estimate_tokens(size_bytes: int) -> int:
    """Estimate token count from file size.  Pearson r=0.997 (2,688 files)."""
    return math.ceil(size_bytes * TOKENS_PER_BYTE)


# ── Base filter ──────────────────────────────────────────────────────────────

class BaseFilter:
    """All filters inherit from this. Override `allows()` to customize."""

    name: str = "BaseFilter"

    def allows(self, file_path: str, size: int) -> bool:
        """Return True if the file should be included in context."""
        return True

    def scan(self, repo_path: str, max_depth: int = 20) -> FilterResult:
        """Walk the repo tree and return a FilterResult."""
        start = time.perf_counter()
        repo_path = str(Path(repo_path).resolve())

        total_files = 0
        total_tokens = 0
        allowed: list[FileInfo] = []
        allowed_tokens = 0

        def walk(current: str, depth: int) -> None:
            nonlocal total_files, total_tokens, allowed_tokens
            if depth > max_depth:
                return
            try:
                entries = os.scandir(current)
            except PermissionError:
                return
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    if entry.name not in PRUNED_DIRS:
                        walk(entry.path, depth + 1)
                elif entry.is_file(follow_symlinks=False):
                    try:
                        size = entry.stat().st_size
                    except OSError:
                        continue
                    tokens = estimate_tokens(size)
                    total_files += 1
                    total_tokens += tokens
                    if self.allows(entry.path, size):
                        allowed.append(FileInfo(entry.path, size, tokens))
                        allowed_tokens += tokens

        walk(repo_path, 0)
        elapsed_ms = (time.perf_counter() - start) * 1000
        reduction = (
            (total_tokens - allowed_tokens) / total_tokens * 100
            if total_tokens > 0 else 0.0
        )
        return FilterResult(
            filter_name=self.name,
            repo_path=repo_path,
            total_files=total_files,
            allowed_files=allowed,
            blocked_files=total_files - len(allowed),
            total_tokens=total_tokens,
            allowed_tokens=allowed_tokens,
            blocked_tokens=total_tokens - allowed_tokens,
            token_reduction_pct=round(reduction, 2),
            processing_ms=round(elapsed_ms, 3),
            overflows_128k=allowed_tokens > 128_000,
        )


# ── NoFilter ─────────────────────────────────────────────────────────────────

class NoFilter(BaseFilter):
    """Admits every file.  Use as baseline."""
    name = "NoFilter"

    def allows(self, file_path: str, size: int) -> bool:
        return True


# ── BinaryFilter ─────────────────────────────────────────────────────────────

class BinaryFilter(BaseFilter):
    """Blocks files whose first 8 bytes match known binary magic-byte signatures."""
    name = "BinaryFilter"

    def allows(self, file_path: str, size: int) -> bool:
        if size == 0:
            return True
        try:
            with open(file_path, "rb") as fh:
                header = fh.read(8)
            for offset, sig in MAGIC_BYTES:
                if header[offset: offset + len(sig)] == sig:
                    return False
            return True
        except OSError:
            return True


# ── ExtensionFilter ───────────────────────────────────────────────────────────

class ExtensionFilter(BaseFilter):
    """Blocks files whose extension is in the noise-extension blocklist."""
    name = "ExtensionFilter"

    def __init__(self, blocked: Optional[set[str]] = None):
        self._blocked = blocked or NOISE_EXTENSIONS

    def allows(self, file_path: str, size: int) -> bool:
        ext = Path(file_path).suffix.lower()
        return ext not in self._blocked


# ── SizeFilter ★ ─────────────────────────────────────────────────────────────

class SizeFilter(BaseFilter):
    """
    The proposed method.  Single integer comparison — zero file reads.

    Args:
        threshold_bytes: Files larger than this are blocked.
                         Default = 1 MB (recommended from paper).
    """

    def __init__(self, threshold_bytes: int = DEFAULT_THRESHOLD):
        self.threshold = threshold_bytes
        kb = threshold_bytes / 1024
        label = f"{kb/1024:.0f}MB" if kb >= 1024 else f"{kb:.0f}KB"
        self.name = f"SizeFilter({label})"

    def allows(self, file_path: str, size: int) -> bool:
        return size <= self.threshold


# ── EntropyFilter ─────────────────────────────────────────────────────────────

class EntropyFilter(BaseFilter):
    """
    Blocks high-entropy (compressed / encrypted / binary) files.
    Shannon entropy of first 256 bytes.  Threshold = 7.0 bits/byte.
    Advantage: catches TFRecord, Parquet, Arrow, ONNX — no magic-byte table needed.
    """
    name = "EntropyFilter"

    def __init__(self, entropy_threshold: float = 7.0, sample_bytes: int = 256):
        self._threshold = entropy_threshold
        self._sample = sample_bytes

    def allows(self, file_path: str, size: int) -> bool:
        if size == 0:
            return True
        try:
            n = min(size, self._sample)
            with open(file_path, "rb") as fh:
                data = fh.read(n)
            freq = [0] * 256
            for b in data:
                freq[b] += 1
            H = 0.0
            for count in freq:
                if count == 0:
                    continue
                p = count / len(data)
                H -= p * math.log2(p)
            return H <= self._threshold
        except OSError:
            return True


# ── SemanticFilter ────────────────────────────────────────────────────────────

class SemanticFilter(BaseFilter):
    """
    Reads first 4 KB of each file and checks for programming-language keywords.
    Blocks files with fewer than min_keywords matches.
    """
    name = "SemanticFilter"

    def __init__(self, read_bytes: int = 4096, min_keywords: int = 3):
        self._read_bytes = read_bytes
        self._min = min_keywords

    def allows(self, file_path: str, size: int) -> bool:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
                text = fh.read(self._read_bytes)
            hits = sum(1 for kw in SEMANTIC_KEYWORDS if kw in text)
            return hits >= self._min
        except OSError:
            return True


# ── HybridFilter ✦ ────────────────────────────────────────────────────────────

class HybridFilter(BaseFilter):
    """
    Recommended filter.  Four gates in ascending I/O cost — early exit on first trigger.

    Gate 1: Binary magic-byte check  (8-byte read,   <0.01 ms)
    Gate 2: Size threshold           (zero read,      <0.01 ms)
    Gate 3: Minification check       (64 KB read,      ~1.5 ms)
    Gate 4: Semantic keyword scoring (4 KB read,       ~6.0 ms)

    Achieves 89.3% ± 9.0% token reduction — lowest variance of any filter.

    Args:
        threshold_bytes: Size gate threshold (default 1 MB).
    """

    def __init__(self, threshold_bytes: int = DEFAULT_THRESHOLD):
        self.threshold = threshold_bytes
        self._binary   = BinaryFilter()
        self._size     = SizeFilter(threshold_bytes)
        self._semantic = SemanticFilter()
        kb = threshold_bytes / 1024
        label = f"{kb/1024:.0f}MB" if kb >= 1024 else f"{kb:.0f}KB"
        self.name = f"HybridFilter({label})"

    def _is_minified(self, file_path: str, size: int) -> bool:
        try:
            n = min(size, 65_536)
            with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
                text = fh.read(n)
            lines = [l for l in text.splitlines() if l]
            if not lines:
                return False
            avg_len = sum(len(l) for l in lines) / len(lines)
            return avg_len > 500
        except OSError:
            return False

    def allows(self, file_path: str, size: int) -> bool:
        if not self._binary.allows(file_path, size):   return False
        if not self._size.allows(file_path, size):     return False
        if self._is_minified(file_path, size):         return False
        # Gate 4: semantic — min_keywords=1 because file already passed
        # 3 other gates; any code keyword is sufficient signal.
        if not SemanticFilter(min_keywords=1).allows(file_path, size): return False
        return True


# ── AdaptiveSizeFilter ────────────────────────────────────────────────────────

class AdaptiveSizeFilter(BaseFilter):
    """
    Sets threshold automatically at the P95 of the repo's file-size distribution.
    Eliminates manual threshold tuning (addresses paper Limitation L1).

    Algorithm:
      Pass 1 — collect all file sizes (stat only, no reads)
      Pass 2 — filter files > P95
    """

    def __init__(self, percentile: int = 95,
                 min_threshold: int = 50 * 1024,
                 max_threshold: int = 10 * 1024 * 1024):
        self._percentile = percentile
        self._min = min_threshold
        self._max = max_threshold
        self._theta: Optional[int] = None
        self.name = f"AdaptiveP{percentile}(pending)"

    def _compute_threshold(self, repo_path: str, max_depth: int = 20) -> int:
        sizes: list[int] = []

        def walk(d: str, depth: int) -> None:
            if depth > max_depth:
                return
            try:
                entries = os.scandir(d)
            except PermissionError:
                return
            for e in entries:
                if e.is_dir(follow_symlinks=False):
                    if e.name not in PRUNED_DIRS:
                        walk(e.path, depth + 1)
                elif e.is_file(follow_symlinks=False):
                    try:
                        sizes.append(e.stat().st_size)
                    except OSError:
                        pass

        walk(repo_path, 0)
        if not sizes:
            return DEFAULT_THRESHOLD
        sizes.sort()
        idx = max(0, math.ceil(self._percentile / 100 * len(sizes)) - 1)
        raw = sizes[idx]
        return max(self._min, min(raw, self._max))

    def scan(self, repo_path: str, max_depth: int = 20) -> FilterResult:
        self._theta = self._compute_threshold(repo_path, max_depth)
        kb = self._theta / 1024
        label = f"{kb/1024:.0f}MB" if kb >= 1024 else f"{kb:.0f}KB"
        self.name = f"AdaptiveP{self._percentile}({label})"
        return super().scan(repo_path, max_depth)

    def allows(self, file_path: str, size: int) -> bool:
        return size <= (self._theta or DEFAULT_THRESHOLD)


# ── ContextBudgetFilter ───────────────────────────────────────────────────────

class ContextBudgetFilter(BaseFilter):
    """
    Greedily admits smallest files first until the token budget is consumed.
    Directly solves the MECW constraint (Equation 1 in the paper).

    This is the theoretically optimal filter: it maximises the number
    of files in context while guaranteeing the budget is not exceeded.

    Args:
        budget_tokens: Maximum tokens to admit (default 128,000 = GPT-4 context).
    """

    def __init__(self, budget_tokens: int = 128_000):
        self._budget = budget_tokens
        self.name = f"ContextBudgetFilter({budget_tokens:,}tok)"

    def scan(self, repo_path: str, max_depth: int = 20) -> FilterResult:
        start = time.perf_counter()
        repo_path = str(Path(repo_path).resolve())

        all_files: list[FileInfo] = []

        def walk(d: str, depth: int) -> None:
            if depth > max_depth:
                return
            try:
                entries = os.scandir(d)
            except PermissionError:
                return
            for e in entries:
                if e.is_dir(follow_symlinks=False):
                    if e.name not in PRUNED_DIRS:
                        walk(e.path, depth + 1)
                elif e.is_file(follow_symlinks=False):
                    try:
                        size = e.stat().st_size
                        all_files.append(FileInfo(e.path, size, estimate_tokens(size)))
                    except OSError:
                        pass

        walk(repo_path, 0)
        all_files.sort(key=lambda f: f.tokens)   # smallest first

        used = 0
        allowed: list[FileInfo] = []
        for f in all_files:
            if used + f.tokens <= self._budget:
                allowed.append(f)
                used += f.tokens

        total_tokens = sum(f.tokens for f in all_files)
        elapsed_ms = (time.perf_counter() - start) * 1000
        reduction = (total_tokens - used) / total_tokens * 100 if total_tokens else 0

        return FilterResult(
            filter_name=self.name,
            repo_path=repo_path,
            total_files=len(all_files),
            allowed_files=allowed,
            blocked_files=len(all_files) - len(allowed),
            total_tokens=total_tokens,
            allowed_tokens=used,
            blocked_tokens=total_tokens - used,
            token_reduction_pct=round(reduction, 2),
            processing_ms=round(elapsed_ms, 3),
            overflows_128k=used > 128_000,
        )

    def allows(self, file_path: str, size: int) -> bool:
        return True  # not used; scan() overrides


# ── Factory ───────────────────────────────────────────────────────────────────

_FILTER_MAP = {
    "none":      NoFilter,
    "binary":    BinaryFilter,
    "extension": ExtensionFilter,
    "size":      SizeFilter,
    "entropy":   EntropyFilter,
    "semantic":  SemanticFilter,
    "hybrid":    HybridFilter,
    "adaptive":  AdaptiveSizeFilter,
    "budget":    ContextBudgetFilter,
}


def create_filter(name: str, threshold_bytes: int = DEFAULT_THRESHOLD,
                  budget_tokens: int = 128_000) -> BaseFilter:
    """
    Create a filter by name.

    Args:
        name: One of 'none', 'binary', 'extension', 'size', 'entropy',
              'semantic', 'hybrid', 'adaptive', 'budget'
        threshold_bytes: Used by size / hybrid / adaptive / entropy filters.
        budget_tokens: Used by budget filter.

    Returns:
        A filter instance with a .scan(repo_path) method.
    """
    name = name.lower()
    if name not in _FILTER_MAP:
        raise ValueError(f"Unknown filter '{name}'. Choose from: {list(_FILTER_MAP)}")
    if name == "budget":
        return ContextBudgetFilter(budget_tokens)
    if name in ("size", "hybrid", "entropy"):
        return _FILTER_MAP[name](threshold_bytes)
    return _FILTER_MAP[name]()


def run_filter(repo_path: str, filter_name: str = "hybrid",
               threshold_bytes: int = DEFAULT_THRESHOLD,
               budget_tokens: int = 128_000) -> dict:
    """
    One-line convenience function. Returns a plain dict.

    Example:
        result = run_filter("/my/repo")
        print(result["token_reduction_pct"])   # e.g. 89.3
        for f in result["allowed_files"]:
            print(f["path"])
    """
    f = create_filter(filter_name, threshold_bytes, budget_tokens)
    return f.scan(repo_path).to_dict()


def build_context(repo_path: str, token_budget: int = 128_000,
                  filter_name: str = "hybrid",
                  threshold_bytes: int = DEFAULT_THRESHOLD) -> str:
    """
    Build a token-safe context string from a repository.
    Designed to be passed directly to an LLM API call.

    Args:
        repo_path: Path to the git repository.
        token_budget: Maximum tokens to include (default 128K).
        filter_name: Which filter to use (default 'hybrid').
        threshold_bytes: Size gate for size-based filters.

    Returns:
        A single string concatenating the content of all allowed files,
        each prefixed with its relative path header.

    Example:
        context = build_context("/path/to/fastapi", token_budget=32_000)
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": context},
                {"role": "user",   "content": "Find the dependency injection function"}
            ]
        )
    """
    flt = create_filter(filter_name, threshold_bytes, budget_tokens=token_budget)
    result = flt.scan(repo_path)

    # Sort smallest-first to pack more files within budget
    files = sorted(result.allowed_files, key=lambda f: f.tokens)

    chunks: list[str] = []
    used_tokens = 0
    base = Path(repo_path).resolve()

    for fi in files:
        if used_tokens + fi.tokens > token_budget:
            continue
        try:
            content = Path(fi.path).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        fp = Path(fi.path).resolve()
        try:
            rel = str(fp.relative_to(base))
        except ValueError:
            rel = fi.path
        chunks.append(f"\n\n--- FILE: {rel} ---\n{content}")
        used_tokens += fi.tokens

    return "".join(chunks)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="CORTEX — Pre-execution repository filter for LLM tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m cortex.filter .
  python -m cortex.filter --filter hybrid /path/to/repo
  python -m cortex.filter --filter adaptive /path/to/repo
  python -m cortex.filter --filter budget --budget 64000 . --json
  python -m cortex.filter --filter size --threshold 524288 /path/to/repo
        """
    )
    parser.add_argument("repo", nargs="?", default=".",
                        help="Path to repository (default: current directory)")
    parser.add_argument("--filter", default="hybrid",
                        choices=list(_FILTER_MAP),
                        help="Filter strategy (default: hybrid)")
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD,
                        help="Size threshold in bytes (default: 1048576 = 1MB)")
    parser.add_argument("--budget", type=int, default=128_000,
                        help="Token budget for 'budget' filter (default: 128000)")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON instead of human-readable summary")
    parser.add_argument("--verbose", action="store_true",
                        help="List all allowed files with sizes")
    args = parser.parse_args()

    flt = create_filter(args.filter, args.threshold, args.budget)
    res = flt.scan(args.repo)

    if args.json:
        print(json.dumps(res.to_dict(), indent=2))
    else:
        print(res.summary())
        if args.verbose:
            print("\nAllowed files:")
            for f in sorted(res.allowed_files, key=lambda x: x.size, reverse=True):
                print(f"  {f.size/1024:8.1f} KB  {f.path}")
        print()
