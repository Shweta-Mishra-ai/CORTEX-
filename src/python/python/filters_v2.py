"""
CORTEX — Extended Filter Suite
================================
Six query-aware and structure-aware filters extending the core filter set.

All filters inherit from BaseFilter and implement the allows() interface,
preserving Zero Disk I/O compatibility where possible.

Classes
-------
BM25FileSelector
    Scores files by BM25 relevance to a task query.
TFIDFRelevanceFilter
    Ranks files by TF-IDF cosine similarity to a task query.
LanguageAwareFilter
    Keyword-based filter covering 15 programming languages.
CompositeFilter
    Combines multiple filters with AND or OR logic.
GitHistoryFilter
    Admits files modified within a configurable recency window.
ImportGraphFilter
    Admits only files statically reachable from an entry point.
"""

from __future__ import annotations

import os
import re
import math
import time
from pathlib import Path
from typing import Optional

import sys as _sys
_src = str(Path(__file__).parent)
if _src not in _sys.path:
    _sys.path.insert(0, _src)

from filter import (
    BaseFilter, FilterResult, FileInfo,
    estimate_tokens, PRUNED_DIRS, NOISE_EXTENSIONS,
    DEFAULT_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Language keyword tables
# ---------------------------------------------------------------------------

LANGUAGE_KEYWORDS: dict[str, list[str]] = {
    "python":     ["def ", "class ", "import ", "return ", "async ", "yield "],
    "javascript": ["function ", "const ", "let ", "var ", "import ", "export ", "require("],
    "typescript": ["interface ", "type ", "enum ", "const ", "export ", "import "],
    "rust":       ["fn ", "impl ", "trait ", "pub ", "use ", "struct ", "enum ", "mod "],
    "go":         ["func ", "package ", "import ", "type ", "var ", "const ", "defer "],
    "ruby":       ["def ", "class ", "module ", "require ", "attr_", "end\n"],
    "java":       ["public ", "private ", "class ", "interface ", "import ", "void ", "@Override"],
    "kotlin":     ["fun ", "class ", "object ", "interface ", "val ", "var ", "companion "],
    "swift":      ["func ", "class ", "struct ", "protocol ", "import ", "var ", "let "],
    "cpp":        ["#include", "namespace ", "class ", "template<", "void ", "int main"],
    "csharp":     ["namespace ", "class ", "public ", "private ", "using ", "void ", "async "],
    "scala":      ["def ", "class ", "object ", "trait ", "import ", "val ", "case class"],
    "elixir":     ["def ", "defmodule ", "defp ", "use ", "import ", "alias ", "|>"],
    "haskell":    ["module ", "import ", "where", "let ", "do\n", "data ", "type "],
    "r":          ["function(", "library(", "<-", "return(", "if (", "for ("],
}

EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python", ".pyw": "python",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".rs": "rust",
    ".go": "go",
    ".rb": "ruby",
    ".java": "java",
    ".kt": "kotlin", ".kts": "kotlin",
    ".swift": "swift",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp",
    ".c": "cpp", ".h": "cpp", ".hpp": "cpp",
    ".cs": "csharp",
    ".scala": "scala",
    ".ex": "elixir", ".exs": "elixir",
    ".hs": "haskell", ".lhs": "haskell",
    ".r": "r", ".R": "r",
}


# ---------------------------------------------------------------------------
# BM25FileSelector
# ---------------------------------------------------------------------------

class BM25FileSelector(BaseFilter):
    """Selects files by BM25 relevance score relative to a natural-language query.

    Parameters
    ----------
    query : str
        Natural-language description of the task or issue.
    token_budget : int
        Maximum tokens to admit. Default: 128 000.
    k1 : float
        BM25 term-frequency saturation parameter. Default: 1.5.
    b : float
        BM25 length-normalisation parameter. Default: 0.75.
    preview_bytes : int
        Bytes read from each file to build the document text. Default: 300.
    """

    name = "BM25FileSelector"

    def __init__(
        self,
        query: str = "",
        token_budget: int = 128_000,
        k1: float = 1.5,
        b: float = 0.75,
        preview_bytes: int = 300,
    ) -> None:
        super().__init__()
        self._query = query
        self._budget = token_budget
        self._k1 = k1
        self._b = b
        self._preview = preview_bytes
        self._query_terms = set(re.findall(r"\w+", query.lower()))
        self.name = f"BM25FileSelector(budget={token_budget:,})"

    def _read_preview(self, path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                return fh.read(self._preview)
        except OSError:
            return ""

    def _build_doc(self, file_path: str) -> str:
        name = Path(file_path).stem.replace("_", " ").replace("-", " ")
        return f"{name} {self._read_preview(file_path)}"

    def _score(self, doc: str, doc_len: float, avg_dl: float,
               df: dict[str, int], n: int) -> float:
        words = doc.lower().split()
        tf: dict[str, int] = {}
        for w in words:
            tf[w] = tf.get(w, 0) + 1
        score = 0.0
        k1, b = self._k1, self._b
        for term in self._query_terms:
            f = tf.get(term, 0)
            if not f:
                continue
            idf = math.log(
                (n - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5) + 1
            )
            score += idf * (f * (k1 + 1)) / (
                f + k1 * (1 - b + b * doc_len / avg_dl)
            )
        return score

    def scan(self, repo_path: str, max_depth: int = 20) -> FilterResult:
        start = time.perf_counter()
        repo_path = str(Path(repo_path).resolve())

        candidates: list[tuple[str, int]] = []

        def walk(d: str, depth: int) -> None:
            if depth > max_depth:
                return
            try:
                entries = list(os.scandir(d))
            except PermissionError:
                return
            for e in entries:
                if e.is_dir(follow_symlinks=False):
                    if e.name not in PRUNED_DIRS:
                        walk(e.path, depth + 1)
                elif e.is_file(follow_symlinks=False):
                    try:
                        candidates.append((e.path, e.stat().st_size))
                    except OSError:
                        pass

        walk(repo_path, 0)
        total_tokens = sum(estimate_tokens(s) for _, s in candidates)

        if not self._query_terms or not candidates:
            candidates.sort(key=lambda x: x[1])
            used, allowed = 0, []
            for p, s in candidates:
                t = estimate_tokens(s)
                if used + t <= self._budget:
                    allowed.append(FileInfo(p, s, t))
                    used += t
        else:
            docs = [(p, s, self._build_doc(p)) for p, s in candidates]
            avg_dl = sum(len(d.split()) for _, _, d in docs) / max(len(docs), 1)
            df: dict[str, int] = {}
            for _, _, d in docs:
                for term in set(d.lower().split()):
                    df[term] = df.get(term, 0) + 1
            n = len(docs)

            scored = sorted(
                [(p, s, self._score(d, len(d.split()), avg_dl, df, n))
                 for p, s, d in docs],
                key=lambda x: x[2],
                reverse=True,
            )
            used, allowed = 0, []
            for p, s, _ in scored:
                t = estimate_tokens(s)
                if used + t <= self._budget:
                    allowed.append(FileInfo(p, s, t))
                    used += t

        elapsed_ms = (time.perf_counter() - start) * 1000
        reduction = (total_tokens - used) / total_tokens * 100 if total_tokens else 0.0

        return FilterResult(
            filter_name=self.name,
            repo_path=repo_path,
            total_files=len(candidates),
            allowed_files=allowed,
            blocked_files=len(candidates) - len(allowed),
            total_tokens=total_tokens,
            allowed_tokens=used,
            blocked_tokens=total_tokens - used,
            token_reduction_pct=round(reduction, 2),
            processing_ms=round(elapsed_ms, 3),
            overflows_128k=used > 128_000,
        )

    def allows(self, file_path: str, size: int) -> bool:
        return True


# ---------------------------------------------------------------------------
# LanguageAwareFilter
# ---------------------------------------------------------------------------

class LanguageAwareFilter(BaseFilter):
    """Keyword-based filter with per-language keyword tables for 15 languages.

    Falls back to a universal keyword set for unrecognised extensions.

    Parameters
    ----------
    min_keywords : int
        Minimum keyword occurrences required to admit a file. Default: 1.
    read_bytes : int
        Bytes read from each file for keyword matching. Default: 4096.
    extra_langs : dict, optional
        Additional ``{"extensions": {...}, "keywords": {...}}`` mappings.
    """

    name = "LanguageAwareFilter"

    UNIVERSAL_KEYWORDS = [
        "def ", "class ", "function ", "const ", "import ", "return ",
        "public ", "private ", "struct ", "interface ", "module ",
    ]

    def __init__(
        self,
        min_keywords: int = 1,
        read_bytes: int = 4096,
        extra_langs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self._min = min_keywords
        self._read = read_bytes
        self._ext_map = dict(EXTENSION_TO_LANGUAGE)
        self._kw_map = dict(LANGUAGE_KEYWORDS)
        if extra_langs:
            self._ext_map.update(extra_langs.get("extensions", {}))
            self._kw_map.update(extra_langs.get("keywords", {}))

    def allows(self, file_path: str, size: int) -> bool:
        if size == 0:
            return False
        ext = Path(file_path).suffix.lower()
        if ext in NOISE_EXTENSIONS:
            return False
        lang = self._ext_map.get(ext)
        keywords = (
            self._kw_map.get(lang, self.UNIVERSAL_KEYWORDS)
            if lang else self.UNIVERSAL_KEYWORDS
        )
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
                text = fh.read(self._read)
            return sum(1 for kw in keywords if kw in text) >= self._min
        except OSError:
            return True


# ---------------------------------------------------------------------------
# TFIDFRelevanceFilter
# ---------------------------------------------------------------------------

class TFIDFRelevanceFilter(BaseFilter):
    """Ranks files by TF-IDF cosine similarity to a task query.

    Parameters
    ----------
    query : str
        Natural-language task description.
    top_fraction : float
        Fraction of files to retain. Default: 0.3.
    token_budget : int
        Maximum tokens to admit. Default: 128 000.
    read_bytes : int
        Bytes read per file for document construction. Default: 512.
    """

    name = "TFIDFRelevanceFilter"

    def __init__(
        self,
        query: str = "",
        top_fraction: float = 0.3,
        token_budget: int = 128_000,
        read_bytes: int = 512,
    ) -> None:
        super().__init__()
        self._query = query
        self._top_frac = top_fraction
        self._budget = token_budget
        self._read = read_bytes
        self._query_terms = set(re.findall(r"\w+", query.lower()))
        self.name = f"TFIDFFilter(top={int(top_fraction * 100)}%)"

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def _read_preview(self, path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                return fh.read(self._read)
        except OSError:
            return ""

    def scan(self, repo_path: str, max_depth: int = 20) -> FilterResult:
        start = time.perf_counter()
        repo_path = str(Path(repo_path).resolve())

        candidates: list[tuple[str, int]] = []

        def walk(d: str, depth: int) -> None:
            if depth > max_depth:
                return
            try:
                entries = list(os.scandir(d))
            except PermissionError:
                return
            for e in entries:
                if e.is_dir(follow_symlinks=False):
                    if e.name not in PRUNED_DIRS:
                        walk(e.path, depth + 1)
                elif e.is_file(follow_symlinks=False):
                    try:
                        candidates.append((e.path, e.stat().st_size))
                    except OSError:
                        pass

        walk(repo_path, 0)
        total_tokens = sum(estimate_tokens(s) for _, s in candidates)

        if not self._query_terms or not candidates:
            n = max(1, int(len(candidates) * self._top_frac))
            selected = sorted(candidates, key=lambda x: x[1])[:n]
        else:
            corpus = [
                (p, s,
                 self._tokenize(Path(p).stem.replace("_", " "))
                 + self._tokenize(self._read_preview(p)))
                for p, s in candidates
            ]
            n_docs = len(corpus)
            df: dict[str, int] = {}
            for _, _, tokens in corpus:
                for term in set(tokens):
                    df[term] = df.get(term, 0) + 1
            idf = {
                t: math.log((n_docs + 1) / (df.get(t, 0) + 1)) + 1
                for t in self._query_terms
            }

            scored: list[tuple[str, int, float]] = []
            for p, s, tokens in corpus:
                tf: dict[str, int] = {}
                for t in tokens:
                    tf[t] = tf.get(t, 0) + 1
                dl = len(tokens) or 1
                score = sum(
                    (tf.get(t, 0) / dl) * idf.get(t, 0)
                    for t in self._query_terms
                )
                scored.append((p, s, score))

            n = max(1, int(len(scored) * self._top_frac))
            selected = [
                (p, s)
                for p, s, _ in sorted(scored, key=lambda x: x[2], reverse=True)[:n]
            ]

        allowed: list[FileInfo] = []
        used = 0
        for p, s in sorted(selected, key=lambda x: x[1]):
            t = estimate_tokens(s)
            if used + t <= self._budget:
                allowed.append(FileInfo(p, s, t))
                used += t

        elapsed_ms = (time.perf_counter() - start) * 1000
        reduction = (total_tokens - used) / total_tokens * 100 if total_tokens else 0.0

        return FilterResult(
            filter_name=self.name,
            repo_path=repo_path,
            total_files=len(candidates),
            allowed_files=allowed,
            blocked_files=len(candidates) - len(allowed),
            total_tokens=total_tokens,
            allowed_tokens=used,
            blocked_tokens=total_tokens - used,
            token_reduction_pct=round(reduction, 2),
            processing_ms=round(elapsed_ms, 3),
            overflows_128k=used > 128_000,
        )

    def allows(self, file_path: str, size: int) -> bool:
        return True


# ---------------------------------------------------------------------------
# CompositeFilter
# ---------------------------------------------------------------------------

class CompositeFilter(BaseFilter):
    """Combines multiple filters with AND or OR logic.

    Parameters
    ----------
    filters : list[BaseFilter]
        Filters to compose.
    mode : {"AND", "OR"}
        Logical operator. Default: "AND".

    Examples
    --------
    >>> f = CompositeFilter([SizeFilter(), LanguageAwareFilter()], mode="AND")
    >>> result = f.scan("/path/to/repo")
    """

    def __init__(self, filters: list[BaseFilter], mode: str = "AND") -> None:
        if mode.upper() not in ("AND", "OR"):
            raise ValueError("mode must be 'AND' or 'OR'")
        self._filters = filters
        self._mode = mode.upper()
        names = "+".join(f.name for f in filters)
        super().__init__()
        self.name = f"Composite({self._mode}:[{names}])"

    def allows(self, file_path: str, size: int) -> bool:
        if self._mode == "AND":
            return all(f.allows(file_path, size) for f in self._filters)
        return any(f.allows(file_path, size) for f in self._filters)


# ---------------------------------------------------------------------------
# GitHistoryFilter
# ---------------------------------------------------------------------------

class GitHistoryFilter(BaseFilter):
    """Admits files modified within a recency window according to git history.

    Requires git to be installed and the target path to be a git repository.
    Falls back to admitting all files when git is unavailable.

    Parameters
    ----------
    days : int
        Recency window in days. Default: 90.
    token_budget : int
        Maximum tokens to admit. Default: 128 000.
    fallback : bool
        Admit all files when git is unavailable. Default: True.
    """

    def __init__(
        self,
        days: int = 90,
        token_budget: int = 128_000,
        fallback: bool = True,
    ) -> None:
        super().__init__()
        self._days = days
        self._budget = token_budget
        self._fallback = fallback
        self._recent: Optional[set[str]] = None
        self.name = f"GitHistoryFilter({days}d)"

    def _get_recent_files(self, repo_path: str) -> Optional[set[str]]:
        import subprocess
        try:
            result = subprocess.run(
                ["git", "log", f"--since={self._days} days ago",
                 "--name-only", "--pretty=format:", "--diff-filter=AM"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return None
            files: set[str] = set()
            for line in result.stdout.splitlines():
                line = line.strip()
                if line:
                    files.add(str(Path(repo_path) / line))
            return files
        except Exception:
            return None

    def scan(self, repo_path: str, max_depth: int = 20) -> FilterResult:
        self._recent = self._get_recent_files(repo_path)
        return super().scan(repo_path, max_depth)

    def allows(self, file_path: str, size: int) -> bool:
        if self._recent is None:
            return True
        return file_path in self._recent


# ---------------------------------------------------------------------------
# ImportGraphFilter
# ---------------------------------------------------------------------------

class ImportGraphFilter(BaseFilter):
    """Admits only files statically reachable from an entry point via imports.

    Supports Python, JavaScript/TypeScript, Go, Rust, and Ruby.

    Parameters
    ----------
    entry_file : str
        Absolute or repo-relative path to the entry point.
    max_hops : int
        Maximum import-graph depth. Default: 5.
    token_budget : int
        Maximum tokens to admit. Default: 128 000.
    """

    IMPORT_PATTERNS: dict[str, list[re.Pattern]] = {
        ".py":  [
            re.compile(r"^from\s+([\w.]+)\s+import", re.MULTILINE),
            re.compile(r"^import\s+([\w.]+)", re.MULTILINE),
        ],
        ".js":  [
            re.compile(r"(?:import|require)\s*[(\'\"](\.\./[\w./]+)[\'\"]", re.MULTILINE),
        ],
        ".ts":  [
            re.compile(r"(?:import|require)\s*[(\'\"](\.\./[\w./]+)[\'\"]", re.MULTILINE),
        ],
        ".go":  [re.compile(r'"([\w./]+)"', re.MULTILINE)],
        ".rs":  [
            re.compile(r"mod\s+(\w+)", re.MULTILINE),
            re.compile(r"use\s+([\w:]+)", re.MULTILINE),
        ],
        ".rb":  [
            re.compile(r"require(?:_relative)?\s+[\'\"]([\w./]+)[\'\"]]", re.MULTILINE),
        ],
    }

    def __init__(
        self,
        entry_file: str = "",
        max_hops: int = 5,
        token_budget: int = 128_000,
    ) -> None:
        super().__init__()
        self._entry = entry_file
        self._hops = max_hops
        self._budget = token_budget
        self._reachable: set[str] = set()
        self.name = f"ImportGraphFilter(hops={max_hops})"

    def _resolve(
        self, imp: str, current: str, repo_path: str, ext: str
    ) -> Optional[str]:
        current_dir = Path(current).parent
        if imp.startswith("."):
            candidates = [
                current_dir / imp,
                current_dir / f"{imp}{ext}",
                current_dir / imp / f"__init__{ext}",
            ]
        else:
            imp_path = imp.replace(".", "/").replace("::", "/")
            candidates = [
                Path(repo_path) / imp_path,
                Path(repo_path) / f"{imp_path}{ext}",
                Path(repo_path) / "src" / f"{imp_path}{ext}",
            ]
        for c in candidates:
            if c.is_file():
                return str(c.resolve())
        return None

    def _extract_imports(self, file_path: str) -> list[str]:
        ext = Path(file_path).suffix.lower()
        patterns = self.IMPORT_PATTERNS.get(ext, [])
        if not patterns:
            return []
        try:
            text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
            imports: list[str] = []
            for pat in patterns:
                imports.extend(pat.findall(text))
            return imports
        except OSError:
            return []

    def _traverse(
        self, file_path: str, repo_path: str, visited: set[str], hop: int
    ) -> None:
        if hop > self._hops or file_path in visited:
            return
        visited.add(file_path)
        self._reachable.add(file_path)
        ext = Path(file_path).suffix.lower()
        for imp in self._extract_imports(file_path):
            resolved = self._resolve(imp, file_path, repo_path, ext)
            if resolved and resolved not in visited:
                self._traverse(resolved, repo_path, visited, hop + 1)

    def scan(self, repo_path: str, max_depth: int = 20) -> FilterResult:
        repo_path = str(Path(repo_path).resolve())
        self._reachable.clear()
        entry = self._entry
        if entry and not Path(entry).is_absolute():
            entry = str(Path(repo_path) / entry)
        if entry and Path(entry).is_file():
            self._traverse(entry, repo_path, set(), 0)
        return super().scan(repo_path, max_depth)

    def allows(self, file_path: str, size: int) -> bool:
        if not self._reachable:
            return True
        return str(Path(file_path).resolve()) in self._reachable


ALL_NEW_FILTERS = [
    "bm25", "tfidf", "language", "composite", "git_history", "import_graph",
]

__all__ = [
    "BM25FileSelector",
    "TFIDFRelevanceFilter",
    "LanguageAwareFilter",
    "CompositeFilter",
    "GitHistoryFilter",
    "ImportGraphFilter",
    "LANGUAGE_KEYWORDS",
    "EXTENSION_TO_LANGUAGE",
    "ALL_NEW_FILTERS",
]