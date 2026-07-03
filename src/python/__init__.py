"""
CORTEX — Context Optimization via Repository Token EXclusion
============================================================
Pre-execution repository filtering for LLM-based developer tools.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/python is on the path regardless of how the package is loaded
_src = str(Path(__file__).parent)
if _src not in sys.path:
    sys.path.insert(0, _src)

from filter import (  # noqa: E402
    SizeFilter,
    HybridFilter,
    AdaptiveSizeFilter,
    ContextBudgetFilter,
    EntropyFilter,
    SemanticFilter,
    BinaryFilter,
    ExtensionFilter,
    NoFilter,
    BaseFilter,
    FilterResult,
    FileInfo,
    create_filter,
    run_filter,
    build_context,
    estimate_tokens,
    TOKENS_PER_BYTE,
    DEFAULT_THRESHOLD,
)

__version__ = "2.0.0"
__all__ = [
    "SizeFilter", "HybridFilter", "AdaptiveSizeFilter",
    "ContextBudgetFilter", "EntropyFilter", "SemanticFilter",
    "BinaryFilter", "ExtensionFilter", "NoFilter", "BaseFilter",
    "FilterResult", "FileInfo",
    "create_filter", "run_filter", "build_context",
    "estimate_tokens", "TOKENS_PER_BYTE", "DEFAULT_THRESHOLD",
]
