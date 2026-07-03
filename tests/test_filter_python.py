"""
CORTEX Python Filter — Test Suite
===================================
Tests for src/python/filter.py

Run with:
    pytest tests/test_filter_python.py -v
    pytest tests/test_filter_python.py -v --tb=short

All tests use in-memory temp directories — no internet required.
"""

import os
import sys
import math
import tempfile
import pytest
from pathlib import Path

# Add src/python to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "python"))

from filter import (
    SizeFilter, HybridFilter, AdaptiveSizeFilter, ContextBudgetFilter,
    EntropyFilter, SemanticFilter, BinaryFilter, ExtensionFilter, NoFilter,
    create_filter, run_filter, build_context,
    estimate_tokens, TOKENS_PER_BYTE, DEFAULT_THRESHOLD,
    FilterResult, FileInfo,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_repo(layout: dict) -> tempfile.TemporaryDirectory:
    """
    Create a temp directory tree from a dict:
      { "subdir/file.py": b"content", "big.csv": b"x" * 2_000_000 }
    Returns the TemporaryDirectory — caller must keep it alive.
    """
    tmp = tempfile.TemporaryDirectory()
    for rel_path, content in layout.items():
        full = Path(tmp.name) / rel_path
        full.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            content = content.encode()
        full.write_bytes(content)
    return tmp


# ── Token estimation ──────────────────────────────────────────────────────────

class TestEstimateTokens:
    def test_zero(self):
        assert estimate_tokens(0) == 0

    def test_constant(self):
        assert estimate_tokens(4) == math.ceil(4 * TOKENS_PER_BYTE)

    def test_1mb(self):
        assert estimate_tokens(1_048_576) == math.ceil(1_048_576 * TOKENS_PER_BYTE)

    def test_rounds_up(self):
        # 1 byte → ceil(0.25) = 1
        assert estimate_tokens(1) == 1


# ── NoFilter ──────────────────────────────────────────────────────────────────

class TestNoFilter:
    def test_admits_everything(self):
        layout = {
            "main.py":   b"def hello(): pass",
            "data.csv":  b"a,b,c\n1,2,3",
            "model.pkl": b"\x80\x04\x95" + b"\x00" * 1000,
        }
        with make_repo(layout) as repo:
            result = NoFilter().scan(repo)
        assert result.total_files == 3
        assert len(result.allowed_files) == 3
        assert result.token_reduction_pct == 0.0

    def test_result_fields_populated(self):
        with make_repo({"a.py": b"x"}) as repo:
            r = NoFilter().scan(repo)
        assert r.total_files == 1
        assert r.total_tokens > 0
        assert r.processing_ms >= 0.0
        assert r.filter_name == "NoFilter"


# ── SizeFilter ────────────────────────────────────────────────────────────────

class TestSizeFilter:
    def test_blocks_large_file(self):
        layout = {
            "small.py":  b"x" * 100,
            "large.csv": b"x" * (2 * 1024 * 1024),  # 2MB
        }
        with make_repo(layout) as repo:
            result = SizeFilter(threshold_bytes=1024 * 1024).scan(repo)
        assert len(result.allowed_files) == 1
        assert result.allowed_files[0].path.endswith("small.py")

    def test_admits_file_exactly_at_threshold(self):
        threshold = 512
        with make_repo({"exact.py": b"x" * threshold}) as repo:
            result = SizeFilter(threshold_bytes=threshold).scan(repo)
        assert len(result.allowed_files) == 1

    def test_blocks_file_one_byte_over(self):
        threshold = 512
        with make_repo({"over.py": b"x" * (threshold + 1)}) as repo:
            result = SizeFilter(threshold_bytes=threshold).scan(repo)
        assert len(result.allowed_files) == 0

    def test_token_reduction_positive(self):
        layout = {
            "small.py": b"x" * 100,
            "huge.log": b"x" * (5 * 1024 * 1024),
        }
        with make_repo(layout) as repo:
            result = SizeFilter().scan(repo)
        assert result.token_reduction_pct > 0

    def test_prunes_node_modules(self):
        layout = {
            "src/app.py":                  b"x" * 100,
            "node_modules/lodash/index.js": b"x" * 100,
        }
        with make_repo(layout) as repo:
            result = SizeFilter().scan(repo)
        assert len(result.allowed_files) == 1
        assert "node_modules" not in result.allowed_files[0].path

    def test_name_contains_threshold(self):
        f = SizeFilter(threshold_bytes=1024 * 1024)
        assert "1MB" in f.name

    def test_empty_repo(self):
        with make_repo({}) as repo:
            result = SizeFilter().scan(repo)
        assert result.total_files == 0
        assert result.token_reduction_pct == 0.0


# ── BinaryFilter ──────────────────────────────────────────────────────────────

class TestBinaryFilter:
    def test_blocks_png(self):
        png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        with make_repo({"image.png": png_header}) as repo:
            result = BinaryFilter().scan(repo)
        assert len(result.allowed_files) == 0

    def test_admits_text(self):
        with make_repo({"code.py": b"def foo(): pass"}) as repo:
            result = BinaryFilter().scan(repo)
        assert len(result.allowed_files) == 1

    def test_admits_empty_file(self):
        with make_repo({"empty.py": b""}) as repo:
            result = BinaryFilter().scan(repo)
        assert len(result.allowed_files) == 1


# ── ExtensionFilter ───────────────────────────────────────────────────────────

class TestExtensionFilter:
    def test_blocks_noise_extension(self):
        with make_repo({"weights.pkl": b"data"}) as repo:
            result = ExtensionFilter().scan(repo)
        assert len(result.allowed_files) == 0

    def test_admits_source_extension(self):
        with make_repo({"app.py": b"pass"}) as repo:
            result = ExtensionFilter().scan(repo)
        assert len(result.allowed_files) == 1

    def test_case_insensitive(self):
        with make_repo({"DATA.CSV": b"a,b"}) as repo:
            result = ExtensionFilter().scan(repo)
        assert len(result.allowed_files) == 0


# ── EntropyFilter ─────────────────────────────────────────────────────────────

class TestEntropyFilter:
    def test_blocks_high_entropy(self):
        # Random-looking bytes — high entropy
        import os as _os
        high_entropy = bytes(range(256)) * 4   # uniform distribution → H ≈ 8
        with make_repo({"random.bin": high_entropy}) as repo:
            result = EntropyFilter(entropy_threshold=7.0).scan(repo)
        assert len(result.allowed_files) == 0

    def test_admits_low_entropy_text(self):
        # Repetitive text — low entropy
        low_entropy = b"hello world " * 50
        with make_repo({"text.txt": low_entropy}) as repo:
            result = EntropyFilter(entropy_threshold=7.0).scan(repo)
        assert len(result.allowed_files) == 1


# ── HybridFilter ──────────────────────────────────────────────────────────────

class TestHybridFilter:
    def test_blocks_large_binary(self):
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * (2 * 1024 * 1024)
        with make_repo({"huge_image.png": png}) as repo:
            result = HybridFilter().scan(repo)
        assert len(result.allowed_files) == 0

    def test_admits_small_source(self):
        with make_repo({"routes.py": b"from flask import Flask\napp = Flask(__name__)\n"}) as repo:
            result = HybridFilter().scan(repo)
        assert len(result.allowed_files) == 1

    def test_reduction_at_least_as_good_as_size_filter(self):
        layout = {
            "src/main.py":    b"def main(): pass",
            "data/train.csv": b"x" * (3 * 1024 * 1024),
            "model.pkl":      b"\x80\x04\x95" + b"x" * 500_000,
        }
        with make_repo(layout) as repo:
            hybrid_r = HybridFilter().scan(repo)
            size_r   = SizeFilter().scan(repo)
        # Hybrid should reduce at least as much as SizeFilter alone
        assert hybrid_r.token_reduction_pct >= size_r.token_reduction_pct - 0.01


# ── AdaptiveSizeFilter ────────────────────────────────────────────────────────

class TestAdaptiveSizeFilter:
    def test_name_updated_after_scan(self):
        layout = {f"file_{i}.py": b"x" * (i * 100) for i in range(1, 11)}
        with make_repo(layout) as repo:
            f = AdaptiveSizeFilter(percentile=95)
            result = f.scan(repo)
        assert f._theta is not None
        # Name should now contain the resolved threshold, not 'pending'
        assert "pending" not in f.name
        assert "95" in f.name

    def test_min_threshold_floor(self):
        # All tiny files — threshold should still be at min floor
        layout = {"tiny.py": b"x" * 10}
        with make_repo(layout) as repo:
            f = AdaptiveSizeFilter(percentile=95, min_threshold=50 * 1024)
            f.scan(repo)
        assert f._theta >= 50 * 1024

    def test_max_threshold_ceiling(self):
        # All massive files — should be capped at max
        layout = {"huge.pkl": b"x" * (50 * 1024 * 1024)}
        with make_repo(layout) as repo:
            f = AdaptiveSizeFilter(percentile=95, max_threshold=10 * 1024 * 1024)
            f.scan(repo)
        assert f._theta <= 10 * 1024 * 1024


# ── ContextBudgetFilter ───────────────────────────────────────────────────────

class TestContextBudgetFilter:
    def test_never_exceeds_budget(self):
        layout = {f"file_{i}.py": b"x" * (i * 1000) for i in range(1, 20)}
        budget = 1000   # very small
        with make_repo(layout) as repo:
            result = ContextBudgetFilter(budget_tokens=budget).scan(repo)
        assert result.allowed_tokens <= budget

    def test_admits_smallest_files_first(self):
        layout = {
            "tiny.py":   b"x" * 10,
            "medium.py": b"x" * 5000,
            "huge.py":   b"x" * 500_000,
        }
        budget = estimate_tokens(10) + estimate_tokens(5000)  # fits tiny + medium
        with make_repo(layout) as repo:
            result = ContextBudgetFilter(budget_tokens=budget).scan(repo)
        names = {Path(f.path).name for f in result.allowed_files}
        assert "tiny.py" in names
        assert "medium.py" in names
        assert "huge.py" not in names


# ── create_filter factory ─────────────────────────────────────────────────────

class TestCreateFilter:
    @pytest.mark.parametrize("name", [
        "none", "binary", "extension", "size",
        "entropy", "semantic", "hybrid", "adaptive", "budget",
    ])
    def test_all_names_valid(self, name):
        f = create_filter(name)
        assert f is not None

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError, match="Unknown filter"):
            create_filter("nonexistent_filter")


# ── run_filter convenience function ──────────────────────────────────────────

class TestRunFilter:
    def test_returns_dict(self):
        with make_repo({"app.py": b"pass"}) as repo:
            result = run_filter(repo)
        assert isinstance(result, dict)
        assert "token_reduction_pct" in result
        assert "allowed_files" in result

    def test_allowed_files_are_dicts(self):
        with make_repo({"app.py": b"pass", "data.csv": b"a,b"}) as repo:
            result = run_filter(repo, filter_name="extension")
        for f in result["allowed_files"]:
            assert "path" in f
            assert "size" in f
            assert "tokens" in f


# ── build_context ─────────────────────────────────────────────────────────────

class TestBuildContext:
    def test_returns_string(self):
        with make_repo({"app.py": b"def main(): pass"}) as repo:
            ctx = build_context(repo, token_budget=10_000)
        assert isinstance(ctx, str)

    def test_contains_file_header(self):
        with make_repo({"app.py": b"def main(): pass"}) as repo:
            ctx = build_context(repo, token_budget=10_000)
        assert "app.py" in ctx

    def test_respects_budget(self):
        layout = {f"file_{i}.py": b"x" * 10_000 for i in range(20)}
        budget = 500
        with make_repo(layout) as repo:
            ctx = build_context(repo, token_budget=budget)
        # Context token count should not dramatically exceed budget
        approx_tokens = len(ctx) * TOKENS_PER_BYTE
        assert approx_tokens <= budget * 2   # generous tolerance for headers


# ── FilterResult ──────────────────────────────────────────────────────────────

class TestFilterResult:
    def test_summary_is_string(self):
        with make_repo({"a.py": b"pass"}) as repo:
            result = SizeFilter().scan(repo)
        assert isinstance(result.summary(), str)

    def test_to_dict_serializable(self):
        import json
        with make_repo({"a.py": b"pass"}) as repo:
            result = SizeFilter().scan(repo)
        d = result.to_dict()
        # Should not raise
        json.dumps(d)

    def test_overflows_128k_flag(self):
        # One large text file that creates many tokens
        with make_repo({"giant.py": b"x = 1\n" * 200_000}) as repo:
            result = NoFilter().scan(repo)
        # Check flag is a boolean
        assert isinstance(result.overflows_128k, bool)
