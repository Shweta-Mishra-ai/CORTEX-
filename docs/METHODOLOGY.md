# CORTEX — Methodology

## Zero Disk I/O Testing

The core unit tests bypass the physical filesystem entirely through
constructor-level dependency injection:

```javascript
// Production:  uses real fs
const filter = new SizeFilter({ threshold: 1024 * 1024 });

// Test:  uses virtual filesystem
const filter = new SizeFilter({
  threshold: 1024 * 1024,
  _fs: mockFS({ '/repo/main.py': { size: 512 } }),
});
```

This eliminates:
- Timing non-determinism from disk cache effects
- Platform-specific filesystem behaviour
- Test suite requiring real repository data

**Result:** core, advanced, and research-filter tests run through Node's
built-in test runner and are designed to be deterministic across platforms.

---

## Token Estimation Heuristic

From Equation (2) in the paper:

```
tokens(f) ≈ k · size_bytes
k = 0.2500 tokens/byte  (Pearson r = 0.997, n = 2,688 files)
```

Validated across 10 extension categories using the `cl100k_base`
tiktoken encoder (GPT-4 production tokenizer).

The maximum error of ~5% occurs on Unicode-dense JSON files and
produces an over-permissive outcome (safe failure mode).

---

## Tail-at-Scale Structure

The core insight justifying size-based filtering:

| Repository     | % Files > 1 MB | % Bytes in those files |
|----------------|---------------|----------------------|
| tensorflow_py  | 0.5%          | 94.0%                |
| pandas_py      | 1.1%          | 80.9%                |
| express_js     | ~2%           | 85.2%                |

A single `stat.size > θ` comparison targets the dominant cost driver.

---

## Statistical Design

- **n = 10** repositories (independent observations)
- **Paired t-test** (SizeFilter vs ExtensionFilter): t(9) = 2.31, p = 0.047
- **Wilson 95% CI** for per-filter mean TRR
- **False Positive Rate** estimated via manual inspection (5 repositories)

See paper Section VI-E for full statistical grounding.
