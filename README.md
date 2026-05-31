# CORTEX — Context Optimization via Repository Token EXclusion

<div align="center">

[![Paper Source](https://img.shields.io/badge/Paper-Source-blue?style=for-the-badge)](paper/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Node.js](https://img.shields.io/badge/Node.js-22+-brightgreen?style=for-the-badge&logo=node.js)](https://nodejs.org)
[![Tests](https://img.shields.io/badge/Tests-Core%20%2B%20Advanced%20%2B%20Research-success?style=for-the-badge)](tests/)
[![Repos](https://img.shields.io/badge/Evaluated%20on-10%20Real%20Repos-orange?style=for-the-badge)](experiments/)

**Pre-Execution Repository Filtering Under Maximum Effective Context Window Constraints**

*Shweta Mishra · Independent Researcher · shweta.mishra.research@gmail.com*

</div>

---

## What is CORTEX?

CORTEX is a **pre-execution, size-based heuristic filtering framework** for LLM-based developer tools.
It intercepts the repository scan *before tokenization begins*, using only OS-level `stat()` metadata
to make filtering decisions in **< 0.01 ms per file** — zero indexing, zero model calls, zero config.

> **Core insight:** In data-heavy repositories, fewer than 2% of files account for over 80% of raw token cost (tail-at-scale structure). A single integer comparison — `file.size > θ` — eliminates this bloat before it reaches the tokenizer.

### Why it matters

Prior work on Maximum Effective Context Window (MECW) shows that LLM accuracy can degrade
well before advertised context limits, making context *quality*, not capacity, the binding
constraint in production developer tools. CORTEX addresses this directly:

| Filter | Mean TRR | Std | Latency |
|--------|----------|-----|---------|
| No filtering (baseline) | 0.0% | 0.0% | 1.67 ms |
| ExtensionFilter (current practice) | 70.3% | ±29.3% | 2.92 ms |
| **SizeFilter(1 MB) — Proposed ★** | **79.6%** | **±13.2%** | **0.30 ms** |
| HybridFilter(1 MB) — Recommended ✦ | 89.3% | ±9.0% | async |

★ 55% lower variance than ExtensionFilter. ✦ Best overall tradeoff (lowest variance, highest reduction).

---

## Key Results

Evaluated on **10 real open-source repositories** (22,046 files, 5 languages):

- **SizeFilter(1 MB):** 79.6% ± 13.2% token reduction · 0.30 ms · zero file reads
- **HybridFilter(1 MB):** 89.3% ± 9.0% token reduction · lowest variance of any filter
- **Pearson r = 0.997** between file size and token count (2,688 files validated)
- **Tail-at-scale:** tensorflow — 0.5% of files = 94% of bytes; pandas — 1.1% = 80.9%
- **Task evaluation (18 tasks, CodeLlama-7B):** reported in the paper; rerun requires Ollama and the evaluation repositories
- **Evidence-risk reduction:** limited-scope behavioral result; regenerate raw local model outputs before citing

---

## Installation

```bash
git clone https://github.com/shweta-mishra-ai/cortex
cd cortex
npm install
```

**Requirements:** Node.js 22+, npm 9+

---

## Quick Start

```javascript
import { SizeFilter, HybridFilter, FilterPipeline } from './src/filters/index.js';

// Recommended: HybridFilter for production
const pipeline = new FilterPipeline({
  filter: new HybridFilter({ threshold: 1024 * 1024 }),  // 1 MB
  repoPath: '/path/to/your/repo',
  maxDepth: 20,
});

const result = await pipeline.run();
console.log(`Token reduction: ${result.tokenReductionPct}%`);
console.log(`Latency: ${result.processingMs}ms`);
console.log(`Files allowed: ${result.allowedFiles.length}`);
```

```javascript
// Lightweight: SizeFilter for sub-millisecond filtering
const filter = new SizeFilter({ threshold: 1024 * 1024 });
const allowed = await filter.scan('/path/to/repo');
```

---

## Architecture

```
Repository Input
      │
      ▼
Directory Scanner  ──  Single-pass recursive traversal (depth-limited)
      │                 No file reads at this stage
      ▼
Heuristic Filter   ──  OS stat() only · <0.01 ms per file
      │                 stat.size > θ  →  Flagged
      ▼
Warning Layer      ──  Non-blocking CLI output · Developer override supported
      │
      ▼
Context Builder    ──  Token-safe file subset · MECW budget enforced
      │
      ▼
LLM Engine         ──  Optimised prompt · Maximum signal-to-noise ratio
```

### HybridFilter Gate Sequence

```
Gate 1: Binary Detection    →  magic-byte check, 8-byte read, <0.01 ms
Gate 2: Size Threshold      →  stat.size > θ,   zero read,    <0.01 ms
Gate 3: Minification Check  →  avg line > 500,  64 KB read,   ~1.5 ms
Gate 4: Semantic Scoring    →  keyword density, 4 KB read,    ~6.0 ms
                                                              ─────────
                               Early exit on first trigger    ~7.5 ms max
```

---

## Filter Reference

| Filter | Method | File Read | Mean TRR | Std |
|--------|--------|-----------|----------|-----|
| `NoFilter` | Admit all | None | 0.0% | 0.0% |
| `GitignoreFilter` | .gitignore patterns | None | 0.0% | 0.0% |
| `MinifiedFilter` | Avg line > 500 chars | 64 KB | 0.0% | 0.0% |
| `BinaryFilter` | Magic-byte signatures | 8 B | 28.8% | 21.8% |
| `ExtensionFilter` | Extension blocklist | None | 70.3% | 29.3% |
| `SizeFilter` ★ | stat.size > θ | **None** | **79.6%** | **13.2%** |
| `SemanticFilter` | Keyword density | 4 KB | 84.5% | 20.9% |
| `SizeFilter(50KB)` | stat.size > 50 KB | None | 89.6% | 9.0% |
| `HybridFilter` ✦ | Gates 1–4 in sequence | ≤ 4 KB | 89.3% | 9.0% |

---

## Configuration

```javascript
// cortex.config.js
export default {
  filter: 'hybrid',           // 'size' | 'hybrid' | 'extension' | 'semantic'
  threshold: 1024 * 1024,     // 1 MB (recommended)
  maxDepth: 20,               // max directory recursion depth
  excludeDirs: [              // always-pruned system directories
    'node_modules', '.git', '__pycache__',
    'dist', 'build', '.venv', 'vendor'
  ],
  nonBlocking: true,          // warning layer is async
  allowOverrides: [],         // explicit file inclusions regardless of size
  tokenizer: 'cl100k_base',  // tiktoken encoding
};
```

---

## Running Experiments

```bash
# Full 10-repository evaluation (reproduces Table IV in paper)
npm run experiment:full

# Single repository
npm run experiment:single -- --repo express_js

# Threshold sensitivity analysis (reproduces Fig. 4)
npm run experiment:threshold

# Token-density validation (reproduces Fig. 9, Table V)
npm run experiment:validate

# Task-level evaluation (reproduces Table VII, requires Ollama + CodeLlama-7B)
npm run experiment:tasks
```

---

## Running Tests

```bash
# All core, advanced, and research-filter tests
npm test

# With coverage
npm run test:coverage

# Specific suites
npm run test:core
npm run test:advanced
npm run test:research
```

Tests use dependency injection for filesystem behavior and are designed to be deterministic across platforms.

---

## Paper Result Verification

```bash
# Check committed efficiency results against headline paper numbers
npm run verify:paper
```

This verifies the tracked efficiency-result artifact under `experiments/results/`.
Task-level CodeLlama outputs are not bundled in this archive; rerun them locally with:

```bash
npm run experiment:tasks
```

---

## Additional Research Filters and Benchmarks

Research-stage filters are available separately from the paper's reported core filters:

```javascript
import {
  PathPatternFilter,
  SourceAwareSizeFilter,
  RiskScoringFilter,
} from 'cortex-filter/research';
```

Development benchmarks:

```bash
npm run benchmark
npm run benchmark:scale
npm run benchmark:matrix
```

These filters and benchmarks are for future experiments and should not be treated
as paper results until evaluated on the full corpus.

---

## Repository Corpus

| Repository | Language | Files | Baseline Tokens | Domain |
|-----------|----------|-------|-----------------|--------|
| express_js | JavaScript | 92 | 2.0 M | Web server |
| fastapi_py | Python | 153 | 6.1 M | API framework |
| gin_go | Go | 103 | 2.0 M | Web server |
| django_py | Python | 972 | 35.7 M | Web framework |
| react_js | JS/TS | 738 | 9.1 M | UI library |
| rails_rb | Ruby | 1,937 | 23.3 M | Web framework |
| pandas_py | Python | 1,332 | 127.0 M | Data library |
| vscode_ts | TypeScript | 3,293 | 165.7 M | IDE editor |
| kubernetes_go | Go | 6,684 | 112.1 M | Orchestration |
| tensorflow_py | Py/C++ | 6,672 | 1.13 B | ML framework |

**Total: 22,046 files · 5 languages · 1.85 B baseline tokens**

---

## Citing CORTEX

```bibtex
@misc{mishra2025cortex,
  title     = {Correctness-Aware Context Hygiene: Pre-Execution Repository
               Filtering Under Maximum Effective Context Window Constraints},
  author    = {Mishra, Shweta},
  year      = {2025},
  note      = {Codename: CORTEX},
  url       = {https://github.com/shweta-mishra-ai/cortex}
}
```

---

## Related Work

| System | Stage | Index | Handles Binary |
|--------|-------|--------|-----------------|
| RepoCoder | Post-read | Yes | No |
| GraphRAG | Post-read | Yes | No |
| AST Chunking | Post-read | No | No |
| Dense Embedding | Post-read | Yes | No |
| **CORTEX (SizeFilter)** | **Pre-read** | **No** | **Yes†** |

† Via size threshold — no content inspection required.

CORTEX is **complementary** to all semantic retrieval approaches: it reduces the candidate file set by 80–97% before any semantic system is invoked, reducing index-construction cost for RepoCoder/GraphRAG and eliminating binary artifacts that break AST parsers.

---

## License

MIT License. See [LICENSE](LICENSE).

---

<div align="center">
<sub>CORTEX · Context Optimization via Repository Token EXclusion</sub><br>
<sub>Shweta Mishra · 2026</sub>
</div>
