# CORTEX — Full Reproducibility Guide

This guide explains how to reproduce the tracked results, tables, and figures
for the paper source: *Correctness-Aware Context Hygiene*.

---

## Quick Start (no repos needed — use committed results)

```bash
# Verify the committed result artifact against headline paper numbers
npm run verify:paper

# Install Python plotting dependencies once
python -m pip install -r requirements.txt

# Generate all paper figures from committed results
python experiments/scripts/plot_figures.py

# Figures saved to paper/figures/
# Copy alongside main.tex and compile:
cd paper && pdflatex main.tex
```

---

## Full Reproduction (clones all 10 repos, runs experiments)

**Estimated time:** 30–60 minutes (depends on clone speed)
**Disk:** ~8 GB for all 10 repositories

```bash
# Step 1: Clone all 10 repositories
node experiments/scripts/clone_repos.js

# Step 2: Run full experiment (reproduces Table IV)
npm run experiment:full

# Step 3: Threshold sensitivity (reproduces Fig. 4)
npm run experiment:threshold

# Step 4: Token-density validation (reproduces Table V)
node experiments/scripts/run_validation.js

# Step 5: FPR estimation (reproduces Section VI-E)
node experiments/scripts/estimate_fpr.js --repo experiments/repos/express_js
node experiments/scripts/estimate_fpr.js --repo experiments/repos/pandas_py

# Step 6: Generate all figures
python experiments/scripts/plot_figures.py

# Step 7: Compile paper
cd paper && pdflatex main.tex
```

---

## Task-Level Evaluation (Table VII)

Requires Ollama with CodeLlama-7B-Instruct.

```bash
# Install Ollama: https://ollama.ai
ollama pull codellama:7b-instruct-q4_0

# Run task evaluation (18 tasks, ~20 minutes)
node experiments/scripts/run_tasks.js

# Results saved to experiments/results/task_results_<timestamp>.json
```

**Note:** Table VII results were produced under these exact conditions.
The context budget was 4,096 tokens for both baseline and filtered conditions.
Ground truth was established by two independent annotators (Cohen's κ=0.81)
*before* running the model.

Raw task-level model outputs are not bundled in this archive. Commit the
generated `task_results_<timestamp>.json` file only after rerunning locally.

---

## Environment

| Requirement | Version |
|-------------|---------|
| Node.js     | ≥ 22.0  |
| Python      | ≥ 3.10  |
| matplotlib  | ≥ 3.8   |
| numpy       | ≥ 1.26  |
| Ollama      | ≥ 0.3 (task eval only) |
| RAM         | 16 GB (CodeLlama-7B 4-bit) |

---

## Committed Results

`experiments/results/results_2026-05-04T00-56-16-227Z.json` contains
the full experiment output used in the paper. All figures can be
regenerated from this file without cloning any repositories.

---

## Verified Numbers

| Claim | Value | Source |
|-------|-------|--------|
| SizeFilter(1MB) mean TRR | 79.6% ± 13.2% | Table IV |
| HybridFilter mean TRR | 89.3% ± 9.0% | Table IV |
| Pearson r (size vs tokens) | 0.997 | Table V |
| tensorflow: % bytes in >1MB files | 94.0% | Fig. 5 |
| pandas: % bytes in >1MB files | 80.9% | Fig. 5 |
| Paired t-test: t(9)=2.31, p=0.047 | SizeFilter vs ExtensionFilter | Sec. VI-E |
| FPR at θ=1MB | ~8.4% | Sec. VI-E |
| Task eval: file accuracy (Top-1) | 25.0% → 72.2% | Table VII |
| Task eval: hallucination rate | 61.1% → 16.7% | Table VII |
