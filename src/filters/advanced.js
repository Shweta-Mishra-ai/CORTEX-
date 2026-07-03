/**
 * CORTEX — Advanced Filter Implementations
 * Three new filters extending the core taxonomy.
 *
 * EntropyFilter:       Shannon entropy on first 256 bytes —
 *                      generalises BinaryFilter without any magic-byte table.
 * AdaptiveSizeFilter:  Sets theta = P95 of repo file-size distribution —
 *                      eliminates manual threshold tuning (addresses L1).
 * ContextBudgetFilter: Greedily admits smallest files first until MECW
 *                      budget is consumed — directly solves Eq. (1).
 */

import fs   from 'fs';
import path from 'path';
import { BaseFilter, PRUNED_DIRS, estimateTokens, DEFAULT_THRESHOLD } from './index.js';

// ─── EntropyFilter ────────────────────────────────────────────────────────────

/**
 * Blocks high-entropy files (compressed, encrypted, binary).
 * Shannon entropy of first 256 bytes:
 *   H = -Σ p_i * log2(p_i)
 * Pure text:    H ≈ 4.0–5.5 bits/byte
 * Compressed:   H ≈ 7.5–8.0 bits/byte (near maximum)
 * Threshold 7.0 gives <1% FPR on source code in practice.
 *
 * Advantage over BinaryFilter: requires NO magic-byte table —
 * catches TFRecord, Parquet, Arrow, ONNX, and any future format.
 */
export class EntropyFilter extends BaseFilter {
  constructor(opts = {}) {
    super('EntropyFilter', opts);
    this._threshold    = opts.entropyThreshold ?? 7.0;   // bits/byte
    this._sampleBytes  = opts.sampleBytes      ?? 256;
  }

  allows(filePath, stat) {
    if (stat.size === 0) return true;
    try {
      const n   = Math.min(stat.size, this._sampleBytes);
      const buf = Buffer.alloc(n);
      const fd  = this._fs.openSync(filePath, 'r');
      const read = this._fs.readSync(fd, buf, 0, n, 0);
      this._fs.closeSync(fd);

      // Byte-frequency histogram
      const freq = new Uint32Array(256);
      for (let i = 0; i < read; i++) freq[buf[i]]++;

      // Shannon entropy
      let H = 0;
      for (let b = 0; b < 256; b++) {
        if (freq[b] === 0) continue;
        const p = freq[b] / read;
        H -= p * Math.log2(p);
      }
      return H <= this._threshold;  // block high-entropy files
    } catch { return true; }
  }
}

// ─── AdaptiveSizeFilter ───────────────────────────────────────────────────────

/**
 * Sets theta dynamically as the P95 of the repository's file-size distribution.
 * Eliminates manual threshold selection and adapts to each repo's profile.
 *
 * Algorithm:
 *   1. First pass: collect all file sizes (stat() only — no reads)
 *   2. Compute P95 of the distribution
 *   3. Second pass: filter files > P95
 *
 * Addresses Limitation L1 from the paper.
 */
export class AdaptiveSizeFilter extends BaseFilter {
  constructor(opts = {}) {
    super('AdaptiveSizeFilter(P95)', opts);
    this._percentile = opts.percentile ?? 95;
    this._minThreshold = opts.minThreshold ?? 50  * 1024;   // floor: 50 KB
    this._maxThreshold = opts.maxThreshold ?? 10  * 1024 * 1024; // ceiling: 10 MB
  }

  /** Compute the Pth percentile of an array (sorted in-place). */
  _percentileOf(sorted, p) {
    if (sorted.length === 0) return DEFAULT_THRESHOLD;
    const idx = Math.ceil((p / 100) * sorted.length) - 1;
    return sorted[Math.max(0, Math.min(idx, sorted.length - 1))];
  }

  /** Walk repo and collect file sizes (O(n) stat() calls). */
  _collectSizes(dir, depth = 0) {
    if (depth > this.maxDepth) return [];
    const sizes = [];
    let entries;
    try { entries = this._fs.readdirSync(dir, { withFileTypes: true }); }
    catch { return []; }
    for (const e of entries) {
      const full = path.join(dir, e.name);
      if (e.isDirectory()) {
        if (!PRUNED_DIRS.has(e.name)) sizes.push(...this._collectSizes(full, depth + 1));
      } else if (e.isFile()) {
        try { sizes.push(this._fs.statSync(full).size); } catch {}
      }
    }
    return sizes;
  }

  /** Override scan() to do a two-pass adaptive scan. */
  async scan(repoPath) {
    // Pass 1: collect sizes, compute adaptive threshold
    const sizes  = this._collectSizes(repoPath).sort((a, b) => a - b);
    const raw    = this._percentileOf(sizes, this._percentile);
    this._theta  = Math.max(this._minThreshold, Math.min(raw, this._maxThreshold));
    this.name    = `AdaptiveSizeFilter(P${this._percentile}=${(this._theta/1024).toFixed(0)}KB)`;

    // Pass 2: standard scan using the computed threshold
    return super.scan(repoPath);
  }

  allows(_filePath, stat) {
    return stat.size <= (this._theta ?? DEFAULT_THRESHOLD);
  }
}

// ─── ContextBudgetFilter ──────────────────────────────────────────────────────

/**
 * Greedily admits files from smallest to largest until the MECW budget
 * is consumed. Directly solves Equation (1) from the paper:
 *
 *   max  |C|   s.t.  Σ tokens(f_i) <= T_MECW
 *    C
 *
 * This is the theoretically optimal filter under the MECW constraint —
 * it maximises the number of files in context while guaranteeing the
 * token budget is not exceeded.
 *
 * @param {object} opts
 * @param {number} opts.budgetTokens  - T_MECW in tokens (default: 128,000)
 */
export class ContextBudgetFilter extends BaseFilter {
  constructor(opts = {}) {
    super('ContextBudgetFilter', opts);
    this._budget = opts.budgetTokens ?? 128_000;
  }

  /** Override scan() — must collect all files first, then sort + greedy admit. */
  async scan(repoPath) {
    const start = performance.now();

    // Collect all files with sizes
    const all = [];
    const walk = (dir, depth) => {
      if (depth > this.maxDepth) return;
      let entries;
      try { entries = this._fs.readdirSync(dir, { withFileTypes: true }); }
      catch { return; }
      for (const e of entries) {
        const full = path.join(dir, e.name);
        if (e.isDirectory()) {
          if (!PRUNED_DIRS.has(e.name)) walk(full, depth + 1);
        } else if (e.isFile()) {
          try {
            const st = this._fs.statSync(full);
            all.push({ path: full, size: st.size, tokens: estimateTokens(st.size) });
          } catch {}
        }
      }
    };
    walk(repoPath, 0);

    // Sort smallest-first (greedy optimal for cardinality maximisation)
    all.sort((a, b) => a.tokens - b.tokens);

    // Greedy admit until budget exhausted
    let used = 0;
    const allowed = [];
    for (const f of all) {
      if (used + f.tokens <= this._budget) {
        allowed.push(f);
        used += f.tokens;
      }
    }

    const totalTokens   = all.reduce((s, f) => s + f.tokens, 0);
    const allowedTokens = used;
    const processingMs  = performance.now() - start;

    return {
      filter:             this.name,
      repoPath,
      totalFiles:         all.length,
      allowedFiles:       allowed.length,
      blockedFiles:       all.length - allowed.length,
      totalTokens,
      allowedTokens,
      blockedTokens:      totalTokens - allowedTokens,
      tokenReductionPct:  +(totalTokens > 0
        ? ((totalTokens - allowedTokens) / totalTokens * 100) : 0).toFixed(2),
      processingMs:       +processingMs.toFixed(3),
      budgetTokens:       this._budget,
      budgetUtilization:  +(allowedTokens / this._budget * 100).toFixed(1),
      overflowsContext128K: allowedTokens > 128_000,
      files:              allowed,
    };
  }

  // Not used directly — scan() overrides
  allows() { return true; }
}

export default { EntropyFilter, AdaptiveSizeFilter, ContextBudgetFilter };
