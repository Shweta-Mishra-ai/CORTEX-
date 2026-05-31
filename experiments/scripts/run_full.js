/**
 * CORTEX — Full Experiment Runner
 * Reproduces Tables IV, V, and Figures 3–9 from the paper.
 *
 * Usage: node experiments/scripts/run_full.js [--repos-dir <path>]
 *
 * Requires the 10 evaluation repositories cloned under experiments/repos/.
 * See docs/REPRODUCE.md for setup instructions.
 */

import fs   from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

import {
  NoFilter, GitignoreFilter, BinaryFilter, ExtensionFilter,
  MinifiedFilter, SizeFilter, SemanticFilter, HybridFilter,
  estimateTokens,
} from '../../src/filters/index.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPOS_DIR = path.join(__dirname, '..', 'repos');
const OUT_DIR   = path.join(__dirname, '..', 'results');

// All 10 paper repositories
const CORPUS = [
  { name: 'express_js',     lang: 'JS',     domain: 'Web server'    },
  { name: 'fastapi_py',     lang: 'Python', domain: 'API framework' },
  { name: 'gin_go',         lang: 'Go',     domain: 'Web server'    },
  { name: 'django_py',      lang: 'Python', domain: 'Web framework' },
  { name: 'react_js',       lang: 'JS/TS',  domain: 'UI library'    },
  { name: 'rails_rb',       lang: 'Ruby',   domain: 'Web framework' },
  { name: 'pandas_py',      lang: 'Python', domain: 'Data library'  },
  { name: 'vscode_ts',      lang: 'TS',     domain: 'IDE editor'    },
  { name: 'kubernetes_go',  lang: 'Go',     domain: 'Orchestration' },
  { name: 'tensorflow_py',  lang: 'Py/C++', domain: 'ML framework'  },
];

// All 9 filters from the paper
const FILTERS = (theta = 1024 * 1024) => [
  new NoFilter(),
  new GitignoreFilter(),
  new MinifiedFilter(),
  new BinaryFilter(),
  new ExtensionFilter(),
  new SizeFilter({ threshold: theta }),
  new SemanticFilter(),
  new SizeFilter({ threshold: 50 * 1024 }),
  new HybridFilter({ threshold: theta }),
];

const THRESHOLD_CURVE = [
  50  * 1024,
  100 * 1024,
  500 * 1024,
  1   * 1024 * 1024,
  5   * 1024 * 1024,
];

// ─── Statistics helpers ────────────────────────────────────────────────────────

function mean(arr) { return arr.reduce((a, b) => a + b, 0) / arr.length; }
function std(arr) {
  const m = mean(arr);
  return Math.sqrt(arr.reduce((s, x) => s + (x - m) ** 2, 0) / (arr.length - 1));
}
function min(arr)    { return Math.min(...arr); }
function max(arr)    { return Math.max(...arr); }
function median(arr) {
  const s = [...arr].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

// Wilson 95% CI for a proportion p with n observations
function wilsonCI(p, n) {
  const z = 1.96;
  const den = 1 + z * z / n;
  const centre = (p + z * z / (2 * n)) / den;
  const margin = (z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / den;
  return {
    lo: Math.max(0, (centre - margin) * 100),
    hi: Math.min(100, (centre + margin) * 100),
  };
}

// ─── Main runner ───────────────────────────────────────────────────────────────

async function runExperiment() {
  fs.mkdirSync(OUT_DIR, { recursive: true });

  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const outPath   = path.join(OUT_DIR, `results_${timestamp}.json`);

  console.log('\n╔══════════════════════════════════════════════════════════╗');
  console.log('║           CORTEX — Full Experiment Runner               ║');
  console.log('╚══════════════════════════════════════════════════════════╝\n');

  const allRepoResults = [];
  const missing = [];

  for (const repo of CORPUS) {
    const repoPath = path.join(REPOS_DIR, repo.name);
    if (!fs.existsSync(repoPath)) {
      console.warn(`  ⚠  Skipping ${repo.name} (not found at ${repoPath})`);
      missing.push(repo.name);
      continue;
    }
    console.log(`  Scanning ${repo.name} (${repo.lang}) ...`);
    const filterResults = [];
    for (const filter of FILTERS()) {
      const result = await filter.scan(repoPath);
      filterResults.push(result);
      process.stdout.write(`    ${filter.name.padEnd(30)} ${result.tokenReductionPct.toFixed(1)}%  ${result.processingMs.toFixed(2)}ms\n`);
    }
    allRepoResults.push({ repo, filterResults });
  }

  if (missing.length > 0) {
    console.log(`\n  Missing repos: ${missing.join(', ')}`);
    console.log('  See docs/REPRODUCE.md for setup instructions.\n');
  }

  // ── Aggregate across repos ─────────────────────────────────────────────────
  const filterNames = FILTERS().map(f => f.name);
  const aggregated  = filterNames.map(name => {
    const reductions = allRepoResults
      .map(r => r.filterResults.find(f => f.filter === name)?.tokenReductionPct)
      .filter(v => v !== undefined);
    const latencies  = allRepoResults
      .map(r => r.filterResults.find(f => f.filter === name)?.processingMs)
      .filter(v => v !== undefined);
    const p = mean(reductions) / 100;
    const n = reductions.length;
    const ci = wilsonCI(p, n);
    return {
      filter:      name,
      n:           n,
      mean:        +mean(reductions).toFixed(2),
      std:         +std(reductions).toFixed(2),
      min:         +min(reductions).toFixed(2),
      max:         +max(reductions).toFixed(2),
      median:      +median(reductions).toFixed(2),
      ci95_lo:     +ci.lo.toFixed(1),
      ci95_hi:     +ci.hi.toFixed(1),
      meanLatencyMs: +mean(latencies).toFixed(2),
    };
  });

  // ── Threshold curve ────────────────────────────────────────────────────────
  const thresholdCurve = [];
  for (const theta of THRESHOLD_CURVE) {
    const f    = new SizeFilter({ threshold: theta });
    const reds = [];
    const lats = [];
    for (const repo of CORPUS) {
      const rp = path.join(REPOS_DIR, repo.name);
      if (!fs.existsSync(rp)) continue;
      const res = await f.scan(rp);
      reds.push(res.tokenReductionPct);
      lats.push(res.processingMs);
    }
    if (reds.length === 0) continue;
    const p  = mean(reds) / 100;
    const ci = wilsonCI(p, reds.length);
    thresholdCurve.push({
      threshold:         theta,
      thresholdLabel:    theta < 1024 * 1024
        ? `${theta / 1024}KB`
        : `${theta / (1024 * 1024)}MB`,
      mean:              +mean(reds).toFixed(2),
      std:               +std(reds).toFixed(2),
      ci95_lo:           +ci.lo.toFixed(1),
      ci95_hi:           +ci.hi.toFixed(1),
      meanLatencyMs:     +mean(lats).toFixed(2),
    });
  }

  // ── Summary table (console) ────────────────────────────────────────────────
  console.log('\n╔══════════════════════════════════════════════════════════════════╗');
  console.log('║                     AGGREGATED RESULTS                         ║');
  console.log('╠══════════════════════════════════════════════════════════════════╣');
  console.log(`║ ${'Filter'.padEnd(30)} ${'Mean'.padStart(6)} ${'±SD'.padStart(6)} ${'95% CI'.padStart(14)} ${'Lat'.padStart(8)} ║`);
  console.log('╠══════════════════════════════════════════════════════════════════╣');
  for (const a of aggregated) {
    const ci = `[${a.ci95_lo.toFixed(1)}, ${a.ci95_hi.toFixed(1)}]`;
    console.log(`║ ${a.filter.padEnd(30)} ${(a.mean.toFixed(1)+'%').padStart(6)} ${('±'+a.std.toFixed(1)+'%').padStart(6)} ${ci.padStart(14)} ${(a.meanLatencyMs.toFixed(1)+'ms').padStart(8)} ║`);
  }
  console.log('╚══════════════════════════════════════════════════════════════════╝\n');

  // ── Save ───────────────────────────────────────────────────────────────────
  const output = {
    metadata: {
      timestamp,
      reposScanned:   allRepoResults.length,
      reposMissing:   missing,
      tokenizer:      'cl100k_base (heuristic k=0.2500)',
      nodeVersion:    process.version,
      paper:          'Mishra 2025 — CORTEX',
    },
    aggregated,
    thresholdCurve,
    repoResults: allRepoResults.map(({ repo, filterResults }) => ({
      repoName:  repo.name,
      lang:      repo.lang,
      domain:    repo.domain,
      filters:   filterResults,
    })),
  };

  fs.writeFileSync(outPath, JSON.stringify(output, null, 2));
  console.log(`  Results saved → ${outPath}\n`);
  return output;
}

runExperiment().catch(console.error);
