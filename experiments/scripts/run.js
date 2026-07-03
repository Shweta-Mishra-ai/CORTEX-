#!/usr/bin/env node
/**
 * CORTEX — Experiment Runner
 * Reproduces Tables III–IV and Figures 3–6 from the paper.
 *
 * Usage:
 *   node experiments/scripts/run.js --mode=full
 *   node experiments/scripts/run.js --mode=single --repo express_js
 *   node experiments/scripts/run.js --mode=threshold
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

// ── Configuration ──────────────────────────────────────────────────────────────
const REPOS_DIR    = process.env.CORTEX_REPOS ?? path.join(__dirname, '../../repos');
const RESULTS_DIR  = path.join(__dirname, '../results');
const THRESHOLD_1MB = 1024 * 1024;

const FILTERS = [
  new NoFilter(),
  new GitignoreFilter(),
  new MinifiedFilter(),
  new BinaryFilter(),
  new ExtensionFilter(),
  new SizeFilter({ threshold: THRESHOLD_1MB }),
  new SemanticFilter(),
  new SizeFilter({ threshold: 50 * 1024 }),
  new HybridFilter({ threshold: THRESHOLD_1MB }),
];

const THRESHOLD_CURVE = [
  50 * 1024,
  100 * 1024,
  500 * 1024,
  1024 * 1024,
  5 * 1024 * 1024,
].map(t => new SizeFilter({ threshold: t }));

// ── Main ───────────────────────────────────────────────────────────────────────
const args = Object.fromEntries(
  process.argv.slice(2)
    .filter(a => a.startsWith('--'))
    .map(a => a.slice(2).split('='))
);
const mode = args.mode ?? 'full';

async function runRepo(repoPath, filters) {
  const results = [];
  for (const filter of filters) {
    console.log(`  [${filter.name}] scanning…`);
    const r = await filter.scan(repoPath);
    results.push(r);
    console.log(`    → ${r.tokenReductionPct}% reduction  ${r.processingMs}ms`);
  }
  return results;
}

async function runFull() {
  console.log('\n╔══════════════════════════════════════╗');
  console.log('║  CORTEX — Full Experiment Run         ║');
  console.log('╚══════════════════════════════════════╝\n');

  if (!fs.existsSync(REPOS_DIR)) {
    console.error(`[ERROR] Repos directory not found: ${REPOS_DIR}`);
    console.error('  Set CORTEX_REPOS env var to your cloned repositories directory.');
    console.error('  Expected structure: repos/express_js/, repos/django_py/, etc.');
    process.exit(1);
  }

  const repoDirs = fs.readdirSync(REPOS_DIR)
    .filter(d => fs.statSync(path.join(REPOS_DIR, d)).isDirectory());

  if (repoDirs.length === 0) {
    console.error('[ERROR] No repository directories found in', REPOS_DIR);
    process.exit(1);
  }

  const allResults = { metadata: { timestamp: new Date().toISOString(), repos: repoDirs }, repoResults: [], aggregated: [] };

  for (const repo of repoDirs) {
    const repoPath = path.join(REPOS_DIR, repo);
    console.log(`\n── Repository: ${repo} ─────────────────`);
    const results = await runRepo(repoPath, FILTERS);
    allResults.repoResults.push({ repo, results });
  }

  // Compute aggregated stats
  for (const filter of FILTERS) {
    const vals = allResults.repoResults
      .map(r => r.results.find(x => x.filter === filter.name)?.tokenReductionPct ?? 0);
    const mean = vals.reduce((a,b)=>a+b,0)/vals.length;
    const std  = Math.sqrt(vals.reduce((a,b)=>a+(b-mean)**2,0)/vals.length);
    allResults.aggregated.push({ filter: filter.name, mean: +mean.toFixed(2), std: +std.toFixed(2) });
  }

  // Write results
  fs.mkdirSync(RESULTS_DIR, { recursive: true });
  const outFile = path.join(RESULTS_DIR, `results_${new Date().toISOString().replace(/[:.]/g,'_')}.json`);
  fs.writeFileSync(outFile, JSON.stringify(allResults, null, 2));
  console.log(`\n✓ Results written → ${outFile}`);

  // Print summary table
  console.log('\n── AGGREGATED RESULTS ─────────────────────────────');
  console.log('Filter                          Mean TRR    Std');
  console.log('─'.repeat(52));
  for (const a of allResults.aggregated) {
    console.log(`${a.filter.padEnd(32)}${String(a.mean+'%').padEnd(12)}${a.std}%`);
  }
}

async function runThreshold() {
  console.log('\n── THRESHOLD SENSITIVITY ANALYSIS ────────────────');
  if (!fs.existsSync(REPOS_DIR)) {
    console.error('[ERROR] Repos directory not found:', REPOS_DIR); process.exit(1);
  }
  const repoDirs = fs.readdirSync(REPOS_DIR)
    .filter(d => fs.statSync(path.join(REPOS_DIR, d)).isDirectory());

  const curve = [];
  for (const filter of THRESHOLD_CURVE) {
    const vals = [];
    for (const repo of repoDirs) {
      const r = await filter.scan(path.join(REPOS_DIR, repo));
      vals.push(r.tokenReductionPct);
    }
    const mean = vals.reduce((a,b)=>a+b,0)/vals.length;
    const std  = Math.sqrt(vals.reduce((a,b)=>a+(b-mean)**2,0)/vals.length);
    curve.push({ filter: filter.name, mean: +mean.toFixed(2), std: +std.toFixed(2) });
    console.log(`${filter.name.padEnd(24)} mean=${mean.toFixed(1)}%  sd=${std.toFixed(1)}%`);
  }

  fs.mkdirSync(RESULTS_DIR, { recursive: true });
  fs.writeFileSync(
    path.join(RESULTS_DIR, 'threshold_curve.json'),
    JSON.stringify(curve, null, 2)
  );
  console.log('\n✓ Threshold curve written → experiments/results/threshold_curve.json');
}

// ── Dispatch ───────────────────────────────────────────────────────────────────
switch (mode) {
  case 'full':      await runFull(); break;
  case 'threshold': await runThreshold(); break;
  case 'single': {
    const repo = args.repo ?? 'express_js';
    const repoPath = path.join(REPOS_DIR, repo);
    if (!fs.existsSync(repoPath)) {
      console.error(`[ERROR] Not found: ${repoPath}`); process.exit(1);
    }
    console.log(`\n── Single repo: ${repo} ─────────`);
    await runRepo(repoPath, FILTERS);
    break;
  }
  default:
    console.error(`Unknown mode: ${mode}`); process.exit(1);
}
