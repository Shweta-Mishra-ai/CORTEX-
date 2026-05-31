/**
 * CORTEX — False Positive Rate (FPR) Estimation
 * Reproduces the FPR values reported in Section VI-E of the paper.
 *
 * Method: For each threshold θ, scan the repo, collect flagged files,
 * then classify each as relevant (FP) or irrelevant (TN) using:
 *   - Extension heuristic (source code extensions = potentially relevant)
 *   - Size bracket analysis
 *   - Human-readable report for manual verification
 *
 * Usage: node experiments/scripts/estimate_fpr.js --repo experiments/repos/express_js
 */

import fs   from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Source code extensions — files with these are "potentially relevant"
const SOURCE_EXTS = new Set([
  '.js','.ts','.jsx','.tsx','.mjs','.cjs',
  '.py','.pyw',
  '.go',
  '.rb','.rake',
  '.rs',
  '.java','.kt','.scala',
  '.c','.cpp','.cc','.h','.hpp',
  '.cs','.vb',
  '.php','.phtml',
  '.swift','.m',
  '.sh','.bash','.zsh','.fish',
  '.sql',
  '.proto',                   // proto buffers — large but relevant
  '.thrift',
  '.md','.rst','.txt','.adoc',  // docs — relevant for summarization tasks
  '.yaml','.yml','.toml','.ini','.cfg','.conf',
  '.json',                    // config JSON — relevant
  '.html','.css','.scss','.sass',
  '.tf','.hcl',               // terraform
  '.dockerfile',
]);

const THRESHOLDS = [
  { label: '50KB',  bytes: 50   * 1024 },
  { label: '100KB', bytes: 100  * 1024 },
  { label: '500KB', bytes: 500  * 1024 },
  { label: '1MB',   bytes: 1024 * 1024 },
  { label: '5MB',   bytes: 5    * 1024 * 1024 },
];

const PRUNED = new Set([
  'node_modules','.git','__pycache__','dist','build',
  '.venv','venv','vendor','target','.next','coverage',
]);

function walk(dir, depth = 0) {
  if (depth > 20) return [];
  const files = [];
  let entries;
  try { entries = fs.readdirSync(dir, { withFileTypes: true }); }
  catch { return []; }
  for (const e of entries) {
    if (e.isDirectory()) {
      if (!PRUNED.has(e.name)) files.push(...walk(path.join(dir, e.name), depth + 1));
    } else if (e.isFile()) {
      const fp = path.join(dir, e.name);
      try {
        const st = fs.statSync(fp);
        files.push({ path: fp, size: st.size, ext: path.extname(fp).toLowerCase() });
      } catch { /* skip */ }
    }
  }
  return files;
}

function estimateFPR(files, threshold) {
  const flagged = files.filter(f => f.size > threshold);
  if (flagged.length === 0) return { fpr: 0, flagged: 0, fp: 0, tn: 0, flaggedFiles: [] };

  // Classify: FP = flagged but source extension (potentially relevant)
  const fp = flagged.filter(f => SOURCE_EXTS.has(f.ext));
  const tn = flagged.filter(f => !SOURCE_EXTS.has(f.ext));

  return {
    fpr:         fp.length / flagged.length,
    flagged:     flagged.length,
    fp:          fp.length,
    tn:          tn.length,
    flaggedFiles: flagged.map(f => ({
      file: path.basename(f.path),
      sizeKB: Math.round(f.size / 1024),
      ext: f.ext,
      likelyFP: SOURCE_EXTS.has(f.ext),
    })).sort((a, b) => b.sizeKB - a.sizeKB).slice(0, 10),
  };
}

// ── Main ──────────────────────────────────────────────────────────────
const args    = process.argv.slice(2);
const repoArg = args.indexOf('--repo');
const repoPath = repoArg >= 0 ? args[repoArg + 1]
  : path.join(__dirname, '../repos/express_js');

if (!fs.existsSync(repoPath)) {
  console.error(`Repo not found: ${repoPath}`);
  console.error('Run: node experiments/scripts/clone_repos.js');
  process.exit(1);
}

console.log(`\nCORTEX — FPR Estimation`);
console.log(`Repository: ${path.basename(repoPath)}\n`);

const files = walk(repoPath);
console.log(`Total files scanned: ${files.length}`);
console.log(`\n${'Threshold'.padEnd(10)} ${'Flagged'.padEnd(10)} ${'FP (src)'.padEnd(10)} ${'TN (data)'.padEnd(12)} ${'Est. FPR'}`);
console.log('─'.repeat(55));

const results = [];
for (const { label, bytes } of THRESHOLDS) {
  const r = estimateFPR(files, bytes);
  const fprPct = (r.fpr * 100).toFixed(1);
  const star = label === '1MB' ? ' ← recommended' : '';
  console.log(`${label.padEnd(10)} ${String(r.flagged).padEnd(10)} ${String(r.fp).padEnd(10)} ${String(r.tn).padEnd(12)} ${fprPct}%${star}`);
  results.push({ threshold: label, ...r });
}

// Show top FP files at 1MB
const r1mb = results.find(r => r.threshold === '1MB');
if (r1mb?.flaggedFiles?.length) {
  console.log('\nTop flagged files at θ=1MB (sorted by size):');
  for (const f of r1mb.flaggedFiles.slice(0, 8)) {
    const fp_flag = f.likelyFP ? '⚠ likely FP' : '✓ data/binary';
    console.log(`  ${f.sizeKB.toString().padStart(6)} KB  ${f.ext.padEnd(8)} ${fp_flag.padEnd(14)} ${f.file}`);
  }
}

console.log('\nNote: FPR estimated via extension heuristic.');
console.log('Source-code extensions flagged = potential false positives.');
console.log('Manual verification recommended for protocol buffer (.proto) files.\n');
