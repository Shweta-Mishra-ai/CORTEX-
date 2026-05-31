/**
 * Threshold sensitivity analysis — reproduces Fig. 4 in paper.
 * Usage: node experiments/scripts/run_threshold.js
 */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { SizeFilter } from '../../src/filters/index.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPOS_DIR = path.join(__dirname, '../repos');

const THRESHOLDS = [
  { label: '50KB',  bytes: 50*1024 },
  { label: '100KB', bytes: 100*1024 },
  { label: '500KB', bytes: 500*1024 },
  { label: '1MB',   bytes: 1024*1024 },
  { label: '5MB',   bytes: 5*1024*1024 },
];

const mean = arr => arr.reduce((a,b)=>a+b,0)/arr.length;
const std  = arr => {
  const m = mean(arr);
  return Math.sqrt(arr.reduce((s,v)=>s+(v-m)**2,0)/(arr.length-1));
};

console.log('\nCORTEX — Threshold Sensitivity Analysis\n');
console.log('θ'.padEnd(10), 'Mean TRR'.padEnd(12), 'Std'.padEnd(10), 'Min'.padEnd(10), 'Max');
console.log('─'.repeat(55));

const repos = fs.existsSync(REPOS_DIR)
  ? fs.readdirSync(REPOS_DIR).filter(d => fs.statSync(path.join(REPOS_DIR,d)).isDirectory())
  : [];

if (!repos.length) {
  console.error('No repos found. Run: node experiments/scripts/clone_repos.js');
  process.exit(1);
}

for (const { label, bytes } of THRESHOLDS) {
  const trrs = [];
  for (const repo of repos) {
    const f   = new SizeFilter({ threshold: bytes });
    const res = await f.scan(path.join(REPOS_DIR, repo));
    trrs.push(res.tokenReductionPct);
  }
  const m = mean(trrs).toFixed(1);
  const s = std(trrs).toFixed(1);
  const mn = Math.min(...trrs).toFixed(1);
  const mx = Math.max(...trrs).toFixed(1);
  const star = label === '1MB' ? ' ★ recommended' : '';
  console.log(`${label.padEnd(10)}${m.padEnd(12)}±${s.padEnd(9)}${mn.padEnd(10)}${mx}${star}`);
}
