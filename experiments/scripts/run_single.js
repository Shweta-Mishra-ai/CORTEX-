/**
 * CORTEX — Single Repository Experiment
 * Usage: node experiments/scripts/run_single.js --repo express_js
 *        node experiments/scripts/run_single.js --repo pandas_py --filter hybrid
 */
import fs   from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import {
  NoFilter, ExtensionFilter, SizeFilter, HybridFilter,
  SemanticFilter, BinaryFilter,
} from '../../src/filters/index.js';
import { EntropyFilter, AdaptiveSizeFilter, ContextBudgetFilter } from '../../src/filters/advanced.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPOS_DIR = path.join(__dirname, '../repos');

const args    = process.argv.slice(2);
const get     = (f, d) => { const i = args.indexOf(f); return i>=0 ? args[i+1] : d; };
const repoName = get('--repo', 'express_js');
const repoPath = path.join(REPOS_DIR, repoName);

if (!fs.existsSync(repoPath)) {
  console.error(`Not found: ${repoPath}\nRun: node experiments/scripts/clone_repos.js`);
  process.exit(1);
}

const ALL_FILTERS = [
  new NoFilter(),
  new BinaryFilter(),
  new ExtensionFilter(),
  new SizeFilter({ threshold: 1024*1024 }),
  new SemanticFilter(),
  new HybridFilter({ threshold: 1024*1024 }),
  new EntropyFilter(),
  new AdaptiveSizeFilter(),
  new ContextBudgetFilter({ budgetTokens: 128000 }),
];

console.log(`\nCORTEX — Single Repo: ${repoName}\n`);
console.log('Filter'.padEnd(30), 'TRR%'.padEnd(8), 'Allowed'.padEnd(10), 'Latency');
console.log('─'.repeat(60));

for (const f of ALL_FILTERS) {
  const res = await f.scan(repoPath);
  console.log(
    res.filter.padEnd(30),
    `${res.tokenReductionPct}%`.padEnd(8),
    String(res.allowedFiles).padEnd(10),
    `${res.processingMs}ms`
  );
}
console.log('');
