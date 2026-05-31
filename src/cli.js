#!/usr/bin/env node
/**
 * CORTEX CLI
 * Usage: npx cortex-filter [options] <repo-path>
 *
 * Options:
 *   --filter <name>      Filter to use: size|hybrid|extension|entropy|adaptive|budget
 *   --threshold <bytes>  Size threshold in bytes (default: 1048576 = 1MB)
 *   --budget <tokens>    Token budget for ContextBudgetFilter (default: 128000)
 *   --json               Output JSON instead of human-readable summary
 *   --verbose            List all allowed files
 *
 * Examples:
 *   npx cortex-filter .
 *   npx cortex-filter --filter hybrid --threshold 524288 /path/to/repo
 *   npx cortex-filter --filter budget --budget 64000 . --json
 */

import { createFilter } from './filters/index.js';
import { EntropyFilter, AdaptiveSizeFilter, ContextBudgetFilter } from './filters/advanced.js';

const args    = process.argv.slice(2);
const get     = (flag, def) => { const i = args.indexOf(flag); return i >= 0 ? args[i+1] : def; };
const has     = (flag)      => args.includes(flag);

const repoPath  = args.find(a => !a.startsWith('--') && args[args.indexOf(a)-1] !== '--filter'
                                && args[args.indexOf(a)-1] !== '--threshold'
                                && args[args.indexOf(a)-1] !== '--budget') ?? '.';
const filterName = get('--filter', 'size');
const threshold  = parseInt(get('--threshold', '1048576'));
const budget     = parseInt(get('--budget', '128000'));
const asJSON     = has('--json');
const verbose    = has('--verbose');

let filter;
switch (filterName.toLowerCase()) {
  case 'entropy':  filter = new EntropyFilter({ threshold });          break;
  case 'adaptive': filter = new AdaptiveSizeFilter({ threshold });     break;
  case 'budget':   filter = new ContextBudgetFilter({ budgetTokens: budget }); break;
  default:         filter = createFilter(filterName, { threshold });
}

const result = await filter.scan(repoPath);

if (asJSON) {
  console.log(JSON.stringify(result, null, 2));
} else {
  console.log(`\nCORTEX — ${result.filter}`);
  console.log(`Repository:      ${result.repoPath}`);
  console.log(`Total files:     ${result.totalFiles.toLocaleString()}`);
  console.log(`Allowed files:   ${result.allowedFiles.toLocaleString()}`);
  console.log(`Blocked files:   ${result.blockedFiles.toLocaleString()}`);
  console.log(`Token reduction: ${result.tokenReductionPct}%`);
  console.log(`Latency:         ${result.processingMs}ms`);
  console.log(`Overflows 128K:  ${result.overflowsContext128K ? 'YES ⚠' : 'No ✓'}`);
  if (result.budgetUtilization !== undefined) {
    console.log(`Budget used:     ${result.budgetUtilization}% of ${budget.toLocaleString()} tokens`);
  }
  if (verbose && result.files) {
    console.log('\nAllowed files:');
    for (const f of result.files) {
      console.log(`  ${(f.size/1024).toFixed(1).padStart(8)} KB  ${f.path}`);
    }
  }
  console.log('');
}
