/**
 * Synthetic scaling benchmark for metadata-only filter decisions.
 *
 * This benchmark does not reproduce paper numbers. It measures how filters
 * behave as the number of file metadata records grows, which is useful for
 * follow-up systems work and CI regression checks.
 */

import { performance } from 'node:perf_hooks';
import { SizeFilter, ExtensionFilter } from '../src/filters/index.js';
import {
  PathPatternFilter,
  SourceAwareSizeFilter,
  RiskScoringFilter,
} from '../src/filters/research.js';

const sizes = [1_000, 10_000, 100_000, 500_000];

const filters = [
  new SizeFilter({ threshold: 1024 * 1024 }),
  new ExtensionFilter(),
  new PathPatternFilter(),
  new SourceAwareSizeFilter(),
  new RiskScoringFilter(),
];

function syntheticFile(i) {
  const ext = i % 17 === 0 ? '.csv' : i % 13 === 0 ? '.log' : i % 5 === 0 ? '.ts' : '.py';
  const dir = i % 23 === 0 ? 'build' : i % 29 === 0 ? 'generated' : 'src';
  const size = i % 97 === 0 ? 8 * 1024 * 1024 : (i % 4096) + 512;
  return { path: `/repo/${dir}/file_${i}${ext}`, stat: { size } };
}

console.log('\nCORTEX scaling benchmark');
console.log('Records are synthetic metadata only; no filesystem reads.\n');
console.log('Filter'.padEnd(28), 'Files'.padStart(10), 'Allowed'.padStart(10), 'ms'.padStart(10));
console.log('-'.repeat(62));

for (const n of sizes) {
  const files = Array.from({ length: n }, (_, i) => syntheticFile(i));
  for (const filter of filters) {
    const start = performance.now();
    let allowed = 0;
    for (const file of files) {
      if (filter.allows(file.path, file.stat)) allowed++;
    }
    const ms = performance.now() - start;
    console.log(
      filter.name.padEnd(28),
      String(n).padStart(10),
      String(allowed).padStart(10),
      ms.toFixed(2).padStart(10),
    );
  }
}
