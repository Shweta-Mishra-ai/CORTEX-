/**
 * Small synthetic benchmark matrix for comparing core and research filters.
 *
 * This is a development benchmark, not a paper-result reproducer.
 */

import { SizeFilter, ExtensionFilter, HybridFilter } from '../src/filters/index.js';
import {
  PathPatternFilter,
  SourceAwareSizeFilter,
  RiskScoringFilter,
} from '../src/filters/research.js';

const corpus = [
  ['/repo/src/router.py', 12_000],
  ['/repo/src/generated_pb2.py', 2_500_000],
  ['/repo/data/train.csv', 50_000_000],
  ['/repo/build/app.bundle.js', 900_000],
  ['/repo/model/weights.pkl', 80_000_000],
  ['/repo/src/app.ts', 24_000],
  ['/repo/package-lock.json', 300_000],
];

const filters = [
  new SizeFilter(),
  new ExtensionFilter(),
  new HybridFilter(),
  new PathPatternFilter(),
  new SourceAwareSizeFilter(),
  new RiskScoringFilter(),
];

console.log('\nCORTEX synthetic filter matrix');
console.log('A = allowed, B = blocked. Synthetic metadata only.\n');
console.log('File'.padEnd(32), ...filters.map((f) => f.name.slice(0, 12).padStart(13)));
console.log('-'.repeat(32 + filters.length * 14));

for (const [filePath, size] of corpus) {
  const row = filters.map((filter) => {
    try {
      return (filter.allows(filePath, { size }) ? 'A' : 'B').padStart(13);
    } catch {
      return 'N/A'.padStart(13);
    }
  });
  console.log(filePath.replace('/repo/', '').padEnd(32), ...row);
}
