/**
 * CORTEX — Latency Benchmark
 * Measures per-file decision time for each filter.
 * Usage: node benchmarks/latency.js
 */
import { performance } from 'perf_hooks';
import {
  NoFilter, BinaryFilter, ExtensionFilter, SizeFilter, HybridFilter,
} from '../src/filters/index.js';

const ITERATIONS = 100_000;
const SAMPLE_STAT = { size: 512 * 1024 };  // 512 KB — below 1 MB threshold
const SAMPLE_PATH = '/fake/src/main.py';

const filters = [
  ['NoFilter',          new NoFilter()],
  ['ExtensionFilter',   new ExtensionFilter()],
  ['SizeFilter(1MB)',   new SizeFilter({ threshold: 1024*1024 })],
  ['BinaryFilter',      new BinaryFilter()],
  ['HybridFilter(1MB)', new HybridFilter({ threshold: 1024*1024 })],
];

console.log('\nCORTEX — Latency Benchmark');
console.log(`Iterations per filter: ${ITERATIONS.toLocaleString()}\n`);
console.log('Filter'.padEnd(22), 'Total (ms)'.padEnd(14), 'Per-file (µs)');
console.log('─'.repeat(50));

for (const [name, filter] of filters) {
  // Warm-up
  for (let i = 0; i < 1000; i++) filter.allows(SAMPLE_PATH, SAMPLE_STAT);

  const t0 = performance.now();
  for (let i = 0; i < ITERATIONS; i++) {
    filter.allows(SAMPLE_PATH, SAMPLE_STAT);
  }
  const totalMs  = performance.now() - t0;
  const perFilUs = (totalMs / ITERATIONS) * 1000;

  console.log(
    name.padEnd(22),
    totalMs.toFixed(1).padEnd(14),
    perFilUs.toFixed(3) + ' µs'
  );
}
console.log('\nNote: BinaryFilter reads first 8 bytes per file; figures reflect in-memory mock.');
