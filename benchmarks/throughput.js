#!/usr/bin/env node
/**
 * CORTEX — Throughput Benchmark
 * Measures per-file filtering latency across all filter types.
 * Reproduces latency column of Table IV.
 *
 * Usage: node benchmarks/throughput.js
 */

import { performance } from 'perf_hooks';
import {
  NoFilter, SizeFilter, HybridFilter,
  ExtensionFilter, BinaryFilter,
} from '../src/filters/index.js';

const ITERATIONS = 10_000;
const FAKE_STAT  = { size: 512 };  // 512-byte source file
const FAKE_PATH  = '/repo/src/index.js';

const filters = [
  new NoFilter(),
  new ExtensionFilter(),
  new SizeFilter({ threshold: 1024 * 1024 }),
  new SizeFilter({ threshold: 50 * 1024 }),
];

console.log('\n╔══════════════════════════════════════════╗');
console.log('║  CORTEX — Per-File Latency Benchmark      ║');
console.log(`╚══════════════════════════════════════════╝`);
console.log(`\nIterations per filter: ${ITERATIONS.toLocaleString()}\n`);
console.log('Filter'.padEnd(36) + 'Avg latency (μs)');
console.log('─'.repeat(54));

for (const f of filters) {
  // Warm-up
  for (let i = 0; i < 100; i++) f.allows(FAKE_PATH, FAKE_STAT);

  const t0 = performance.now();
  for (let i = 0; i < ITERATIONS; i++) f.allows(FAKE_PATH, FAKE_STAT);
  const elapsed = performance.now() - t0;

  const avgUs = (elapsed / ITERATIONS * 1000).toFixed(3);
  console.log(`${f.name.padEnd(36)}${avgUs} μs`);
}

console.log('\nNote: BinaryFilter and HybridFilter involve file reads;');
console.log('their latency depends on OS cache and file content.');
console.log('Run on representative repository for realistic numbers.\n');
