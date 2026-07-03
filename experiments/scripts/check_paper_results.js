/**
 * Verifies that the committed efficiency-result artifact contains the headline
 * numbers reported in the paper. This does not rerun external repositories;
 * use npm run experiment:full for a fresh full reproduction.
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const resultsPath = path.join(
  __dirname,
  '..',
  'results',
  'results_2026-05-04T00-56-16-227Z.json',
);

const expected = [
  ['metadata.reposScanned', 10],
  ['metadata.totalFiles', 22046],
  ['filter:SizeFilter(1MB).avgTokenReductionPct', 79.6],
  ['filter:SizeFilter(1MB).stdTokenReductionPct', 13.2],
  ['filter:HybridFilter(1MB).avgTokenReductionPct', 89.3],
  ['filter:HybridFilter(1MB).stdTokenReductionPct', 9.0],
  ['heuristicValidation.sampleSize', 2688],
  ['heuristicValidation.pearsonR', 0.997],
];

function getPath(obj, key) {
  if (key.startsWith('filter:')) {
    const [filterName, metric] = key.slice('filter:'.length).split('.');
    return obj.aggregated.find((row) => row.filter === filterName)?.[metric];
  }
  return key.split('.').reduce((value, part) => value?.[part], obj);
}

function nearlyEqual(actual, wanted) {
  if (typeof wanted !== 'number') return actual === wanted;
  return Math.abs(Number(actual) - wanted) <= 1e-9;
}

if (!fs.existsSync(resultsPath)) {
  console.error(`Missing committed result artifact: ${resultsPath}`);
  process.exit(1);
}

const data = JSON.parse(fs.readFileSync(resultsPath, 'utf8'));
let failures = 0;

for (const [key, wanted] of expected) {
  const actual = getPath(data, key);
  if (!nearlyEqual(actual, wanted)) {
    console.error(`FAIL ${key}: expected ${wanted}, got ${actual}`);
    failures++;
  } else {
    console.log(`OK   ${key}: ${actual}`);
  }
}

const taskFiles = fs.readdirSync(path.join(__dirname, '..', 'results'))
  .filter((name) => name.startsWith('task_results_') && name.endsWith('.json'));

if (taskFiles.length === 0) {
  console.warn(
    'WARN No committed task_results_*.json found. Task-level CodeLlama results '
    + 'require a local Ollama rerun and are not verified by this script.',
  );
}

process.exit(failures === 0 ? 0 : 1);
