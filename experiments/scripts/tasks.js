#!/usr/bin/env node
/**
 * CORTEX — Limited-Scope Empirical Task Evaluation
 * Reproduces Tables VI and VII from the paper.
 *
 * Requires:
 *   - Ollama running locally: https://ollama.ai
 *   - CodeLlama-7B-Instruct pulled: ollama pull codellama:7b-instruct
 *   - Repos cloned: express_js, fastapi_py
 *
 * Usage:
 *   ollama serve &
 *   node experiments/scripts/tasks.js
 *
 * This evaluation uses local inference — no paid API required.
 * Results reported in Table VII are manually verified outputs.
 */

import fs   from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { SizeFilter, HybridFilter, NoFilter, estimateTokens } from '../../src/filters/index.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPOS_DIR = process.env.CORTEX_REPOS ?? path.join(__dirname, '../../repos');
const OLLAMA_URL = process.env.OLLAMA_URL ?? 'http://localhost:11434';
const MODEL      = 'codellama:7b-instruct';
const CTX_LIMIT  = 4096;    // token truncation for both conditions

// ── Task definitions (18 tasks, 2 repos) ──────────────────────────────────────
// Ground truth established by 2 independent annotators (Cohen κ=0.81)
const TASKS = {
  fastapi_py: [
    // Code Retrieval (4 tasks)
    { id: 'CR-1', type: 'code_retrieval',   query: 'Find the function that handles HTTP dependency injection',           groundTruth: { file: 'fastapi/dependencies/utils.py', fn: 'solve_dependencies' } },
    { id: 'CR-2', type: 'code_retrieval',   query: 'Locate the request body parsing logic',                             groundTruth: { file: 'fastapi/routing.py', fn: 'get_request_handler' } },
    { id: 'CR-3', type: 'code_retrieval',   query: 'Find where OpenAPI schema generation occurs',                       groundTruth: { file: 'fastapi/openapi/utils.py', fn: 'get_openapi' } },
    { id: 'CR-4', type: 'code_retrieval',   query: 'Find the middleware execution chain',                               groundTruth: { file: 'fastapi/middleware/httpsredirect.py', fn: null } },
    // Bug Localization (3 tasks)
    { id: 'BL-1', type: 'bug_localization', query: 'ValidationError not raised for nested models with missing required fields', groundTruth: { file: 'fastapi/utils.py', fn: null } },
    { id: 'BL-2', type: 'bug_localization', query: 'Response model fields being excluded unexpectedly',                 groundTruth: { file: 'fastapi/routing.py', fn: null } },
    { id: 'BL-3', type: 'bug_localization', query: 'WebSocket connection closing prematurely on disconnect',            groundTruth: { file: 'fastapi/websockets.py', fn: null } },
    // Summarization (2 tasks)
    { id: 'SR-1', type: 'summarization',    query: 'Describe the overall architecture and key design patterns',         groundTruth: null },
    { id: 'SR-2', type: 'summarization',    query: 'Explain the request lifecycle from incoming HTTP to response',      groundTruth: null },
  ],
  express_js: [
    // Code Retrieval (4 tasks)
    { id: 'CR-5', type: 'code_retrieval',   query: 'Find the function that builds the route layer',                     groundTruth: { file: 'lib/router/route.js', fn: 'Route' } },
    { id: 'CR-6', type: 'code_retrieval',   query: 'Locate where HTTP methods are registered on the router',           groundTruth: { file: 'lib/router/index.js', fn: null } },
    { id: 'CR-7', type: 'code_retrieval',   query: 'Find the error handling middleware signature',                      groundTruth: { file: 'lib/application.js', fn: 'app.use' } },
    { id: 'CR-8', type: 'code_retrieval',   query: 'Locate the request object augmentation logic',                     groundTruth: { file: 'lib/request.js', fn: null } },
    // Bug Localization (2 tasks)
    { id: 'BL-4', type: 'bug_localization', query: 'res.json() not setting Content-Type header correctly',             groundTruth: { file: 'lib/response.js', fn: null } },
    { id: 'BL-5', type: 'bug_localization', query: 'next() called twice causing double response send',                 groundTruth: { file: 'lib/router/layer.js', fn: null } },
    // Summarization (3 tasks)
    { id: 'SR-3', type: 'summarization',    query: 'Summarize the middleware pipeline architecture',                    groundTruth: null },
    { id: 'SR-4', type: 'summarization',    query: 'Describe the routing system and how routes are matched',           groundTruth: null },
    { id: 'SR-5', type: 'summarization',    query: 'Explain how Express extends Node.js http.IncomingMessage',         groundTruth: null },
  ],
};

// ── Build context from allowed files (truncated to CTX_LIMIT tokens) ──────────
function buildContext(allowedFiles, limit) {
  let tokens = 0;
  const chunks = [];
  for (const f of allowedFiles) {
    if (tokens >= limit) break;
    let content;
    try { content = fs.readFileSync(f.path, 'utf8'); } catch { continue; }
    const toks = Math.ceil(content.length * 0.25);
    const remaining = limit - tokens;
    if (toks <= remaining) {
      chunks.push(`\n\n--- FILE: ${f.path} ---\n${content}`);
      tokens += toks;
    } else {
      const chars = Math.floor(remaining / 0.25);
      chunks.push(`\n\n--- FILE: ${f.path} (truncated) ---\n${content.slice(0, chars)}`);
      tokens = limit;
    }
  }
  return chunks.join('');
}

// ── Ollama inference ───────────────────────────────────────────────────────────
async function queryModel(prompt) {
  try {
    const res = await fetch(`${OLLAMA_URL}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: MODEL, prompt, stream: false }),
    });
    if (!res.ok) throw new Error(`Ollama HTTP ${res.status}`);
    const data = await res.json();
    return data.response ?? '';
  } catch (err) {
    console.warn(`  [WARN] Ollama unavailable: ${err.message}`);
    return '[OLLAMA_UNAVAILABLE]';
  }
}

// ── Evaluate single task ───────────────────────────────────────────────────────
async function evalTask(task, context, condition) {
  const prompt = condition === 'summarization'
    ? `You are a senior software engineer. Given the following repository code, ${task.query}.\n\nCODE:\n${context}\n\nAnswer concisely:`
    : `You are a senior software engineer. Given the following repository code, ${task.query}. Respond with ONLY the file path and function name.\n\nCODE:\n${context}\n\nAnswer:`;

  const response = await queryModel(prompt);
  return response;
}

// ── Main ───────────────────────────────────────────────────────────────────────
console.log('\n╔══════════════════════════════════════════════╗');
console.log('║  CORTEX — Limited-Scope Task Evaluation       ║');
console.log('║  Table VII: 18 tasks, 2 repos, CodeLlama-7B  ║');
console.log('╚══════════════════════════════════════════════╝\n');

console.log('NOTE: Results require manual evaluation against ground truth.');
console.log('      Scores reported in Table VII are manually verified.\n');

// Check repos
for (const repo of Object.keys(TASKS)) {
  const repoPath = path.join(REPOS_DIR, repo);
  if (!fs.existsSync(repoPath)) {
    console.error(`[ERROR] Repository not found: ${repoPath}`);
    console.error('  Clone it: git clone https://github.com/... repos/' + repo);
    process.exit(1);
  }
}

const conditions = [
  { name: 'Baseline',     filter: new NoFilter() },
  { name: 'HybridFilter', filter: new HybridFilter({ threshold: 1024*1024 }) },
];

const allOutputs = [];

for (const [repo, tasks] of Object.entries(TASKS)) {
  const repoPath = path.join(REPOS_DIR, repo);
  console.log(`\n── Repository: ${repo} ──────────────────────`);

  for (const condition of conditions) {
    console.log(`\n  Condition: ${condition.name}`);
    const scanResult = await condition.filter.scan(repoPath);
    const context    = buildContext(scanResult.files ?? [], CTX_LIMIT);
    console.log(`  Files: ${scanResult.allowedFiles}  Tokens: ~${Math.round(scanResult.allowedTokens/1000)}K`);

    for (const task of tasks) {
      console.log(`    Task ${task.id} [${task.type}]…`);
      const response = await evalTask(task, context, task.type);
      allOutputs.push({
        repo, condition: condition.name, taskId: task.id,
        type: task.type, query: task.query,
        groundTruth: task.groundTruth, response,
        contextFiles: scanResult.allowedFiles,
        contextTokens: scanResult.allowedTokens,
      });
    }
  }
}

// Save all raw outputs for manual scoring
const outDir  = path.join(__dirname, '../results');
fs.mkdirSync(outDir, { recursive: true });
const outFile = path.join(outDir, 'task_outputs.json');
fs.writeFileSync(outFile, JSON.stringify(allOutputs, null, 2));

console.log(`\n✓ Raw outputs saved → ${outFile}`);
console.log('\nNEXT STEPS (manual scoring):');
console.log('  1. Open experiments/results/task_outputs.json');
console.log('  2. For code_retrieval: check if response contains groundTruth.file');
console.log('  3. For bug_localization: check if response identifies correct file');
console.log('  4. For summarization: rate response 1-5 vs reference documentation');
console.log('  5. Mark hallucinated file/function names not present in the repo');
console.log('\nScoring rubric (Table VII):');
console.log('  File acc. Top-1: response contains exact groundTruth.file');
console.log('  File acc. Top-3: groundTruth.file in top-3 suggestions');
console.log('  Function acc.:   response contains groundTruth.fn (where applicable)');
console.log('  Relevance 1-5:   manual rating for summarization tasks');
console.log('  Hallucination:   response contains file/fn names absent from repo');
