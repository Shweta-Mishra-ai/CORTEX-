/**
 * CORTEX — Task-Level Evaluation Runner (Table VII in paper)
 *
 * Requires: Ollama running locally with CodeLlama-7B-Instruct
 *   ollama pull codellama:7b-instruct-q4_0
 *   ollama serve
 *
 * Usage: node experiments/scripts/run_tasks.js
 *
 * The 18 tasks are defined below with ground-truth answers.
 * Two conditions are evaluated: Baseline (no filtering) and Filtered
 * (HybridFilter, theta=1MB). Context budget: 4,096 tokens for both.
 *
 * Metrics: File accuracy (Top-1, Top-3), Function accuracy,
 *          Relevance score (1-5), Hallucination rate.
 */

import fs   from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { HybridFilter, NoFilter, estimateTokens } from '../../src/filters/index.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPOS_DIR = path.join(__dirname, '../repos');
const OUT_DIR   = path.join(__dirname, '../results');

// ── Task definitions (18 tasks, 2 repos) ──────────────────────────────
const TASKS = [
  // ── express_js: Code Retrieval (4 tasks) ──
  { id:'E-CR-1', repo:'express_js', type:'code_retrieval',
    query:'Which function handles HTTP request routing?',
    groundTruthFile:'lib/router/index.js', groundTruthFn:'proto.handle' },
  { id:'E-CR-2', repo:'express_js', type:'code_retrieval',
    query:'Where is middleware chaining implemented?',
    groundTruthFile:'lib/application.js', groundTruthFn:'app.use' },
  { id:'E-CR-3', repo:'express_js', type:'code_retrieval',
    query:'Which file defines the Response prototype?',
    groundTruthFile:'lib/response.js', groundTruthFn:'res.send' },
  { id:'E-CR-4', repo:'express_js', type:'code_retrieval',
    query:'Where are HTTP status helpers defined?',
    groundTruthFile:'lib/utils.js', groundTruthFn:'setCharset' },
  // ── express_js: Bug Localization (2 tasks) ──
  { id:'E-BL-1', repo:'express_js', type:'bug_localization',
    query:'TypeError occurs when calling next() with a non-Error argument.',
    groundTruthFile:'lib/router/layer.js' },
  { id:'E-BL-2', repo:'express_js', type:'bug_localization',
    query:'404 handler not triggered for trailing slash routes.',
    groundTruthFile:'lib/router/index.js' },
  // ── express_js: Summarization (2 tasks) ──
  { id:'E-SR-1', repo:'express_js', type:'summarization',
    query:'Summarize the repository architecture and key modules.',
    groundTruthSummary:'Express.js is a minimal Node.js web framework providing routing, middleware, and HTTP utilities.' },
  { id:'E-SR-2', repo:'express_js', type:'summarization',
    query:'What request lifecycle does this framework implement?',
    groundTruthSummary:'Request flows through application middleware stack, then router, then route-level middleware, then handler.' },
  // ── fastapi_py: Code Retrieval (4 tasks) ──
  { id:'F-CR-1', repo:'fastapi_py', type:'code_retrieval',
    query:'Which class handles dependency injection?',
    groundTruthFile:'fastapi/dependencies/utils.py', groundTruthFn:'solve_dependencies' },
  { id:'F-CR-2', repo:'fastapi_py', type:'code_retrieval',
    query:'Where is request body validation implemented?',
    groundTruthFile:'fastapi/routing.py', groundTruthFn:'request_body_to_args' },
  { id:'F-CR-3', repo:'fastapi_py', type:'code_retrieval',
    query:'Which function generates OpenAPI schema?',
    groundTruthFile:'fastapi/openapi/utils.py', groundTruthFn:'get_openapi' },
  { id:'F-CR-4', repo:'fastapi_py', type:'code_retrieval',
    query:'Where are path operation decorators defined?',
    groundTruthFile:'fastapi/applications.py', groundTruthFn:'get' },
  // ── fastapi_py: Bug Localization (3 tasks) ──
  { id:'F-BL-1', repo:'fastapi_py', type:'bug_localization',
    query:'Pydantic ValidationError not converted to HTTP 422 in some edge cases.',
    groundTruthFile:'fastapi/exception_handlers.py' },
  { id:'F-BL-2', repo:'fastapi_py', type:'bug_localization',
    query:'Optional query parameters return None instead of default value.',
    groundTruthFile:'fastapi/dependencies/utils.py' },
  { id:'F-BL-3', repo:'fastapi_py', type:'bug_localization',
    query:'Background tasks run before response is sent in async mode.',
    groundTruthFile:'fastapi/routing.py' },
  // ── fastapi_py: Summarization (3 tasks) ──
  { id:'F-SR-1', repo:'fastapi_py', type:'summarization',
    query:'Summarize the dependency injection system.',
    groundTruthSummary:'FastAPI resolves dependencies recursively at request time using type annotations and Depends() markers.' },
  { id:'F-SR-2', repo:'fastapi_py', type:'summarization',
    query:'How does FastAPI generate OpenAPI documentation?',
    groundTruthSummary:'FastAPI introspects route decorators and Pydantic models to generate a compliant OpenAPI 3.0 schema.' },
  { id:'F-SR-3', repo:'fastapi_py', type:'summarization',
    query:'Describe the validation pipeline for incoming requests.',
    groundTruthSummary:'Requests are validated via Pydantic models, with automatic type coercion and error aggregation into 422 responses.' },
];

// ── Context builder ────────────────────────────────────────────────────
const CONTEXT_BUDGET = 4096;  // tokens (paper setting)

async function buildContext(repoPath, filter, budgetTokens) {
  const result = await filter.scan(repoPath);
  let tokens = 0; const chunks = [];
  // Smallest-first greedy packing within budget
  const sorted = [...result.files].sort((a,b) => a.tokens - b.tokens);
  for (const f of sorted) {
    if (tokens + f.tokens > budgetTokens) continue;
    try {
      const content = fs.readFileSync(f.path, 'utf8');
      chunks.push(`// FILE: ${path.relative(repoPath, f.path)}\n${content}`);
      tokens += f.tokens;
    } catch {}
  }
  return { context: chunks.join('\n\n'), tokens, fileCount: chunks.length };
}

// ── Ollama call ────────────────────────────────────────────────────────
async function callOllama(prompt, context) {
  const fullPrompt = `Repository context:\n${context}\n\n${prompt}\n\nAnswer concisely:`;
  try {
    const res = await fetch('http://localhost:11434/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'codellama:7b-instruct-q4_0',
        prompt: fullPrompt,
        stream: false,
        options: { num_predict: 150, temperature: 0.0 },
      }),
    });
    if (!res.ok) return { response: '', error: `HTTP ${res.status}` };
    const data = await res.json();
    return { response: data.response ?? '' };
  } catch (e) {
    return { response: '', error: e.message };
  }
}

// ── Evaluation ─────────────────────────────────────────────────────────
function evalCodeRetrieval(response, task) {
  const r  = response.toLowerCase();
  const fn = task.groundTruthFile.split('/').pop().replace(/\.[^.]+$/, '');
  const fileHit = r.includes(task.groundTruthFile.toLowerCase())
    || r.includes(fn.toLowerCase());
  const fnHit   = task.groundTruthFn
    ? r.includes(task.groundTruthFn.toLowerCase()) : fileHit;
  // Hallucination: mentions a file that doesn't exist (simple heuristic)
  const hallucinated = !fileHit && /\.(js|py|go|rb|ts)/.test(r);
  return { fileAccTop1: fileHit, functionAcc: fnHit, hallucinated };
}

function evalBugLocalization(response, task) {
  const r  = response.toLowerCase();
  const fn = task.groundTruthFile.split('/').pop().replace(/\.[^.]+$/, '');
  const hit = r.includes(fn.toLowerCase())
    || r.includes(task.groundTruthFile.toLowerCase());
  const hallucinated = !hit && /\.(js|py|go|rb|ts)/.test(r);
  return { fileAccTop1: hit, functionAcc: hit, hallucinated };
}

function evalSummarization(response, task) {
  const r   = response.toLowerCase();
  const kws = task.groundTruthSummary.toLowerCase().split(' ')
    .filter(w => w.length > 4).slice(0, 5);
  const hits = kws.filter(kw => r.includes(kw)).length;
  const relevance = 1 + Math.round((hits / kws.length) * 4);  // 1–5 scale
  return { fileAccTop1: hits >= 3, functionAcc: hits >= 3,
           relevance, hallucinated: false };
}

function evaluate(response, task) {
  switch (task.type) {
    case 'code_retrieval':  return evalCodeRetrieval(response, task);
    case 'bug_localization':return evalBugLocalization(response, task);
    case 'summarization':   return evalSummarization(response, task);
  }
}

// ── Main ───────────────────────────────────────────────────────────────
async function main() {
  console.log('\n╔══════════════════════════════════════════════════════════╗');
  console.log('║  CORTEX — Task-Level Evaluation  (Table VII in paper)    ║');
  console.log('║  18 tasks · 2 repos · CodeLlama-7B-Instruct · 4096 tok   ║');
  console.log('╚══════════════════════════════════════════════════════════╝\n');

  const conditions = [
    { label: 'Baseline', filter: (repo) => new NoFilter() },
    { label: 'Filtered', filter: (repo) => new HybridFilter({ threshold: 1024*1024 }) },
  ];

  const taskResults = {};
  for (const cond of conditions) {
    console.log(`\n── Condition: ${cond.label} ────────────────────────`);
    taskResults[cond.label] = { fileAccTop1: 0, functionAcc: 0, relevanceSum: 0,
                                 hallucinations: 0, relevanceTasks: 0, n: 0 };

    for (const task of TASKS) {
      const repoPath = path.join(REPOS_DIR, task.repo);
      if (!fs.existsSync(repoPath)) {
        console.warn(`  [SKIP] ${task.id}: ${task.repo} not found`);
        continue;
      }

      const filter = cond.filter(repoPath);
      const ctx    = await buildContext(repoPath, filter, CONTEXT_BUDGET);
      const { response, error } = await callOllama(task.query, ctx.context);

      if (error) {
        console.warn(`  [ERROR] ${task.id}: ${error} (is Ollama running?)`);
        continue;
      }

      const eval_result = evaluate(response, task);
      const r = taskResults[cond.label];
      r.n++;
      if (eval_result.fileAccTop1)  r.fileAccTop1++;
      if (eval_result.functionAcc)  r.functionAcc++;
      if (eval_result.hallucinated) r.hallucinations++;
      if (eval_result.relevance) { r.relevanceSum += eval_result.relevance; r.relevanceTasks++; }

      const tick = eval_result.fileAccTop1 ? '✓' : '✗';
      console.log(`  ${tick} ${task.id.padEnd(8)} ctx=${ctx.tokens}tok files=${ctx.fileCount}`);
    }
  }

  // Print summary table
  console.log('\n\n══ RESULTS SUMMARY (Table VII) ══\n');
  console.log('Metric'.padEnd(26), 'Baseline'.padEnd(12), 'Filtered'.padEnd(12), 'Delta');
  console.log('─'.repeat(60));

  const metrics = [
    { label: 'File Acc. (Top-1)', key: 'fileAccTop1', pct: true },
    { label: 'Function Acc.',     key: 'functionAcc', pct: true },
    { label: 'Hallucination Rate',key: 'hallucinations', pct: true },
  ];
  for (const m of metrics) {
    const base = taskResults['Baseline'];
    const filt = taskResults['Filtered'];
    const bv   = base.n ? (base[m.key] / base.n * 100).toFixed(1) + '%' : 'N/A';
    const fv   = filt.n ? (filt[m.key] / filt.n * 100).toFixed(1) + '%' : 'N/A';
    const delta= base.n && filt.n
      ? ((filt[m.key]/filt.n - base[m.key]/base.n) * 100).toFixed(1)+'pp' : 'N/A';
    console.log(m.label.padEnd(26), bv.padEnd(12), fv.padEnd(12), delta);
  }
  const bRel = taskResults['Baseline'].relevanceTasks
    ? (taskResults['Baseline'].relevanceSum / taskResults['Baseline'].relevanceTasks).toFixed(1) : 'N/A';
  const fRel = taskResults['Filtered'].relevanceTasks
    ? (taskResults['Filtered'].relevanceSum  / taskResults['Filtered'].relevanceTasks).toFixed(1) : 'N/A';
  console.log('Relevance (1-5)'.padEnd(26), bRel.padEnd(12), fRel.padEnd(12),
    bRel !== 'N/A' && fRel !== 'N/A' ? `+${(fRel-bRel).toFixed(1)}pts` : 'N/A');

  // Save results
  const ts  = new Date().toISOString().replace(/[:.]/g,'-');
  const out = path.join(OUT_DIR, `task_results_${ts}.json`);
  fs.mkdirSync(OUT_DIR, { recursive: true });
  fs.writeFileSync(out, JSON.stringify({ taskResults, tasks: TASKS }, null, 2));
  console.log(`\n✓ Saved → ${out}\n`);
}

main().catch(err => { console.error(err); process.exit(1); });
