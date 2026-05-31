/**
 * CORTEX — Advanced Filter Tests (Zero Disk I/O)
 * Tests for EntropyFilter, AdaptiveSizeFilter, ContextBudgetFilter.
 */
import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { EntropyFilter, AdaptiveSizeFilter, ContextBudgetFilter } from '../src/filters/advanced.js';
import { estimateTokens } from '../src/filters/index.js';

const REPO = '/repo';

function mockFS(files) {
  const norm = (p) => p.replace(/\\/g, '/').replace(/\/+$/, '') || '/';
  const store = {};
  for (const [p, meta] of Object.entries(files)) store[norm(p)] = meta;

  return {
    readdirSync(dir, opts) {
      const root = norm(dir);
      const prefix = root === '/' ? '/' : root + '/';
      const seen = new Set(); const entries = [];
      for (const p of Object.keys(store)) {
        if (!p.startsWith(prefix)) continue;
        const rel = p.slice(prefix.length); const part = rel.split('/')[0];
        if (seen.has(part)) continue; seen.add(part);
        const isDir = rel.includes('/');
        if (opts?.withFileTypes)
          entries.push({ name: part, isDirectory: () => isDir, isFile: () => !isDir });
        else entries.push(part);
      }
      return entries;
    },
    statSync(p) {
      const f = store[norm(p)];
      if (!f) throw Object.assign(new Error('ENOENT'), { code: 'ENOENT' });
      return { size: f.size };
    },
    openSync:  (p) => norm(p),
    readSync:  (fd, buf) => {
      const f = store[norm(fd)]; if (!f?.content) return 0;
      f.content.copy(buf); return Math.min(f.content.length, buf.length);
    },
    closeSync: () => {},
    readFileSync: (p, enc) => {
      const f = store[norm(p)]; if (!f) throw new Error('ENOENT');
      return enc ? (f.content ?? Buffer.alloc(0)).toString(enc) : f.content ?? Buffer.alloc(0);
    },
  };
}

function make(fileMap) {
  const full = {};
  for (const [name, meta] of Object.entries(fileMap))
    full[`${REPO}/${name}`] = meta;
  return mockFS(full);
}

// ─── EntropyFilter ────────────────────────────────────────────────────
describe('EntropyFilter', () => {
  it('E-1: low-entropy text (source code) → ALLOW', () => {
    const src = 'function hello() {\n  return "world";\n}\n'.repeat(10);
    const content = Buffer.from(src);
    const _fs = make({ 'app.js': { size: content.length, content } });
    const f = new EntropyFilter({ _fs });
    assert.ok(f.allows(`${REPO}/app.js`, { size: content.length }));
  });

  it('E-2: high-entropy binary (random bytes ≈ compressed) → BLOCK', () => {
    // Random bytes have H ≈ 8 bits/byte — should be blocked
    const content = Buffer.alloc(256);
    for (let i = 0; i < 256; i++) content[i] = i; // uniform = max entropy
    const _fs = make({ 'model.bin': { size: content.length, content } });
    const f = new EntropyFilter({ _fs });
    assert.ok(!f.allows(`${REPO}/model.bin`, { size: content.length }));
  });

  it('E-3: zero-byte file → ALLOW', () => {
    const _fs = make({ 'empty.py': { size: 0, content: Buffer.alloc(0) } });
    const f = new EntropyFilter({ _fs });
    assert.ok(f.allows(`${REPO}/empty.py`, { size: 0 }));
  });

  it('E-4: custom entropy threshold', () => {
    // uniform buffer → entropy = 8 bits/byte
    const content = Buffer.alloc(256);
    for (let i = 0; i < 256; i++) content[i] = i;
    const _fs = make({ 'data': { size: 256, content } });
    const strict = new EntropyFilter({ entropyThreshold: 6.0, _fs });
    const loose  = new EntropyFilter({ entropyThreshold: 9.0, _fs });
    assert.ok(!strict.allows(`${REPO}/data`, { size: 256 }));
    assert.ok( loose.allows(`${REPO}/data`,  { size: 256 }));
  });

  it('E-5: Python source realistic entropy → ALLOW', () => {
    const src = `import os\nimport sys\nfrom pathlib import Path\n\ndef main():\n    for f in Path('.').rglob('*.py'):\n        print(f)\n\nif __name__ == '__main__':\n    main()\n`;
    const content = Buffer.from(src);
    const _fs = make({ 'main.py': { size: content.length, content } });
    const f = new EntropyFilter({ _fs });
    assert.ok(f.allows(`${REPO}/main.py`, { size: content.length }));
  });
});

// ─── AdaptiveSizeFilter ────────────────────────────────────────────────
describe('AdaptiveSizeFilter', () => {
  it('A-1: sets theta to P95 of file-size distribution', async () => {
    // 20 files: 19 small (1KB) + 1 huge (10MB) → P95 ≈ 10MB
    const files = {};
    for (let i = 0; i < 19; i++)
      files[`src/file${i}.py`] = { size: 1024 };
    files['data/huge.csv'] = { size: 10 * 1024 * 1024 };
    const _fs = make(files);
    const f = new AdaptiveSizeFilter({ _fs });
    const res = await f.scan(REPO);
    // Huge file should be blocked; 19 small files allowed
    assert.equal(res.allowedFiles, 19);
    assert.equal(res.blockedFiles, 1);
  });

  it('A-2: floor enforced — theta never below minThreshold', async () => {
    // All files tiny → P95 tiny, but floor = 50KB
    const files = {};
    for (let i = 0; i < 10; i++)
      files[`src/f${i}.js`] = { size: 100 };
    const _fs = make(files);
    const f = new AdaptiveSizeFilter({ minThreshold: 50 * 1024, _fs });
    await f.scan(REPO);
    assert.ok(f._theta >= 50 * 1024);
  });

  it('A-3: ceiling enforced — theta never above maxThreshold', async () => {
    // All files huge → P95 huge, but ceiling = 1MB
    const files = {};
    for (let i = 0; i < 10; i++)
      files[`data/f${i}.bin`] = { size: 100 * 1024 * 1024 };
    const _fs = make(files);
    const f = new AdaptiveSizeFilter({ maxThreshold: 1024 * 1024, _fs });
    await f.scan(REPO);
    assert.ok(f._theta <= 1024 * 1024);
  });
});

// ─── ContextBudgetFilter ───────────────────────────────────────────────
describe('ContextBudgetFilter', () => {
  it('C-1: admits smallest files first within budget', async () => {
    const _fs = make({
      'src/a.py':  { size: 1000  },   // ~250 tokens
      'src/b.py':  { size: 2000  },   // ~500 tokens
      'data/c.csv':{ size: 100000 },  // ~25,000 tokens
    });
    const f = new ContextBudgetFilter({ budgetTokens: 1000, _fs });
    const res = await f.scan(REPO);
    // Only a.py + b.py fit (750 tokens); c.csv exceeds budget
    assert.equal(res.allowedFiles, 2);
    assert.ok(res.allowedTokens <= 1000);
  });

  it('C-2: budgetUtilization reported correctly', async () => {
    const _fs = make({ 'main.py': { size: 4000 } }); // ~1000 tokens
    const f = new ContextBudgetFilter({ budgetTokens: 2000, _fs });
    const res = await f.scan(REPO);
    assert.ok(res.budgetUtilization > 0);
    assert.ok(res.budgetUtilization <= 100);
  });

  it('C-3: never exceeds budget', async () => {
    const files = {};
    for (let i = 0; i < 50; i++)
      files[`src/f${i}.js`] = { size: 1024 * (i + 1) };
    const _fs = make(files);
    const f = new ContextBudgetFilter({ budgetTokens: 10000, _fs });
    const res = await f.scan(REPO);
    assert.ok(res.allowedTokens <= 10000,
      `Budget exceeded: ${res.allowedTokens} > 10000`);
  });

  it('C-4: empty repo → zero files, zero tokens', async () => {
    const _fs = make({});
    const f = new ContextBudgetFilter({ budgetTokens: 128000, _fs });
    const res = await f.scan(REPO);
    assert.equal(res.totalFiles, 0);
    assert.equal(res.allowedTokens, 0);
  });

  it('C-5: token estimation consistent with estimateTokens()', async () => {
    const size = 4096;
    const _fs = make({ 'f.py': { size } });
    const f = new ContextBudgetFilter({ budgetTokens: 99999, _fs });
    const res = await f.scan(REPO);
    assert.equal(res.allowedTokens, estimateTokens(size));
  });
});
