/**
 * Clone all 10 paper repositories into experiments/repos/
 * Usage: node experiments/scripts/clone_repos.js
 */
import { execSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPOS_DIR = path.join(__dirname, '../repos');
if (!fs.existsSync(REPOS_DIR)) fs.mkdirSync(REPOS_DIR, { recursive: true });

const REPOS = [
  { name: 'express_js',     url: 'https://github.com/expressjs/express' },
  { name: 'fastapi_py',     url: 'https://github.com/tiangolo/fastapi' },
  { name: 'gin_go',         url: 'https://github.com/gin-gonic/gin' },
  { name: 'django_py',      url: 'https://github.com/django/django' },
  { name: 'react_js',       url: 'https://github.com/facebook/react' },
  { name: 'rails_rb',       url: 'https://github.com/rails/rails' },
  { name: 'pandas_py',      url: 'https://github.com/pandas-dev/pandas' },
  { name: 'vscode_ts',      url: 'https://github.com/microsoft/vscode' },
  { name: 'kubernetes_go',  url: 'https://github.com/kubernetes/kubernetes' },
  { name: 'tensorflow_py',  url: 'https://github.com/tensorflow/tensorflow' },
];

for (const { name, url } of REPOS) {
  const dest = path.join(REPOS_DIR, name);
  if (fs.existsSync(dest)) {
    console.log(`[SKIP] ${name} — already exists`);
    continue;
  }
  console.log(`[CLONE] ${name} from ${url}`);
  try {
    execSync(`git clone --depth=1 "${url}" "${dest}"`, { stdio: 'inherit' });
    console.log(`[OK] ${name}`);
  } catch (e) {
    console.error(`[FAIL] ${name}: ${e.message}`);
  }
}
console.log('\nDone. Run: npm run experiment:full');
