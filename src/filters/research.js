/**
 * Research-stage filters for future CORTEX experiments.
 *
 * These filters are intentionally separated from the paper's core reported
 * filters. They are useful for follow-up experiments on false positives,
 * generated-code handling, and repository hygiene, but they should not be
 * cited as part of the published results unless separately evaluated.
 */

import path from 'path';
import {
  BaseFilter,
  DEFAULT_THRESHOLD,
  NOISE_EXTENSIONS,
} from './index.js';

export const SOURCE_EXTENSIONS = new Set([
  '.c', '.cc', '.cpp', '.cs', '.go', '.h', '.hpp', '.java', '.js',
  '.jsx', '.kt', '.mjs', '.php', '.py', '.rb', '.rs', '.scala',
  '.sh', '.swift', '.ts', '.tsx',
]);

export const GENERATED_PATH_PATTERNS = [
  /(^|\/)(dist|build|coverage|target|vendor|third_party)\//,
  /(^|\/)(generated|gen|autogen|fixtures|snapshots)\//,
  /(^|\/)(package-lock\.json|yarn\.lock|pnpm-lock\.yaml|poetry\.lock|go\.sum)$/,
  /\.(min|bundle)\.(js|css)$/,
  /\.map$/,
];

function normalizePath(filePath) {
  return filePath.replace(/\\/g, '/').toLowerCase();
}

/**
 * Blocks common generated, vendored, build, and lock-file paths before any
 * content read. This complements SizeFilter by catching small-but-noisy files.
 */
export class PathPatternFilter extends BaseFilter {
  constructor(opts = {}) {
    super('PathPatternFilter', opts);
    this.patterns = opts.patterns ?? GENERATED_PATH_PATTERNS;
  }

  allows(filePath) {
    const normalized = normalizePath(filePath);
    return !this.patterns.some((pattern) => pattern.test(normalized));
  }
}

/**
 * Size filter with a larger allowance for source-code extensions. This is a
 * false-positive mitigation for large legitimate source files such as generated
 * bindings. It should be evaluated against task evidence before replacing the
 * simpler SizeFilter in any paper claim.
 */
export class SourceAwareSizeFilter extends BaseFilter {
  constructor(opts = {}) {
    super('SourceAwareSizeFilter', opts);
    this.threshold = opts.threshold ?? DEFAULT_THRESHOLD;
    this.sourceThreshold = opts.sourceThreshold ?? 5 * DEFAULT_THRESHOLD;
    this.sourceExtensions = opts.sourceExtensions ?? SOURCE_EXTENSIONS;
    this.noiseExtensions = opts.noiseExtensions ?? NOISE_EXTENSIONS;
  }

  allows(filePath, stat) {
    const ext = path.extname(filePath).toLowerCase();
    if (this.noiseExtensions.has(ext)) return false;
    if (stat.size <= this.threshold) return true;
    if (this.sourceExtensions.has(ext)) return stat.size <= this.sourceThreshold;
    return false;
  }
}

/**
 * Lightweight metadata risk model. The score is intentionally transparent:
 * extension, path, and size contribute additive risk. This is suitable for
 * ablation studies where reviewers need to inspect why files were blocked.
 */
export class RiskScoringFilter extends BaseFilter {
  constructor(opts = {}) {
    super('RiskScoringFilter', opts);
    this.threshold = opts.threshold ?? DEFAULT_THRESHOLD;
    this.blockScore = opts.blockScore ?? 3;
    this.pathFilter = opts.pathFilter ?? new PathPatternFilter(opts);
  }

  score(filePath, stat) {
    const ext = path.extname(filePath).toLowerCase();
    let score = 0;
    if (stat.size > this.threshold) score += 2;
    if (NOISE_EXTENSIONS.has(ext)) score += 2;
    if (!this.pathFilter.allows(filePath, stat)) score += 2;
    if (SOURCE_EXTENSIONS.has(ext)) score -= 1;
    return score;
  }

  allows(filePath, stat) {
    return this.score(filePath, stat) < this.blockScore;
  }
}

export default {
  PathPatternFilter,
  SourceAwareSizeFilter,
  RiskScoringFilter,
  SOURCE_EXTENSIONS,
  GENERATED_PATH_PATTERNS,
};
