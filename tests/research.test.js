/**
 * Tests for research-stage filters. These filters are not part of the paper's
 * reported core results; they are maintained for follow-up experiments.
 */

import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import {
  PathPatternFilter,
  SourceAwareSizeFilter,
  RiskScoringFilter,
} from '../src/filters/research.js';

describe('PathPatternFilter', () => {
  const filter = new PathPatternFilter();

  it('blocks generated and build paths', () => {
    assert.equal(filter.allows('/repo/dist/app.js'), false);
    assert.equal(filter.allows('/repo/generated/api_client.py'), false);
  });

  it('allows normal source paths', () => {
    assert.equal(filter.allows('/repo/src/router/index.js'), true);
    assert.equal(filter.allows('/repo/fastapi/routing.py'), true);
  });

  it('handles Windows separators', () => {
    assert.equal(filter.allows('C:\\repo\\build\\bundle.js'), false);
  });

  it('blocks common lock files', () => {
    assert.equal(filter.allows('/repo/package-lock.json'), false);
    assert.equal(filter.allows('/repo/pnpm-lock.yaml'), false);
  });
});

describe('SourceAwareSizeFilter', () => {
  it('allows large source files within the source threshold', () => {
    const filter = new SourceAwareSizeFilter({
      threshold: 1024,
      sourceThreshold: 10 * 1024,
    });
    assert.equal(filter.allows('/repo/src/generated_types.ts', { size: 5000 }), true);
  });

  it('blocks large non-source files', () => {
    const filter = new SourceAwareSizeFilter({ threshold: 1024 });
    assert.equal(filter.allows('/repo/data/train.csv', { size: 5000 }), false);
  });

  it('blocks noisy extensions even when small', () => {
    const filter = new SourceAwareSizeFilter({ threshold: 1024 * 1024 });
    assert.equal(filter.allows('/repo/app.log', { size: 128 }), false);
  });
});

describe('RiskScoringFilter', () => {
  it('blocks files with multiple metadata risk signals', () => {
    const filter = new RiskScoringFilter({ threshold: 1024 });
    assert.equal(filter.allows('/repo/build/model.pkl', { size: 10_000 }), false);
  });

  it('allows ordinary source files', () => {
    const filter = new RiskScoringFilter({ threshold: 1024 });
    assert.equal(filter.allows('/repo/src/main.py', { size: 900 }), true);
  });
});
