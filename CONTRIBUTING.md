# Contributing to CORTEX

## How to Add a New Filter

1. Implement the filter in `src/filters/index.js` or `src/filters/advanced.js`
2. Extend `BaseFilter` and implement `allows(filePath, stat)`
3. Add it to the `createFilter()` factory function
4. Write tests in `tests/filters.test.js` following the Zero Disk I/O pattern
5. Add it to the experiment runner in `experiments/scripts/run_full.js`
6. Update `README.md` with the new filter in the reference table

## Filter Design Principles

- **Stat-first:** Prefer `stat.size` decisions over file reads
- **Early-exit:** If a filter must read, read as few bytes as possible
- **Safe failure:** When uncertain, allow the file (over-permissive > over-restrictive)
- **Injectable:** Accept `_fs` in constructor opts for Zero Disk I/O testing
- **Named:** Filter name should include key parameters (e.g., `SizeFilter(1MB)`)

## Running Tests

```bash
npm test                    # all 45 tests
npm run test:coverage       # with coverage report
```

All tests must pass. No physical disk access in tests.

## Planned Filters (Good First Issues)

- `LastModifiedFilter` — exclude files not modified in N days (`stat.mtime`)
- `DuplicateFilter`    — deduplicate by content hash (xxHash of first 4KB)
- `OwnershipFilter`    — exclude files owned by build bots (`stat.uid`)

Open an issue if you want to work on one of these.
