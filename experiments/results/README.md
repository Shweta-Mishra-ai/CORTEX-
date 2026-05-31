# Result Artifacts

This directory keeps one canonical efficiency-result artifact used by the
paper:

- `results_2026-05-04T00-56-16-227Z.json`

Generated experiment outputs are ignored by default so fresh local runs do not
pollute Git history. The canonical artifact is intentionally tracked so
`npm run verify:paper` can check the paper's headline efficiency numbers.

Task-level CodeLlama outputs are not bundled in this archive. Reproduce them
with:

```bash
ollama pull codellama:7b-instruct-q4_0
npm run experiment:tasks
```

Those results depend on local model/runtime conditions and should be committed
only when the raw output file is available.
