# CLAUDE.md

## Testing conventions

- Always use table-driven (parameterized) tests when a function has multiple scenarios.
- In table-driven tests, group related fields using nested structs (`mock` and `expected`) instead of flat prefixed fields like `mockReturn`, `expectedErr`.
- Use `github.com/stretchr/testify/assert` for assertions.
- Do not use decorative separator comments (e.g., `// ----`) to visually split sections in files, or any unnecessary comments.
- When table-driven tests have complex mock/expected fields, define named types instead of repeating verbose anonymous struct literals in every test case.

## Retrieval evaluation harness

`internal/services/eval_test.go` is an offline gate that measures retrieval
quality so chunking/retrieval changes can be judged objectively instead of by
eye. It runs the real chunk → embed → cosine-rank path against a golden set,
using a deterministic local bag-of-words embedder, so it needs no API keys and
makes no network calls.

Run it:

```bash
go test ./internal/services/ -run TestEvalRetrieval -v
```

The `-v` output logs the metric line, e.g.
`retrieval eval over 18 cases (k=4): hit@1=0.722 recall@4=0.944 mrr=0.815`.
It computes `hit@1`, `recall@k` (k = `maxContentChunks`, matching what
production feeds the LLM), and `MRR`, and asserts each against a threshold. The
gate also runs as part of the normal `go test ./...` suite in CI.

Add a golden case by dropping a plain-text document in `testdata/eval/docs/`
and appending an entry to `testdata/eval/golden.json`: `document` is the file
name, `expected_source` must be a short verbatim snippet from that document
(kept under the chunk overlap so it always lands whole in a chunk), and
`expected_answer` is reserved for a future answer-quality eval and is not
asserted on today.

Thresholds in `TestEvalRetrieval` are calibrated just below the current
baseline. Treat them as a ratchet: raise them as retrieval improves so the gate
keeps catching regressions, rather than leaving them static.

