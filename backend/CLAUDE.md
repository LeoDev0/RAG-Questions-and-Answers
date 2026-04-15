# CLAUDE.md

## Testing conventions

- Always use table-driven (parameterized) tests when a function has multiple scenarios.
- In table-driven tests, group related fields using nested structs (`mock` and `expected`) instead of flat prefixed fields like `mockReturn`, `expectedErr`.
- Use `github.com/stretchr/testify/assert` for assertions.
- Do not use decorative separator comments (e.g., `// ----`) to visually split sections in files, or any unnecessary comments.
- When table-driven tests have complex mock/expected fields, define named types instead of repeating verbose anonymous struct literals in every test case.

