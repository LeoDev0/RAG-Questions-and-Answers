# CLAUDE.md

## Testing conventions

- Always use table-driven (parameterized) tests when a function has multiple scenarios.
- In table-driven tests, group related fields using nested structs (`mock` and `expected`) instead of flat prefixed fields like `mockReturn`, `expectedErr`.
- Use `github.com/stretchr/testify/assert` for assertions.
