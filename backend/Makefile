# golangci-lint version, kept in sync with .github/workflows/ci-go.yml
GOLANGCI_LINT_VERSION := v2.12.2
GOLANGCI_LINT := $(shell command -v golangci-lint 2>/dev/null || echo $(shell go env GOPATH)/bin/golangci-lint)

.PHONY: lint lint-fix lint-install test fmt vet

# Run golangci-lint over the module. Installs the pinned version on first use.
lint: lint-install
	$(GOLANGCI_LINT) run ./...

# Apply autofixes from linters that support them (formatting/style). Semantic
# linters like errcheck and unused have no autofix and must be fixed by hand.
lint-fix: lint-install
	$(GOLANGCI_LINT) run --fix ./...

lint-install:
	@command -v golangci-lint >/dev/null 2>&1 || test -x "$(GOLANGCI_LINT)" || \
		go install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@$(GOLANGCI_LINT_VERSION)

test:
	go test ./...

fmt:
	gofmt -w .

vet:
	go vet ./...
