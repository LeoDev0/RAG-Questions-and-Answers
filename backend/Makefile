# golangci-lint version, kept in sync with .github/workflows/ci-go.yml
GOLANGCI_LINT_VERSION := v2.12.2
GOLANGCI_LINT := $(shell command -v golangci-lint 2>/dev/null || echo $(shell go env GOPATH)/bin/golangci-lint)

.PHONY: lint lint-install test fmt vet

# Run golangci-lint over the module. Installs the pinned version on first use.
lint: lint-install
	$(GOLANGCI_LINT) run ./...

lint-install:
	@command -v golangci-lint >/dev/null 2>&1 || test -x "$(GOLANGCI_LINT)" || \
		go install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@$(GOLANGCI_LINT_VERSION)

test:
	go test ./...

fmt:
	gofmt -w .

vet:
	go vet ./...
