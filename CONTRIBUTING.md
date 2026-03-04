# Contributing

## Development Setup

```bash
git clone https://github.com/username/computer-vision-api.git
cd computer-vision-api
pip install -e ".[dev]"
```

## Code Quality

- **Lint**: `make lint`
- **Format**: `make format`
- **Tests**: `make test`
- **Coverage**: `make test-cov`

## Guidelines

1. All code must pass `ruff check` and `ruff format`
2. New features require unit tests with >80% coverage
3. Use type hints throughout
4. Follow existing patterns in the codebase
5. Update docs for user-facing changes

## Pull Request Process

1. Create a feature branch from `develop`
2. Write tests first, then implement
3. Ensure CI passes
4. Request review from a maintainer
