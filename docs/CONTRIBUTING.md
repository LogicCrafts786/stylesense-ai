# Contributing to StyleSense AI

Thanks for your interest in improving StyleSense AI! This guide covers
code style, branching, testing, and PR expectations.

---

## Getting Started

1. Fork the repository and clone your fork.
2. Follow [`SETUP_GUIDE.md`](SETUP_GUIDE.md) to get a working local environment.
3. Install dev dependencies: `pip install -e ".[dev]"`
4. Create a feature branch: `git checkout -b feature/your-feature-name`

---

## Code Style

- **Type hints are required** on all function signatures (params and return types).
- **Docstrings are required** on all public classes and functions (Google-style,
  matching the existing codebase — see any file in `src/` for examples).
- Follow existing module structure: business logic in `src/`, thin wrappers
  in `@tool`-decorated functions for LangChain compatibility.
- Use the custom exception hierarchy in `src/utils/exceptions.py` rather
  than raising bare `Exception` or built-ins.
- Log meaningfully: use `get_logger(__name__)` and log at the appropriate
  level (`debug` for internals, `info` for notable events, `warning` for
  recoverable issues, `error` for failures).

Run formatting/linting before committing (if configured in your fork):
```bash
black src/ tests/
ruff check src/ tests/
```

---

## Branching & Commits

- Branch naming: `feature/<short-description>`, `fix/<short-description>`,
  `docs/<short-description>`.
- Commit messages: imperative mood, concise summary line
  (e.g., `Add FAISS fallback embedding for offline mode`).
- Keep PRs focused — one feature/fix per PR where possible.

---

## Testing Requirements

- All new tools/services/models must include corresponding unit tests
  under `tests/`, mirroring the `src/` structure.
- Mock external API calls (Gemini, weather, scraping) using `pytest-mock`
  — tests should never require real API keys or network access to pass.
- Mark tests that intentionally require live credentials with
  `@pytest.mark.integration` (excluded from default CI runs).
- Run the full suite before submitting: `pytest`
- Aim to maintain or improve test coverage; check with `pytest --cov=src`.

---

## Pull Request Checklist

- [ ] Code follows existing style and includes type hints + docstrings
- [ ] New functionality has corresponding unit tests
- [ ] `pytest` passes locally
- [ ] No secrets or `.env` values committed
- [ ] Updated relevant docs (`README.md`, `ARCHITECTURE.md`, `API_REFERENCE.md`)
      if the change affects public interfaces or setup steps
- [ ] PR description explains the "why," not just the "what"

---

## Reporting Issues

When filing an issue, please include:
- Steps to reproduce
- Expected vs. actual behavior
- Python version and OS
- Relevant log output from `logs/stylesense_*.log` (redact any API keys)

---

## Code of Conduct

Be respectful, constructive, and patient with other contributors. This
project welcomes contributions from developers of all experience levels.
