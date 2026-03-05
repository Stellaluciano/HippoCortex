# Contributing to HippoCortex

Thanks for your interest in improving HippoCortex.

## Development setup

```bash
git clone <repo-url>
cd hippocortex
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Run checks (recommended):

```bash
python examples/quickstart.py
python scripts/render_diagram.py
pytest -q
```

## CI and quality gates

GitHub Actions runs the following checks on pull requests and protected branches:

- PR title/body English-only metadata guard
- Pytest test suite
- Python 3.11 and 3.12 compatibility

Please run the same checks locally before opening a PR to keep feedback cycles fast.

## Contribution flow

1. Open an issue (bug/report/proposal) before large changes.
2. Create a focused branch.
3. Keep PRs small and explain architecture impact.
4. Update docs and changelog for user-visible changes.
5. Ensure quickstart and tests still run.

## Commit style

Use clear, imperative subjects, for example:

- `docs: clarify consolidation flow`
- `feat: add replay strategy hook`
- `fix: preserve provenance ids`

## Pull request checklist

- [ ] PR title and PR body contain English text only (no non-English characters)
- [ ] Tests or runnable validation included
- [ ] README/docs updated when needed
- [ ] CHANGELOG entry added
- [ ] Security-sensitive changes documented
