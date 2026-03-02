# Contributing to HippoCortex

Thanks for your interest in improving HippoCortex.

## Development setup

```bash
git clone <repo-url>
cd hippocortex
python -m venv .venv
source .venv/bin/activate
```

Run checks:

```bash
python examples/quickstart.py
python scripts/render_diagram.py
pytest -q
```

## Contribution flow

1. Open an issue (bug/report/proposal) before large changes.
2. Create a focused branch.
3. Keep PRs small and explain architecture impact.
4. Update docs and changelog for user-visible changes.
5. Ensure quickstart still runs.

## Commit style

Use clear, imperative subjects, for example:
- `docs: clarify consolidation flow`
- `feat: add replay strategy hook`
- `fix: preserve provenance ids`

## Pull request checklist

- [ ] Tests or runnable validation included
- [ ] README/docs updated when needed
- [ ] CHANGELOG entry added
- [ ] Security-sensitive changes documented
