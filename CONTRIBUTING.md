# Contributing

Thanks for your interest in contributing to **weitsicht**!

## How to help
- Report bugs or request features via GitHub Issues.
- Improve documentation, examples, and user guides.
- Submit pull requests (bug fixes, new features, refactors, performance, tests).
- Help triage issues by confirming bugs or providing minimal reproductions.

## Development setup (quick)
Requirements: Python 3.10+

```bash
# clone your fork
git clone https://github.com/<your-user>/weitsicht.git
cd weitsicht

# create a branch
git checkout -b feature/short-description
# (if working on an issue, consider: issue-123-short-description)

# install with dev tooling (tests, ruff, pyright, docs tooling, pre-commit)
python -m venv .venv
pip install -e ".[develop]"
```

## Run checks locally
```bash
pytest
ruff format
ruff check
pyright
pre-commit run --all-files
```

Notes:
- To skip slow tests: `pytest -m "not slow"`
- Some tests use data/scripts in `examples/`; keep the folder available when running the suite.

## Docs (optional)
```bash
pip install -e ".[docs]"  # if you didn't install develop extras
cd docs
make html
```

## Pull requests
- Target the `main` branch.
- Keep changes focused and update tests/docs when relevant.
- To auto-close issues, use GitHub keywords in the PR description (e.g. `Fixes #123`).

## Need help?
Open an issue and describe your idea or blocker (labels like `question` / `help wanted` are welcome).

For the full (Sphinx) contribution guide, see `docs/source/contribution.rst`.
