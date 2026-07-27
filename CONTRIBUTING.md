# Contributing

ICAPA accepts changes to the provider-neutral research core under the MIT
License. By submitting a contribution, you agree that it may be distributed
under that license and confirm that you have the right to contribute it.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
git config --local core.hooksPath .githooks
```

The hooks use `python3` by default. Set `ICAPA_PYTHON` to the absolute path of
the project interpreter if the deployment uses another Python executable.

Run the repository guard and test suite before opening a pull request:

```bash
python3 scripts/repository_guard.py --index
pytest -q
```

## Public boundary

Portfolio-construction implementation directories are intentionally represented
by placeholder files in the distributed repository. Do not commit local
implementation sources, generated research artifacts, provider credentials,
licensed datasets, client configuration, or workspace output.

Local implementation sources ignored by Git are not backed up by this
repository. Store them in an independently secured and versioned location.
Never use `git clean -fdx` in a working directory that contains local-only
implementation sources.

Keep pull requests focused, add tests for behavior changes, and use English for
source code, documentation, filenames, commit messages, and user-facing text.
