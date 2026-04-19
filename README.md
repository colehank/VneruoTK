# VneuroTK

[![Tests](https://img.shields.io/github/actions/workflow/status/colehank/VneruoTK/tests.yml?branch=main)](https://github.com/colehank/VneruoTK/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/VneuroTK?color=blue)](https://pypi.org/project/VneuroTK/)  
A unified toolkit for visual neuroscience.  

Design philosophy through `<V N>`:
- V: `VneuroTK.vision`, DNN representations/labels of visiual stimuli, like ViT, CNN, etc.
- N: `VneuroTK.neuro`, Multi-modal neural recordings, like ephys, M/EEG, fMRI, etc.

## for contributors
run once after cloning:

```bash
uv sync --dev
uv run pre-commit install --hook-type pre-commit --hook-type pre-push
```

**CI checks** run automatically on every push to `main`:

| Check | Command |
|---|---|
| Lint + format | `ruff check . && ruff format --check .` |
| Tests | `pytest tests/ -v` |

**Releasing a new version** — version is derived from the git tag, no files need editing:

```bash
git tag vx.x.x
git push origin vx.x.x   # triggers build + publish to PyPI
```