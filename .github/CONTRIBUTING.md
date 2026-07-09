# Contributing to axolotl

First of all, thank you for your interest in contributing to axolotl! We appreciate the time and effort you're willing to invest in making our project better. This document provides guidelines and information to make the contribution process as smooth as possible.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Submitting Pull Requests](#submitting-pull-requests)
- [Style Guidelines](#style-guidelines)
  - [Code Style](#code-style)
  - [Commit Messages](#commit-messages)
- [Additional Resources](#additional-resources)

## Code of Conduct

All contributors are expected to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before participating in the axolotl community.

## Getting Started

Bugs? Please check for open issue else create a new [Issue](https://github.com/axolotl-ai-cloud/axolotl/issues/new).

PRs are **greatly welcome**!

1. Fork the repository and clone it to your local machine.
2. Set up the development environment by following the instructions in the [README.md](https://github.com/axolotl-ai-cloud/axolotl/tree/main/README.md) file.
3. Explore the codebase, run tests, and verify that everything works as expected.

Please run below to setup env
```bash
# Install axolotl + dev and test dependencies
export UV_TORCH_BACKEND=cu128  # or cu130
uv venv --no-project --relocatable
source .venv/bin/activate
uv pip install --no-build-isolation -e '.[deepspeed]' --group dev --group test
pre-commit install

# test
pytest tests/
```

CI tests across a matrix of Python and PyTorch versions — see [tests.yml](workflows/tests.yml) for the current one. Tests default to `-m 'not slow'`. Run the CPU suite locally (GPU e2e runs in separate jobs — see below):

```bash
pytest -m "not slow" -n4 --dist loadfile --ignore=tests/e2e tests/
```

### Running e2e (GPU) tests locally

Recommended for larger changes before opening a PR. Needs an NVIDIA GPU. Run in the public Docker image with your checkout mounted ([docs/docker.qmd](../docs/docker.qmd) lists the available tags):

```bash
docker run --gpus all --rm -it --ipc=host -v "$PWD:/workspace/axolotl" -w /workspace/axolotl \
  axolotlai/axolotl-uv:main-latest
```

The runtime image omits test deps, so install them, then run a test:

```bash
uv pip install --group test                  # tbparse, etc.
pytest tests/e2e/test_lora_llama.py          # LoRA smoke test
pytest tests/e2e/multigpu/                    # needs >= 2 GPUs
```

Some tests require flash-attn (`uv pip install flash-attn --no-build-isolation`). `cicd/cicd.sh` and `cicd/multigpu.sh` list CI's exact run order.

## How to Contribute

### Reporting Bugs

If you encounter a bug or issue while using axolotl, please open a new issue on the [GitHub Issues](https://github.com/axolotl-ai-cloud/axolotl/issues) page. Provide a clear and concise description of the problem, steps to reproduce it, and any relevant error messages or logs.

### Suggesting Enhancements

We welcome ideas for improvements and new features. To suggest an enhancement, open a new issue on the [GitHub Issues](https://github.com/axolotl-ai-cloud/axolotl/issues) page. Describe the enhancement in detail, explain the use case, and outline the benefits it would bring to the project.

### Submitting Pull Requests

1. Create a new branch for your feature or bugfix. Use a descriptive name like `feature/your-feature-name` or `fix/your-bugfix-name`.
2. Make your changes, following the [Style Guidelines](#style-guidelines) below.
3. Test your changes and ensure that they don't introduce new issues or break existing functionality.
4. Commit your changes, following the [commit message guidelines](#commit-messages).
5. Push your branch to your fork on GitHub.
6. Open a new pull request against the `main` branch of the axolotl repository. PR formatting is prescribed in the [PR template](PULL_REQUEST_TEMPLATE.md); reference any related issues.

#### Skipping CI Checks

You can skip certain CI checks by including specific keywords in your commit messages:

- `[skip ci]` or `skip ci` - Skips all CI checks for that commit
- `[skip-e2e]` or `skip-e2e` - Skips only end-to-end tests while running other CI checks. You may also include this in the title of your PR to disable end-to-end tests for the entire PR.

## Style Guidelines

### Code Style

axolotl uses [Ruff](https://docs.astral.sh/ruff/) as its code style guide. Please ensure that your code follows these guidelines.

Use the pre-commit linter to ensure that your code is formatted consistently. It installs and runs the **exact versions CI uses**, so don't rely on a system-installed `ruff`/`mypy`:
```bash
pre-commit install        # one-time
pre-commit run --all-files
```

The exact ruff/mypy/bandit versions are pinned in [`.pre-commit-config.yaml`](../.pre-commit-config.yaml) — the same file CI's pre-commit job runs from, so local and CI never drift.

To run ruff outside pre-commit, pin it to the `ruff-pre-commit` rev in that file so output matches CI, e.g. `uvx ruff@<rev> check` / `uvx ruff@<rev> format`.

### Commit Messages

Write clear and concise commit messages that briefly describe the changes made in each commit. Use the imperative mood and start with a capitalized verb, e.g., "Add new feature" or "Fix bug in function".

## Additional Resources

- [GitHub Help](https://help.github.com/)
- [GitHub Pull Request Documentation](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests)
- [Ruff](https://docs.astral.sh/ruff/)

Thank you once again for your interest in contributing to axolotl. We look forward to collaborating with you and creating an even better project together!
