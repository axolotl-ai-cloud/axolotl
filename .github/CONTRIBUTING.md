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

## Code of Conductcode

All contributors are expected to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before participating in the axolotl community.

## Getting Started

Bugs? Please check for open issue else create a new [Issue](https://github.com/OpenAccess-AI-Collective/axolotl/issues/new).

PRs are **greatly welcome**!

1. Fork the repository and clone it to your local machine.
2. Set up the development environment by following the instructions in the [README.md](https://github.com/OpenAccess-AI-Collective/axolotl/tree/main/README.md) file.
3. Explore the codebase, run tests, and verify that everything works as expected.

Please run below to setup env
```bash
pip3 install -r requirements-dev.txt -r requirements-tests.txt
pre-commit install

# test
pytest tests/
```

## How to Contribute

### Reporting Bugs

If you encounter a bug or issue while using axolotl, please open a new issue on the [GitHub Issues](https://github.com/OpenAccess-AI-Collective/axolotl/issues) page. Provide a clear and concise description of the problem, steps to reproduce it, and any relevant error messages or logs.

### Suggesting Enhancements

We welcome ideas for improvements and new features. To suggest an enhancement, open a new issue on the [GitHub Issues](https://github.com/OpenAccess-AI-Collective/axolotl/issues) page. Describe the enhancement in detail, explain the use case, and outline the benefits it would bring to the project.

### Submitting Pull Requests

1. Create a new branch for your feature or bugfix. Use a descriptive name like `feature/your-feature-name` or `fix/your-bugfix-name`.
2. Make your changes, following the [Style Guidelines](#style-guidelines) below.
3. Test your changes and ensure that they don't introduce new issues or break existing functionality.
4. Commit your changes, following the [commit message guidelines](#commit-messages).
5. Push your branch to your fork on GitHub.
6. Open a new pull request against the `main` branch of the axolotl repository. Include a clear and concise description of your changes, referencing any related issues.

## Style Guidelines

### Code Style

axolotl uses [{codestyle}]({URLofCodestyle}) as its code style guide. Please ensure that your code follows these guidelines.

### Commit Messages

Write clear and concise commit messages that briefly describe the changes made in each commit. Use the imperative mood and start with a capitalized verb, e.g., "Add new feature" or "Fix bug in function".

## Additional Resources

- [GitHub Help](https://help.github.com/)
- [GitHub Pull Request Documentation](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests)
- [{codestyle}]({URLofCodestyle})

Thank you once again for your interest in contributing to axolotl. We look forward to collaborating with you and creating an even better project together!
