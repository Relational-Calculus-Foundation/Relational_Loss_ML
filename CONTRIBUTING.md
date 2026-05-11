# Contributing to the Relational Calculus Foundation

Thank you for your interest in contributing! The Relational Calculus Foundation
is an open-source project dedicated to providing relational calculus tools for
sustainable science. We welcome contributions of all kinds: code, documentation,
bug reports, feature requests, and even ideas for new application domains.

This document outlines the guidelines for contributing effectively and ensuring
your work gets integrated smoothly.

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](./CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code. Please report any
unacceptable behavior to the project maintainers.

## How Can I Contribute?

### Reporting Bugs

If you find a bug, please open a [GitHub issue](https://github.com/Relational-Calculus-Foundation/Relational_Loss_ML/issues/new?template=bug_report.md)
with the following details:

- A clear and descriptive title.
- A step-by-step description of how to reproduce the bug.
- The expected behavior and what actually happened.
- Your Python version and list of installed packages (run `pip list`).
- Any error messages or screenshots that might help.

### Suggesting Enhancements

If you have an idea for a new feature or an improvement, open an issue with the
"enhancement" label. Explain the use case, the problem it would solve, and, if
possible, outline a potential implementation approach.

### Contributing Code

1. **Fork the repository** and clone your fork locally.
2. **Create a branch** for your contribution. Use a descriptive name, e.g.,
   `feature/add-chemistry-decoder` or `fix/typo-in-readme`.
3. **Make your changes**, adhering to the code style guidelines below.
4. **Add tests** for new features or bug fixes. The framework uses `pytest`;
   place your tests in the `tests/` directory or alongside the relevant module.
5. **Run the existing tests** to ensure you haven't introduced any regressions:
   ```bash
   pip install -r requirements-dev.txt   # or just pytest
   pytest tests/
