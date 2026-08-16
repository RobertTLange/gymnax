# Contributing to `gymnax`
We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## We Develop with GitHub

We use GitHub to host code, track issues and feature requests, and accept pull
requests.

## We Use [GitHub Flow](https://docs.github.com/en/get-started/using-github/github-flow)

Pull requests are the best way to propose changes. We actively welcome them:

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Run `uv sync --locked --all-extras`.
5. Run `uv run ruff check .`, `uv run ruff format --check .`, and
   `uv run pytest --all`.
6. Open the pull request.

## Contributions are under Apache-2.0

By contributing, you agree that your contributions are licensed under the
project's [Apache License 2.0](LICENSE).

## Report bugs using [GitHub issues](https://github.com/RobertTLange/gymnax/issues)

Use an issue for public bugs and feature proposals.

## Write bug reports with detail, background, and sample code

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can.
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## Use Ruff

The codebase uses [Ruff](https://docs.astral.sh/ruff/) for linting and
formatting. Run the commands above before opening a pull request.

## References
This document was adapted from the open-source contribution guidelines for [Facebook's Draft](https://github.com/facebook/draft-js/blob/a9316a723f9e918afde44dea68b5f9f39b7d9b00/CONTRIBUTING.md) and from the [Transcriptase adapted version](https://gist.github.com/briandk/3d2e8b3ec8daf5a27a62).


### Things That Need Your Help a.k.a. a TODO-List

You can find a couple things that need to be tackled in the [issues of this project](https://github.com/RobertTLange/gymnax/issues). Below is a quick overview of large milestones that could need your help:

- [ ] Add a set of jit-compatible action and observation wrappers.
    - [ ] Framestacking
    - [ ] Reward normalization
    - [ ] Sticky actions
    - [ ] Greyscaling
    - [ ] Observation normalization
- [ ] Better documentation via `mkdocs`.
- [ ] More examples for doing cool stuff with vectorized environments.
