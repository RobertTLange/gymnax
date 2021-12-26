# Contributing to `gymnax`
We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## We Develop with Github
We use github to host code, to track issues and feature requests, as well as accept pull requests.

## We Use [Github Flow](https://guides.github.com/introduction/flow/index.html), So All Code Changes Happen Through Pull Requests
Pull requests are the best way to propose changes to the codebase (we use [Github Flow](https://guides.github.com/introduction/flow/index.html)). We actively welcome your pull requests:

1. Fork the repo and create your branch from `master`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Any contributions you make will be under the MIT Software License
In short, when you submit code changes, your submissions are understood to be under the same [MIT License](http://choosealicense.com/licenses/mit/) that covers the project. Feel free to contact the maintainers if that's a concern.

## Report bugs using Github's [issues](https://github.com/RobertTLange/gymnax/issues)
We use GitHub issues to track public bugs. Report a bug by [opening a new issue](); it's that easy!

## Write bug reports with detail, background, and sample code

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can.
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## Use the Black Coding Style
The codebase follows the [Black](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html) coding style. Using a autoformatter can make your life easier!

## License
By contributing, you agree that your contributions will be licensed under its MIT License.

## References
This document was adapted from the open-source contribution guidelines for [Facebook's Draft](https://github.com/facebook/draft-js/blob/a9316a723f9e918afde44dea68b5f9f39b7d9b00/CONTRIBUTING.md) and from the [Transcriptase adapted version](https://gist.github.com/briandk/3d2e8b3ec8daf5a27a62).


### Things That Need Your Help a.k.a. a TODO-List

You can find a couple things that need to be tackled in the [issues of this project](https://github.com/RobertTLange/gymnax/issues). Below is a quick overview of large milestones that could need your help:

- [ ] Build `env.render(state)` support by adapting original plotting code.
- [ ] Add a set of jit-compatible action and observation wrappers.
    - [ ] Framestacking
    - [ ] Reward normalization
    - [ ] Sticky actions
    - [ ] Greyscaling
    - [ ] Observation normalization
- [ ] Environment registration ala `gym` factory so that users can easily write their own environments and create jittable instances via `gymnax.make(<env_name>)`. Currently this is fairly minimal.
- [ ] Better documentation via `mkdocs`.
- [ ] More examples for doing cool stuff with vectorized environments.
