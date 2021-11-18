# Contributing to `gymnax`

### Issues

##### New environment implementations
`gymnax` focuses on core/classic environments, which capture many dimensions of the RL problem formulation (exploration vs. exploitation, non-stationarity, generalization) and have an experimental/educational purpose. If you believe your favorite environment fits in this category, please open an issue.

### Pull requests

We welcome everyone who found a bug and corrected it to become a contributor. More information on how to make pull requests can be found in the [Github help](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request) pages.

### Examples

If you believe that there are fundamental examples missing in the [`examples`](examples/), please feel free to add them and to create a pull request.

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
