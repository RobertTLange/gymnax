# OpenAI gym - Classic Control Environments
| Environment Name | Observation Space  |  State Space        | Action Space        | $\text{dim}(\mathcal{O})$, $\text{dim}(\mathcal{S})$, $\text{dim}(\mathcal{A})$ |
| ---------------- |:------------------:| :------------------:| :------------------:| :------------------:|
| `Pendulum-v0`    | $\cos(\theta)$, $\sin(\theta), \dot{\theta}$ | $\theta, \dot{\theta}, t,$ `done` | `Continuous(1)` - Torque (clipped) | $(3, 4, 1)$ |
| `CartPole-v0`    | $x, \dot{x}, \theta, \dot{\theta}$ | $x, \dot{x}, \theta, \dot{\theta}, t,$ `done` | `Discrete(2)` - "Left"/"Right" | $(4, 6, 1)$ |
| `MountainCar-v0`    | $x, \dot{x}$ | $x, \dot{x}, t,$ `done` | `Discrete(3)` - "Left"/"NoOps"/"Right" | $(4, 6, 1)$ |
| `MountainCarContinuous-v0`    | $x, \dot{x}$ | $x, \dot{x}, t,$ `done` | `Continuous(1)` - Force | $(4, 6, 1)$ |
| `Acrobot-v1`    | $\cos(\theta_1), \sin(\theta_1), \cos(\theta_2), \sin(\theta_2), \dot{\theta_1}, \dot{\theta_2}$ | $\theta_1, \theta_2, \dot{\theta_1}, \dot{\theta_2}, t,$ `done` | `Discrete(3)` - +1, 0, -1 Force | $(6, 6, 1)$ |


# DeepMind BSuite - Agent Evaluation Environments
| Environment Name | Observation Space  |  State Space        | Action Space        | $\text{dim}(\mathcal{O})$, $\text{dim}(\mathcal{S})$, $\text{dim}(\mathcal{A})$ |
| ---------------- |:------------------:| :------------------:| :------------------:| :------------------:|
| `Catch-bsuite`    | 
