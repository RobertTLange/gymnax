# Classic-control environment spaces

Gymnax space declarations are parameter-aware. Query them directly with
`env.observation_space(params)`, `env.state_space(params)`, and
`env.action_space(params)` for the authoritative contract.

| Environment | Observation | State | Action |
| --- | --- | --- | --- |
| `Pendulum-v1` | $cos(\theta)$, $\sin(\theta)$, $\dot{\theta}$ | $\theta$, $\dot{\theta}$, time | Continuous torque |
| `CartPole-v1` | $x$, $\dot{x}$, $\theta$, $\dot{\theta}$ | $x$, $\dot{x}$, $\theta$, $\dot{\theta}$, time | Discrete left/right |
| `MountainCar-v0` | position, velocity | position, velocity, time | Discrete left/no-op/right |
| `MountainCarContinuous-v0` | position, velocity | position, velocity, time | Continuous force |
| `Acrobot-v1` | trigonometric joint positions and angular velocities | joint positions, angular velocities, time | Discrete torque |

Terminal status is returned by `Environment.step` as separate `terminated` and
`truncated` values; it is not stored in environment state spaces. See the
[API RFC](api_rfc.md) for the 1.0 transition contract and the
[README](../README.md) for the complete registered-environment list.
