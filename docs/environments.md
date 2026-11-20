# Environments

Construct any environment by passing its ID to `gymnax.make`:

```python
import gymnax

env, params = gymnax.make("Breakout-MinAtar")
print(gymnax.registered_envs)  # all available IDs
```

The spaces below hold for the default constructor arguments and parameters. Each ID
links to a page with the dynamics, constructor arguments and parameters.

## Classic control

Ports of the Gymnasium classic control tasks.

| ID | Observation | Actions | Reference |
| --- | --- | --- | --- |
| [`CartPole-v1`](reference/environments/classic_control/cartpole.md) | `Box(4,)` | `Discrete(2)` | [Brockman et al. (2016)](https://arxiv.org/abs/1606.01540) |
| [`Pendulum-v1`](reference/environments/classic_control/pendulum.md) | `Box(3,)` | `Box(1,)` | [Brockman et al. (2016)](https://arxiv.org/abs/1606.01540) |
| [`Acrobot-v1`](reference/environments/classic_control/acrobot.md) | `Box(6,)` | `Discrete(3)` | [Brockman et al. (2016)](https://arxiv.org/abs/1606.01540) |
| [`MountainCar-v0`](reference/environments/classic_control/mountain_car.md) | `Box(2,)` | `Discrete(3)` | [Brockman et al. (2016)](https://arxiv.org/abs/1606.01540) |
| [`MountainCarContinuous-v0`](reference/environments/classic_control/continuous_mountain_car.md) | `Box(2,)` | `Box(1,)` | [Brockman et al. (2016)](https://arxiv.org/abs/1606.01540) |

## bsuite

Diagnostic tasks from the Behaviour Suite, each probing one core capability such as memory, exploration or credit assignment.

| ID | Observation | Actions | Reference |
| --- | --- | --- | --- |
| [`Catch-bsuite`](reference/environments/bsuite/catch.md) | `Box(10, 5)` | `Discrete(3)` | [Osband et al. (2020)](https://arxiv.org/abs/1908.03568) |
| [`DeepSea-bsuite`](reference/environments/bsuite/deep_sea.md) | `Box(8, 8)` | `Discrete(2)` | [Osband et al. (2020)](https://arxiv.org/abs/1908.03568) |
| [`MemoryChain-bsuite`](reference/environments/bsuite/memory_chain.md) | `Box(3,)` | `Discrete(2)` | [Osband et al. (2020)](https://arxiv.org/abs/1908.03568) |
| [`UmbrellaChain-bsuite`](reference/environments/bsuite/umbrella_chain.md) | `Box(3,)` | `Discrete(2)` | [Osband et al. (2020)](https://arxiv.org/abs/1908.03568) |
| [`DiscountingChain-bsuite`](reference/environments/bsuite/discounting_chain.md) | `Box(2,)` | `Discrete(5)` | [Osband et al. (2020)](https://arxiv.org/abs/1908.03568) |
| [`MNISTBandit-bsuite`](reference/environments/bsuite/mnist.md) | `Box(28, 28)` | `Discrete(10)` | [Osband et al. (2020)](https://arxiv.org/abs/1908.03568) |
| [`SimpleBandit-bsuite`](reference/environments/bsuite/bandit.md) | `Box(1, 1)` | `Discrete(11)` | [Osband et al. (2020)](https://arxiv.org/abs/1908.03568) |

## MinAtar

Miniature Atari games on a 10 x 10 grid. Observations have one binary channel per object type.

| ID | Observation | Actions | Reference |
| --- | --- | --- | --- |
| [`Asterix-MinAtar`](reference/environments/minatar/asterix.md) | `Box(10, 10, 4)` | `Discrete(5)` | [Young & Tian (2019)](https://arxiv.org/abs/1903.03176) |
| [`Breakout-MinAtar`](reference/environments/minatar/breakout.md) | `Box(10, 10, 4)` | `Discrete(3)` | [Young & Tian (2019)](https://arxiv.org/abs/1903.03176) |
| [`Freeway-MinAtar`](reference/environments/minatar/freeway.md) | `Box(10, 10, 7)` | `Discrete(3)` | [Young & Tian (2019)](https://arxiv.org/abs/1903.03176) |
| [`Seaquest-MinAtar`](reference/environments/minatar/seaquest.md) | `Box(10, 10, 10)` | `Discrete(6)` | [Young & Tian (2019)](https://arxiv.org/abs/1903.03176) |
| [`SpaceInvaders-MinAtar`](reference/environments/minatar/space_invaders.md) | `Box(10, 10, 6)` | `Discrete(4)` | [Young & Tian (2019)](https://arxiv.org/abs/1903.03176) |

## Miscellaneous

Meta-learning tasks, simple continuous control and grid worlds.

| ID | Observation | Actions | Reference |
| --- | --- | --- | --- |
| [`FourRooms-misc`](reference/environments/misc/rooms.md) | `Box(4,)` | `Discrete(4)` | [Sutton et al. (1999)](https://people.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf) |
| [`MetaMaze-misc`](reference/environments/misc/meta_maze.md) | `Box(15,)` | `Discrete(4)` | [Miconi et al. (2020)](https://arxiv.org/abs/2002.10585) |
| [`PointRobot-misc`](reference/environments/misc/point_robot.md) | `Box(6,)` | `Box(2,)` | [Dorfman et al. (2021)](https://openreview.net/pdf?id=IBdEfhLveS) |
| [`BernoulliBandit-misc`](reference/environments/misc/bernoulli_bandit.md) | `Box(4,)` | `Discrete(2)` | [Wang et al. (2017)](https://arxiv.org/abs/1611.05763) |
| [`GaussianBandit-misc`](reference/environments/misc/gaussian_bandit.md) | `Box(4,)` | `Discrete(2)` | [Lange & Sprekeler (2022)](https://arxiv.org/abs/2010.04466) |
| [`Reacher-misc`](reference/environments/misc/reacher.md) | `Box(8,)` | `Box(2,)` | [Lenton et al. (2021)](https://github.com/unifyai/gym/) |
| [`Swimmer-misc`](reference/environments/misc/swimmer.md) | `Box(16,)` | `Box(2,)` | [Lenton et al. (2021)](https://github.com/unifyai/gym/) |
| [`Pong-misc`](reference/environments/misc/pong.md) | `Box(30, 40, 3)` | `Discrete(3)` | [Kirsch (2018)](https://github.com/BlackHC/batch_pong_poc) |
| [`FrozenLake-misc`](reference/environments/misc/frozen_lake.md) | `Discrete(16)` | `Discrete(4)` | [Brockman et al. (2016)](https://arxiv.org/abs/1606.01540) |
| [`Snake-misc`](reference/environments/misc/snake.md) | `Box(10, 10, 3)` | `Discrete(4)` |  |

## Trained agents and speed

[gymnax-blines](https://github.com/RobertTLange/gymnax-blines) provides PPO and
evolution strategy training pipelines, trained checkpoints and speed benchmarks for
most environments.
