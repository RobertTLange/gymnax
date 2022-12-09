def pytest_addoption(parser):
    parser.addoption("--all", action="store_true", help="run all combinations")


def pytest_generate_tests(metafunc):
    if "bsuite_env_name" in metafunc.fixturenames:
        if metafunc.config.getoption("all"):
            metafunc.parametrize(
                "bsuite_env_name",
                [
                    "Catch-bsuite",
                    "DeepSea-bsuite",
                    "DiscountingChain-bsuite",
                    "MemoryChain-bsuite",
                    "UmbrellaChain-bsuite",
                    "MNISTBandit-bsuite",
                    "SimpleBandit-bsuite",
                ],
            )
        else:
            metafunc.parametrize("bsuite_env_name", ["UmbrellaChain-bsuite"])
    elif "gym_env_name" in metafunc.fixturenames:
        if metafunc.config.getoption("all"):
            metafunc.parametrize(
                "gym_env_name",
                [
                    "Pendulum-v1",
                    "CartPole-v1",
                    "MountainCar-v0",
                    "MountainCarContinuous-v0",
                    "Acrobot-v1",
                ],
            )
        else:
            metafunc.parametrize("gym_env_name", ["Pendulum-v1"])
    elif "misc_env_name" in metafunc.fixturenames:
        if metafunc.config.getoption("all"):
            metafunc.parametrize(
                "misc_env_name",
                [
                    "BernoulliBandit-misc",
                    "GaussianBandit-misc",
                    "FourRooms-misc",
                    "MetaMaze-misc",
                    "PointRobot-misc",
                    "Reacher-misc",
                ],
            )
        else:
            metafunc.parametrize("misc_env_name", ["Reacher-misc"])
    elif "viz_env_name" in metafunc.fixturenames:
        if metafunc.config.getoption("all"):
            metafunc.parametrize(
                "viz_env_name",
                [
                    "MetaMaze-misc",
                    "PointRobot-misc",
                    "Catch-bsuite",
                    "SpaceInvaders-MinAtar",
                ],
            )
        else:
            metafunc.parametrize("viz_env_name", ["SpaceInvaders-MinAtar"])
