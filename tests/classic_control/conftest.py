def pytest_addoption(parser):
    parser.addoption("--all", action="store_true", help="run all combinations")


def pytest_generate_tests(metafunc):
    if "env_name" in metafunc.fixturenames:
        if metafunc.config.getoption("all"):
            metafunc.parametrize(
                "env_name",
                [
                    "Pendulum-v1",
                    "CartPole-v1",
                    "MountainCar-v0",
                    "MountainCarContinuous-v0",
                    "Acrobot-v1",
                ],
            )
        else:
            metafunc.parametrize("env_name", ["Acrobot-v1"])
