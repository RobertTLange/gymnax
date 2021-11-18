def pytest_addoption(parser):
    parser.addoption("--all", action="store_true", help="run all combinations")


def pytest_generate_tests(metafunc):
    if "env_name" in metafunc.fixturenames:
        if metafunc.config.getoption("all"):
            metafunc.parametrize(
                "env_name",
                ["BernoulliBandit-misc", "GaussianBandit-misc", "FourRooms-misc"],
            )
        else:
            metafunc.parametrize("env_name", ["BernoulliBandit-misc"])
