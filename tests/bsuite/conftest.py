def pytest_addoption(parser):
    parser.addoption("--all", action="store_true", help="run all combinations")


def pytest_generate_tests(metafunc):
    if "env_name" in metafunc.fixturenames:
        if metafunc.config.getoption("all"):
            metafunc.parametrize("env_name", ["Catch-bsuite",
                                              "DeepSea-bsuite",
                                              "DiscountingChain-bsuite",
                                              "MemoryChain-bsuite",
                                              "UmbrellaChain-bsuite",
                                              "MNISTBandit-bsuite",
                                              "SimpleBandit-bsuite"])
        else:
            metafunc.parametrize("env_name", ["DeepSea-bsuite"])
