import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run integration tests that require live S3 credentials",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: tests that require CUDA or heavy torch GPU paths")
    config.addinivalue_line("markers", "integration: tests that hit real remote services (S3)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-integration", default=False):
        return
    skip_integration = pytest.mark.skip(reason="need --run-integration for S3 integration tests")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
