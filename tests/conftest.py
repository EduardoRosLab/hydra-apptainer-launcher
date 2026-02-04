import shutil

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--sif-path",
        action="store",
        default=None,
        help="Path to the Apptainer .sif image for cluster tests",
    )


@pytest.fixture
def sif_path(request):
    """Return the --sif-path value, or skip if not provided."""
    path = request.config.getoption("--sif-path")
    if path is None:
        pytest.skip("--sif-path not provided")
    return path


def has_sbatch():
    return shutil.which("sbatch") is not None


def has_apptainer():
    return shutil.which("apptainer") is not None
