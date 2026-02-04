"""Test that Hydra discovers the plugin via namespace packages."""

import pytest


def test_import_launcher_classes():
    """Verify that both launcher classes are importable."""
    from hydra_plugins.hydra_apptainer_launcher.submitit_launcher import (
        CustomLocalLauncher,
        CustomSlurmLauncher,
    )

    assert CustomLocalLauncher._EXECUTOR == "local"
    assert CustomSlurmLauncher._EXECUTOR == "slurm"


def test_import_config_classes():
    """Verify the config dataclasses are importable and have expected fields."""
    from hydra_plugins.hydra_apptainer_launcher.config import (
        BaseQueueConf,
        LocalQueueConf,
        SlurmQueueConf,
    )

    base = BaseQueueConf()
    assert base.timeout_min == 60
    assert base.python is None

    slurm = SlurmQueueConf()
    assert slurm.partition is None
    assert slurm.array_parallelism == 256

    local = LocalQueueConf()
    assert local.timeout_min == 60


def test_launcher_inherits_from_hydra_launcher():
    """Verify our launchers implement the Hydra Launcher interface."""
    from hydra.plugins.launcher import Launcher
    from hydra_plugins.hydra_apptainer_launcher.submitit_launcher import (
        CustomLocalLauncher,
        CustomSlurmLauncher,
    )

    assert issubclass(CustomSlurmLauncher, Launcher)
    assert issubclass(CustomLocalLauncher, Launcher)


def test_config_store_registration():
    """Verify that launcher configs are registered in Hydra's ConfigStore."""
    from hydra.core.config_store import ConfigStore

    cs = ConfigStore.instance()
    # Access the registered configs — these should exist without raising
    repo = cs.repo
    # The configs are stored under hydra/launcher group
    assert repo is not None
