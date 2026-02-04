"""Minimal example: verify the Hydra → submitit → Apptainer pipeline works."""

import logging
import os
import platform

import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=".", config_name="config")
def my_app(cfg: DictConfig) -> None:
    log.info(f"Hello from {platform.node()}!")
    log.info(f"PID: {os.getpid()}")
    log.info(f"Python: {platform.python_version()} at {os.path.dirname(platform.python_compiler())}")
    log.info(f"Config: greeting={cfg.greeting}, repeat={cfg.repeat}")

    for i in range(cfg.repeat):
        log.info(f"  {cfg.greeting} (#{i + 1})")


if __name__ == "__main__":
    my_app()
