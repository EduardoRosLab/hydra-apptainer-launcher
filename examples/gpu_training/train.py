"""Example: GPU training job submitted to SLURM inside an Apptainer container."""

import logging
import os
import platform

import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=".", config_name="config")
def train(cfg: DictConfig) -> None:
    log.info(f"Node: {platform.node()}, PID: {os.getpid()}")
    log.info(f"Training with lr={cfg.lr}, batch_size={cfg.batch_size}, seed={cfg.seed}")
    log.info(f"Epochs: {cfg.num_epochs}")

    # log nvidia-smi output
    try:
        import subprocess

        nvidia_smi = subprocess.check_output(["nvidia-smi"]).decode("utf-8")
        log.info("nvidia-smi output:\n" + nvidia_smi)
    except Exception as e:
        log.warning(f"Could not run nvidia-smi: {e}")

    # Simulate training loop
    for epoch in range(cfg.num_epochs):
        loss = 1.0 / (epoch + 1) * cfg.lr
        if epoch % 10 == 0:
            log.info(f"  Epoch {epoch}/{cfg.num_epochs}, loss={loss:.6f}")

    log.info("Training complete.")


if __name__ == "__main__":
    train()
