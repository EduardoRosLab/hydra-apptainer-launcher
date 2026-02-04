"""Integration test: run a Hydra multirun job via CustomLocalLauncher."""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).parent.parent / "examples" / "hello_cluster"


def test_local_launcher_hello(tmp_path):
    """Run the hello_cluster example with the local executor (no SLURM needed)."""
    result = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES_DIR / "hello.py"),
            "-m",
            f"hydra.sweep.dir={tmp_path}",
            "hydra/launcher=submitit_local",
            "greeting=Hello,Hola",
            "repeat=1",
        ],
        capture_output=True,
        text=True,
        cwd=str(EXAMPLES_DIR),
        timeout=120,
    )

    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

    assert result.returncode == 0, f"Process failed with:\n{result.stderr}"

    # Verify submitit created its working directory
    submitit_dirs = list(tmp_path.rglob(".submitit"))
    assert len(submitit_dirs) > 0, "No .submitit directory created"


def test_local_launcher_returns_results(tmp_path):
    """Verify the local launcher actually executes the task function."""
    script = tmp_path / "marker_app.py"
    config = tmp_path / "marker_config.yaml"
    marker = tmp_path / "marker.txt"

    script.write_text(
        f"""\
import hydra
from omegaconf import DictConfig
from pathlib import Path

@hydra.main(version_base=None, config_path=".", config_name="marker_config")
def app(cfg: DictConfig) -> None:
    Path("{marker}").write_text(f"executed with value={{cfg.value}}")

if __name__ == "__main__":
    app()
"""
    )

    config.write_text("value: 42\n")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "-m",
            f"hydra.sweep.dir={tmp_path / 'sweep'}",
            "hydra/launcher=submitit_local",
            "value=42",
        ],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        timeout=120,
    )

    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

    assert result.returncode == 0, f"Process failed with:\n{result.stderr}"
    assert marker.exists(), "Marker file not created — task function did not execute"
    assert "executed with value=42" in marker.read_text()
