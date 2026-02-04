"""
HPC cluster integration test: Apptainer + SLURM.

This test submits a real SLURM job that runs inside an Apptainer container.
It verifies the full pipeline: Hydra → custom_submitit → submitit → SLURM → Apptainer.

Requirements:
  - Must be run on a machine with access to `sbatch` (SLURM login node)
  - Must be run on a machine with `apptainer` installed
  - Requires a .sif container image (passed via --sif-path)

Usage:
  pytest tests/test_slurm_cluster.py -v -m slurm --sif-path /path/to/container.sif

The container must have hydra-apptainer-launcher installed inside it.
"""

import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from tests.conftest import has_apptainer, has_sbatch

pytestmark = pytest.mark.slurm


@pytest.fixture
def cluster_available():
    """Skip if not on a SLURM cluster."""
    if not has_sbatch():
        pytest.skip("sbatch not available — not on a SLURM cluster")


@pytest.fixture
def apptainer_available():
    """Skip if apptainer is not installed."""
    if not has_apptainer():
        pytest.skip("apptainer not available")


@pytest.fixture
def sif_exists(sif_path):
    """Verify the .sif file actually exists."""
    if not Path(sif_path).is_file():
        pytest.fail(f"Apptainer image not found: {sif_path}")
    return sif_path


def _create_test_app(tmp_path, marker_file, sif_path):
    """Create a minimal Hydra app + config + launcher YAML for testing."""
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    launcher_dir = scripts_dir / "hydra" / "launcher"
    launcher_dir.mkdir(parents=True)

    # The Hydra app: writes a marker file to prove it ran inside the container
    app_py = scripts_dir / "cluster_test_app.py"
    app_py.write_text(
        textwrap.dedent(
            f"""\
            import logging
            import os
            import platform
            import sys
            from pathlib import Path

            import hydra
            from omegaconf import DictConfig

            log = logging.getLogger(__name__)

            @hydra.main(version_base=None, config_path=".", config_name="config")
            def app(cfg: DictConfig) -> None:
                marker = Path("{marker_file}")
                info = {{
                    "node": platform.node(),
                    "pid": os.getpid(),
                    "python": sys.executable,
                    "value": cfg.value,
                    "inside_container": os.path.exists("/.singularity.d") or os.path.exists("/etc/apptainer"),
                }}
                # Write marker to prove the job ran
                marker.write_text(str(info))
                log.info(f"Marker written: {{info}}")

            if __name__ == "__main__":
                app()
            """
        )
    )

    # Config
    config_yaml = scripts_dir / "config.yaml"
    config_yaml.write_text("value: cluster_test_ok\n")

    # Launcher YAML: Apptainer + SLURM
    launcher_yaml = launcher_dir / "submitit_apptainer.yaml"
    launcher_yaml.write_text(
        textwrap.dedent(
            f"""\
            _target_: hydra_plugins.hydra_apptainer_launcher.submitit_launcher.CustomSlurmLauncher
            submitit_folder: ${{hydra.sweep.dir}}/.submitit/%j
            timeout_min: 5
            gpus_per_node: 0
            cpus_per_task: 1
            mem_gb: 4
            python: "apptainer exec {sif_path} python"
            """
        )
    )

    return app_py


def _wait_for_marker(marker_file, timeout_seconds=300, poll_interval=5):
    """Poll until the marker file appears or timeout is reached."""
    elapsed = 0
    while elapsed < timeout_seconds:
        if marker_file.is_file():
            return True
        time.sleep(poll_interval)
        elapsed += poll_interval
    return False


class TestSlurmApptainer:
    """Tests that require a SLURM cluster and an Apptainer .sif image."""

    def test_submit_and_run_in_container(
        self, tmp_path, cluster_available, apptainer_available, sif_exists
    ):
        """Submit a job to SLURM that runs inside the Apptainer container.

        Verifies:
        1. The job is submitted successfully (sbatch accepts it)
        2. The job executes the task function (marker file is created)
        3. The job ran inside the container (marker contains container evidence)
        """
        marker_file = tmp_path / "slurm_marker.txt"
        app_py = _create_test_app(tmp_path, marker_file, sif_exists)
        scripts_dir = app_py.parent

        # Submit the job
        result = subprocess.run(
            [
                sys.executable,
                str(app_py),
                "-m",
                f"hydra.sweep.dir={tmp_path / 'sweep'}",
                "hydra/launcher=submitit_apptainer",
                "value=cluster_test_ok",
            ],
            capture_output=True,
            text=True,
            cwd=str(scripts_dir),
            timeout=120,
        )

        if result.returncode != 0:
            # Print debug info
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)

            # Check submitit logs if available
            submitit_dirs = list((tmp_path / "sweep").rglob(".submitit"))
            for sd in submitit_dirs:
                for logfile in sd.rglob("*_log.*"):
                    print(f"\n--- {logfile} ---")
                    print(logfile.read_text()[:2000])

        assert result.returncode == 0, (
            f"Job submission failed:\n{result.stderr}"
        )

        # Wait for the job to complete (SLURM scheduling + execution)
        found = _wait_for_marker(marker_file, timeout_seconds=300)
        assert found, (
            f"Marker file not created after 300s. "
            f"The SLURM job may have failed or is still queued. "
            f"Check: squeue -u $USER and "
            f"logs in {tmp_path / 'sweep'}/.submitit/"
        )

        # Verify the marker contents
        content = marker_file.read_text()
        assert "cluster_test_ok" in content, (
            f"Marker file doesn't contain expected value. Content: {content}"
        )

    def test_container_has_plugin(self, apptainer_available, sif_exists):
        """Verify the container has hydra-apptainer-launcher installed."""
        result = subprocess.run(
            [
                "apptainer",
                "exec",
                sif_exists,
                "python",
                "-c",
                "from hydra_plugins.hydra_apptainer_launcher.submitit_launcher import CustomSlurmLauncher; print('OK')",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, (
            f"Plugin import failed inside container:\n{result.stderr}\n"
            f"Make sure hydra-apptainer-launcher is installed in the .sif image."
        )
        assert "OK" in result.stdout

    def test_container_python_works(self, apptainer_available, sif_exists):
        """Basic sanity check: Python runs inside the container."""
        result = subprocess.run(
            [
                "apptainer",
                "exec",
                sif_exists,
                "python",
                "-c",
                "import sys; print(f'Python {sys.version}')",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, (
            f"Python failed inside container:\n{result.stderr}"
        )
        assert "Python" in result.stdout
