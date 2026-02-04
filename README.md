# Hydra Apptainer Launcher 🐍📦🚀

[Hydra](https://hydra.cc) is a powerful framework from Meta for elegantly configuring complex applications. It lets you compose configurations dynamically, override parameters from the command line, and run hyperparameter sweeps with minimal code changes.

This plugin extends Hydra's job launching capabilities by transparently executing your Python scripts inside Apptainer containers on HPC systems — ensuring all dependencies are available on compute nodes that lack your development environment.

**Features:**
- 🚀 Launch SLURM jobs that run inside Apptainer containers
- 🔧 Full control over SLURM resources (GPUs, CPUs, memory, partitions)
- 🔄 Native support for Hydra multirun (hyperparameter sweeps)
- 📦 Zero code changes to your Hydra applications
- 🧪 Local testing mode without SLURM


## The Problem

On HPC clusters, **compute nodes don't have your Python dependencies installed**. Apptainer containers carry the entire runtime environment to every node, and this plugin makes Hydra launch everything inside those containers transparently.

```
Login Node: atchbp                        Compute Node: compute-0-[0-10]
┌──────────────────┐                    ┌──────────────────────────────────────┐
│ python train.py  │    SLURM job       │ apptainer exec --nv container.sif   │
│                  │ ──────────────►    │   python train.py                   │
│ (launches job,   │   (sbatch)         │                                     │
│  does NOT run    │                    │ (runs INSIDE the container where    │
│  the training)   │                    │  all dependencies are installed)     │
└──────────────────┘                    └──────────────────────────────────────┘
         │                                         │
   Shared Filesystem: /path/to/my_project.sif  ◄───┘
```

## Install

```bash
pip install git+https://github.com/EduardoRosLab/hydra-apptainer-launcher.git
```

Or for development:
```bash
git clone https://github.com/EduardoRosLab/hydra-apptainer-launcher.git
pip install -e ./hydra-apptainer-launcher
```

## Quick Start

### 1. Create your Hydra app

```python
# scripts/train.py
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path=".", config_name="config")
def train(cfg: DictConfig) -> None:
    import torch  # imported INSIDE the container on the compute node
    print(f"Training with lr={cfg.lr} on {torch.cuda.device_count()} GPUs")

if __name__ == "__main__":
    train()
```

### 2. Create your app config

This YAML file contains your application's hyperparameters.

```yaml
# scripts/config.yaml
lr: 0.001
batch_size: 256
```

### 3. Create a launcher config

This YAML file tells Hydra to use the Apptainer + SLURM launcher and specify the SLURM resources.

```yaml
# scripts/hydra/launcher/submitit_apptainer.yaml
_target_: hydra_plugins.hydra_apptainer_launcher.submitit_launcher.CustomSlurmLauncher
submitit_folder: ${hydra.sweep.dir}/.submitit/%j
timeout_min: 360
partition: full
gpus_per_node: 1
cpus_per_task: 8
mem_gb: 32
python: "apptainer exec --nv ${hydra.runtime.cwd}/my_project.sif python"
```

### Project Structure

After installing the plugin, your project needs:

1. Launcher YAML file(s) in `scripts/hydra/launcher/` — adapted to your resources
2. A `Dockerfile` that installs your dependencies + this plugin (see [templates](examples/templates/))
3. A `container.sh` to build the `.sif` image

```
my_project/
├── scripts/
│   ├── train.py
│   ├── config.yaml
│   └── hydra/
│       └── launcher/
│           └── submitit_apptainer.yaml
├── Dockerfile
├── container.sh
└── requirements.txt    # includes: hydra-apptainer-launcher
```


### 4. Run

Hydra allows you to do parameter sweeps easily. Each combination becomes a separate SLURM job, each running inside the container.

```bash
# Single job on SLURM inside the container
python scripts/train.py -m hydra/launcher=submitit_apptainer

# Parameter sweep (each combination = 1 SLURM job)
python scripts/train.py -m hydra/launcher=submitit_apptainer lr=0.001,0.0001 batch_size=128,512
```

---

## How It Works

The official `hydra-submitit-launcher` plugin assumes the compute node has the same Python environment as the login node. On HPC clusters, that's not the case.

This plugin adds a **`python` parameter** that replaces the Python command submitit uses:

```
Without this plugin (official):
  submitit generates → python -u submitit_pickled_job.py

With this plugin:
  submitit generates → apptainer exec --nv container.sif python -u submitit_pickled_job.py
```

The entire execution — deserialization, config loading, your function — happens inside the container where all dependencies are installed.

### Full Flow

```
You (login node):
  python train.py -m hydra/launcher=submitit_apptainer lr=0.001,0.0001
     │
     ▼
Hydra: sees -m (multirun) → generates 2 configs (lr=0.001, lr=0.0001)
     │    → delegates to the launcher plugin specified in the YAML
     ▼
hydra-apptainer-launcher: creates submitit AutoExecutor
     │    → builds job parameters for each config combination
     │    → sets python="apptainer exec --nv container.sif python"
     ▼
submitit: generates sbatch script for each job
     │    → serializes (pickles) the task function + config
     │    → calls sbatch to submit to SLURM
     ▼
SLURM: schedules jobs on compute nodes
     │
     ▼
Compute node: runs the sbatch script, which calls:
  apptainer exec --nv /path/container.sif python -u submitit_pickled_job.py
     │
     ▼
Inside the container:
  - submitit deserializes the task function and config
  - Hydra applies the sweep overrides (lr=0.001 for job 1, lr=0.0001 for job 2)
  - Your training function runs with full access to all installed dependencies
  - Results are written to the shared filesystem
     │
     ▼
Login node: submitit collects results from the shared filesystem
```

---

## Setup

### Prerequisites

On the **login node** (where you launch jobs):
- Python 3.8+ with Hydra installed
- This plugin installed (`pip install hydra-apptainer-launcher`)
- Access to `sbatch` (SLURM commands)
- The `.sif` container file on a shared filesystem

**You do NOT need your project's heavy dependencies (JAX, PyTorch, etc.) on the login node.** Those live inside the container. You only need Hydra and this plugin to *submit* jobs.

## Building Your Container

The container must include **everything your code needs to run**: Python, all pip packages, your source code, this plugin, and any system libraries.

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install your Python dependencies
COPY requirements.txt /app/
RUN pip install -r requirements.txt

# Install this plugin INSIDE the container
# Required: submitit deserializes the launcher object on the compute node
RUN pip install git+https://github.com/EduardoRosLab/hydra-apptainer-launcher.git

# Install your project
COPY . /app/
RUN pip install -e .

CMD ["/bin/bash"]
```

**Why must `hydra-apptainer-launcher` be installed inside the container?**

When submitit runs on the compute node (inside the container), it deserializes the launcher object. That object is an instance of `CustomSlurmLauncher` from this package. If the package isn't inside the container, deserialization fails with `ImportError`.

### Build Script

```bash
#!/bin/bash
# container.sh — Build pipeline: Dockerfile → Docker → tar → Apptainer .sif

rm -f my_project.sif my_project.tar
docker rmi -f my_project

docker build . -t my_project
docker save -o my_project.tar my_project:latest
apptainer build my_project.sif docker-archive://my_project.tar
```

### Verify

```bash
# Interactive shell inside the container
apptainer shell --nv my_project.sif

# Quick import test
apptainer exec --nv my_project.sif python -c "import torch; print(torch.cuda.is_available())"

# Run your script (locally, no SLURM)
apptainer exec --nv my_project.sif python scripts/train.py
```

---

## Examples

See the [examples/](examples/) directory for runnable examples:

- **[hello_cluster](examples/hello_cluster/)** — Minimal app to verify the full pipeline works
- **[gpu_training](examples/gpu_training/)** — Simulated GPU training with hyperparameter sweeps
- **[templates](examples/templates/)** — Dockerfile and container build script templates

### Parameter Sweeps

Each parameter combination becomes a separate SLURM job, each running inside the container:

```bash
# 12 jobs = 3 lr x 2 batch_size x 2 seeds
python scripts/train.py -m \
  hydra/launcher=submitit_apptainer \
  lr=0.001,0.0003,0.0001 \
  batch_size=128,512 \
  seed=1,2
```

### Limit Parallel Jobs

```bash
python scripts/train.py -m \
  hydra/launcher=submitit_apptainer \
  hydra.launcher.array_parallelism=4 \
  lr=0.01,0.003,0.001,0.0003,0.0001
```

### Override Resources from CLI

```bash
python scripts/train.py -m \
  hydra/launcher=submitit_apptainer \
  hydra.launcher.timeout_min=720 \
  hydra.launcher.gpus_per_node=2 \
  hydra.launcher.partition=long_gpu
```

---

## Configuration Reference

### Base Parameters (all executors)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `submitit_folder` | str | `${hydra.sweep.dir}/.submitit/%j` | Where submitit stores job state/logs |
| `timeout_min` | int | `60` | Maximum job runtime in minutes |
| `cpus_per_task` | int | `null` | CPU cores per task |
| `gpus_per_node` | int | `null` | GPUs per node |
| `tasks_per_node` | int | `1` | Tasks per node |
| `mem_gb` | int | `null` | Memory reservation in GB |
| `nodes` | int | `1` | Number of nodes |
| `name` | str | `${hydra.job.name}` | SLURM job name |
| `stderr_to_stdout` | bool | `false` | Redirect stderr to stdout |
| `python` | str | `null` | **Custom Python command — set this for Apptainer** |

### SLURM-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `partition` | str | `null` | SLURM partition (queue) |
| `qos` | str | `null` | Quality of Service |
| `account` | str | `null` | Billing/project account |
| `constraint` | str | `null` | Hardware constraints (e.g., `a100`) |
| `exclude` | str | `null` | Nodes to exclude |
| `gres` | str | `null` | Generic resources (e.g., `gpu:a100:1`) |
| `cpus_per_gpu` | int | `null` | CPUs per GPU |
| `gpus_per_task` | int | `null` | GPUs per task |
| `mem_per_gpu` | str | `null` | Memory per GPU |
| `mem_per_cpu` | str | `null` | Memory per CPU |
| `signal_delay_s` | int | `120` | USR1 signal seconds before timeout |
| `max_num_timeout` | int | `0` | Max retries on timeout |
| `array_parallelism` | int | `256` | Max parallel jobs in array |
| `setup` | list[str] | `null` | Shell commands before srun in sbatch |
| `srun_args` | list[str] | `null` | Additional srun arguments |
| `additional_parameters` | dict | `{}` | Arbitrary SLURM parameters |


---

Check [submitit-launch](https://hydra.cc/docs/plugins/submitit_launcher/) for more details.

## Testing

```bash
pip install -r requirements-dev.txt

# Unit and local integration tests (no cluster needed)
pytest tests/test_plugin_discovery.py tests/test_local_launcher.py -v

# HPC cluster test (run ON the cluster, requires a .sif image)
pytest tests/test_slurm_cluster.py -v -m slurm --sif-path /path/to/container.sif
```



## License

MIT
