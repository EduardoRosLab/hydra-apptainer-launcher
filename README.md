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

### Prerequisites

On the **login node** (where you launch jobs):
- A python virtual environment so you can install the plugin
- Access to `sbatch` (SLURM commands)
- The `.sif` container file on a shared filesystem

**You do NOT need your project's heavy dependencies (JAX, PyTorch, etc.) on the login node.** Those live inside the container. You only need Hydra and this plugin to *submit* jobs.

### Installation on the venv
```bash
pip install git+https://github.com/EduardoRosLab/hydra-apptainer-launcher.git
```

Or for development or testing:
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

@hydra.main(version_base=None, config_path=".", config_name="config") # Hydra entry point. This is where the magic happens.
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

This YAML file tells Hydra to use the Apptainer + SLURM launcher and specify the SLURM resources. Check [submitit-slurm-launch](https://hydra.cc/docs/plugins/submitit_launcher/) for more details on available parameters.

```yaml
# scripts/hydra/launcher/submitit_apptainer.yaml
_target_: hydra_plugins.hydra_apptainer_launcher.submitit_launcher.CustomSlurmLauncher
submitit_folder: ${hydra.sweep.dir}/.submitit/%j
timeout_min: 360
partition: full
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

#parameter sweep with each job in a different node
python3 scripts/train.py -m \
    hydra/launcher=submitit_apptainer \
    +'hydra.launcher.additional_parameters={exclusive: ""}' \
    lr=0.001,0.0001 batch_size=128,512
```



### Output Structure

After running a multirun sweep, Hydra creates the following directory structure for each job:

```
multirun/
└── 2024-01-15/
    └── 12-34-56/                    # Timestamp of launch
        ├── .submitit/               # Submitit JOBS and launched script
        │   ├── JOB_ID/*sh          # SLURM job scripts
        │   └── JOB_ID_0/*        # Output and error logs per job 0
                    
        ├── 0/                       # First job (lr=0.001, batch_size=128)
        │   ├── .hydra/
        │   │   ├── config.yaml      # Resolved config for this job
        │   │   ├── hydra.yaml
        │   │   └── overrides.yaml
        │   └── train.log            # Your application output
        ├── 1/                       # Second job (lr=0.001, batch_size=512)
        │   ├── .hydra/
        │   └── train.log
        ├── 2/                       # Third job (lr=0.0001, batch_size=128)
        │   ├── .hydra/
        │   └── train.log
        ├── 3/                       # Fourth job (lr=0.0001, batch_size=512)
        │   ├── .hydra/
        │   └── train.log
        ├── .hydra/
        │   └── config.yaml          # Multirun config
        ├── multirun.yaml            # Summary of all runs
        └── optimization_results.yaml # Optional: results aggregation
```

Each numbered subdirectory (`0/`, `1/`, `2/`, ...) corresponds to one parameter combination and contains:
- `.hydra/config.yaml` — the fully resolved configuration for that specific run
- Your application's outputs (logs, checkpoints, etc.)

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


## Testing

```bash
pip install -r requirements-dev.txt

# Unit and local integration tests (no cluster needed)
pytest tests/test_plugin_discovery.py tests/test_local_launcher.py -v

# HPC cluster test (run ON the cluster, requires a .sif image)
pytest tests/test_slurm_cluster.py -v -m slurm --sif-path /path/to/container.sif
```


## Troubleshooting

- **Pickle error on compute node**: Ensure same python version on apptainer and login node. Ensure `hydra-apptainer-launcher` is installed inside the container.

## License

MIT
