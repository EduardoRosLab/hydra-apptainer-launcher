# Running Examples

This guide demonstrates how to set up and run the hydra-apptainer-launcher examples.
## Setup

1. **Clone the repository:**
    ```bash
    git clone https://github.com/EduardoRosLab/hydra-apptainer-launcher.git
    cd hydra-apptainer-launcher
    ```

2. **Create and activate a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3. **Install the hydra-apptainer-launcher package:**
    ```bash
    pip install .
    ```

4. **Navigate to the examples directory:**
    ```bash
    cd examples/templates
    ```

5. **Build the Apptainer container:**
    ```bash
    chmod +x ./container.sh
    ./container.sh
    ```

## Running Examples

### Hello Cluster Example

Run a simple parameter sweep with different greeting messages:

```bash
python3 ../hello_cluster/hello.py -m hydra/launcher=submitit_apptainer +greetings="AAA","BBB","CCC"
```

This command:
- Uses the `-m` flag to enable multirun mode for parameter sweeps
- Specifies `hydra/launcher=submitit_apptainer` to use the Apptainer launcher
- Sweeps over three greeting values: "AAA", "BBB", and "CCC"

Expected output:
```
[2026-02-04 13:56:19,034][HYDRA] Submitit 'slurm' sweep output dir : multirun/2026-02-04/13-56-18
[2026-02-04 13:56:19,036][HYDRA] 	#0 : +greetings=AAA
[2026-02-04 13:56:19,050][HYDRA] 	#1 : +greetings=BBB
[2026-02-04 13:56:19,055][HYDRA] 	#2 : +greetings=CCC
```

### GPU Training Example

Run a hyperparameter sweep for a training job:

```bash
python3 ../gpu_training/train.py -m hydra/launcher=submitit_apptainer lr=0.001,0.0001 batch_size=128,512
```

This command:
- Sweeps over two learning rates: 0.001 and 0.0001
- Sweeps over two batch sizes: 128 and 512
- Creates 4 jobs total (2 × 2 parameter combinations)

Expected output:
```
[2026-02-04 13:56:37,771][HYDRA] Submitit 'slurm' sweep output dir : multirun/2026-02-04/13-56-37
[2026-02-04 13:56:37,772][HYDRA] 	#0 : lr=0.001 batch_size=128
[2026-02-04 13:56:37,777][HYDRA] 	#1 : lr=0.001 batch_size=512
[2026-02-04 13:56:37,782][HYDRA] 	#2 : lr=0.0001 batch_size=128
[2026-02-04 13:56:37,786][HYDRA] 	#3 : lr=0.0001 batch_size=512
```

### Viewing Job Results

Check the training logs for a specific job:

```bash
cat multirun/2026-02-04/13-56-37/0/train.log
```

Example log output showing GPU utilization and training progress:
```
[2026-02-04 13:56:38,832][__main__][INFO] - Training with lr=0.001, batch_size=128, seed=42
[2026-02-04 13:56:38,832][__main__][INFO] - Epochs: 100
[2026-02-04 13:56:38,989][__main__][INFO] - nvidia-smi output:
Wed Feb  4 13:56:38 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.211.01             Driver Version: 570.211.01     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5080        Off |   00000000:01:00.0 Off |                  N/A |
|  0%   34C    P8             14W /  360W |      15MiB /  16303MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

[2026-02-04 13:56:38,990][__main__][INFO] -   Epoch 0/100, loss=0.001000
[2026-02-04 13:56:38,990][__main__][INFO] -   Epoch 10/100, loss=0.000091
[2026-02-04 13:56:38,990][__main__][INFO] -   Epoch 20/100, loss=0.000048
...
[2026-02-04 13:56:38,990][__main__][INFO] - Training complete.
[2026-02-04 13:56:38,990][submitit][INFO] - Job completed successfully
```
