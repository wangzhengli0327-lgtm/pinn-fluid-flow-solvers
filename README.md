# PINN Fluid Flow Solvers

Physics-informed neural network (PINN) implementations for nonlinear
fluid-flow benchmark problems. The repository contains four experiments:

1. A forward Burgers equation problem with a known exact solution.
2. A forward one-dimensional Burgers equation problem.
3. Steady lid-driven cavity flow at Reynolds number 100.
4. Inverse identification of the diffusion coefficient in the Burgers equation.

## Repository layout

```text
src/                         Shared models, sampling, gradients, metrics, and plotting
task1_burgers_exact.py       Burgers equation with an exact solution
task2_burgers_forward.py     Forward Burgers equation solver
task3_cavity_Re100.py        Lid-driven cavity PINN solver
task4_burgers_inverse.py     Inverse Burgers parameter identification
requirements.txt             Python dependencies
```

Generated figures, model checkpoints, training logs, reports, course materials,
and reference datasets are intentionally excluded.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build separately if GPU acceleration is required.

## Reference data

Tasks 2, 3, and 4 use external reference datasets for evaluation or supervised
samples. Place the following files in a local `data/` directory:

```text
data/Burgers_star_data.txt
data/Cavity_star_data_Re100.txt
```

Each text file must contain a header followed by at least three numeric columns.
The datasets are not distributed in this repository.

## Usage

Run a short smoke-training configuration:

```bash
python task1_burgers_exact.py --mode debug
python task2_burgers_forward.py --mode debug
python task3_cavity_Re100.py --mode debug
python task4_burgers_inverse.py --mode debug
```

Tasks 2–4 require their corresponding reference data files. Use `--help` on any
script to inspect its available training and data options:

```bash
python task3_cavity_Re100.py --help
```

Training outputs are written locally to `figures/`, `checkpoints/`, and `logs/`;
these generated directories are ignored by Git.
