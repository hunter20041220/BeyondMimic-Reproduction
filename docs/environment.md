# Environment

## Lightweight CPU Smoke Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[analysis,dev]"
python scripts/00_setup/check_environment.py
```

## Torch Training Environment

```bash
pip install -r requirements/torch.txt
pip install -e .
```

The original reproduction work used Python 3.10 and PyTorch CUDA 12.1 wheels.
CPU is enough for the synthetic smoke pipeline; GPU is recommended for VAE and
denoiser training beyond tiny fixtures.

## IsaacLab Tracking Environment

Tracking-policy rollout requires external simulator dependencies that are not
vendored here:

```text
Isaac Sim 4.5
IsaacLab 2.1
RSL-RL
Unitree G1 assets
```

Install those according to the upstream IsaacLab documentation, then write
rollouts using the schema in `docs/data_format.md`.
