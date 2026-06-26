# Troubleshooting

## `ModuleNotFoundError: beyondmimic_repro`

Install the package in editable mode:

```bash
pip install -e .
```

or run scripts with `PYTHONPATH=src`.

## Torch Is Missing

The data and guidance smoke tests can run without torch. VAE and denoiser
training need:

```bash
pip install -r requirements/torch.txt
```

## Environment Sanity Check

Run:

```bash
python scripts/00_setup/check_environment.py
```

It records Python, NumPy, optional Torch availability, and package version in
`outputs/metrics/environment_check.json`.

## No LAFAN1 CSV Files Found

Put retargeted CSV files under `data/raw/lafan1`, or run:

```bash
python scripts/01_data/make_synthetic_fixture.py
```

to generate a tiny local fixture.
