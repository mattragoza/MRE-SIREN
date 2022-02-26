# SIREN for magnetic resonance elastography

The goal of this project is to perform elasticity reconstruction for magnetic resonance elastography using sinusoidal representation networks (SIRENs) with physical constraints.

## Environment setup

```
conda env create -n MRE-SIREN --file environment.yml
conda activate MRE-SIREN
python -m ipykernel install --user --name=MRE-SIREN
```

## Project status

- Aim 1: Replicate MDEV.m in python
	- Run `download_data.sh` to download data files
	- Run `MDEV.m` with matlab to compute elasticity
	- Run `BIOQIC-data-exploration.ipynb` to replicate

- Aim 2: Train phantom SIREN model
	- TODO
