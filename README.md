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
	- Run `download_data.sh` to download BIOQIC data files
	- Run `notebooks/BIOQIC-data-exploration.ipynb` to view data
	- Run `matlab -batch MDEV` to perform elasticity inversion
	- Run `notebooks/MDEV-inversion-method.pynb` to replicate

- Aim 2: Elasticity reconstruction with SIREN
	- Run `notebooks/SIREN-testing.ipynb` to train on toy data
	- Run `notebooks/MRE-SIREN-training.ipynb` to train on BIOQIC
	- TODO solve Helmholtz equation using SIREN

