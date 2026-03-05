# Monkey Face Aging with GFT-GCN

Graph spectral neural network pipeline for predicting age from 3D monkey face meshes.

## Features

- Graph Fourier Transform (GFT) decomposition
- GraphSAGE age regression
- 10-fold cross validation
- Spectral phenomics extraction
- Ridge regression baseline

## Project structure
project/
│
├── notebooks
│ └── train_monkey_gftgcn_cv10.ipynb
│
├── data
│ ├── monkey_ID_age_sex.txt
│ └── input_obj/
│
└── results/


## Requirements
torch
torch-geometric
trimesh
scipy
numpy

## Run

```bash
python train_monkey_gftgcn_cv10.py
