#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Monkey Face Aging – GFT-GCN Pipeline

Features
--------
• Graph Fourier Transform decomposition
• GraphSAGE age regression
• 10-fold cross validation
• Spectral phenomics extraction
• Ridge regression baseline

Project structure
-----------------
project/
│
├── scripts/
│   └── train_monkey_gftgcn_cv10.py
│
├── data/
│   ├── monkey_ID_age_sex.txt
│   └── input_obj/
│
└── results/

Requirements
------------
torch
torch-geometric
trimesh
scipy
numpy
"""

import os
import csv
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import trimesh

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected
from torch_geometric.nn import SAGEConv, global_mean_pool, global_max_pool

import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


def find_template(obj_dir):
    for f in os.listdir(obj_dir):
        if f.endswith(".obj"):
            return os.path.join(obj_dir, f)
    raise RuntimeError("No obj file found for template")


# ------------------------------------------------------------
# Graph construction
# ------------------------------------------------------------

def build_edge_index(template_obj):

    mesh = trimesh.load(template_obj, process=False)

    edges = set()

    for a, b, c in mesh.faces:
        edges.add((a, b))
        edges.add((b, c))
        edges.add((c, a))

    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()

    return to_undirected(edge_index)


# ------------------------------------------------------------
# Label parsing
# ------------------------------------------------------------

def parse_monkey_txt(txt, obj_dir):

    rows = []

    with open(txt) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            p = line.split()

            sid = p[0]

            try:
                age = float(p[1])
            except:
                continue

            obj = os.path.join(obj_dir, f"{sid}.obj")

            if os.path.exists(obj):

                rows.append(("MONKEY", sid, age, obj))

    return rows


# ------------------------------------------------------------
# GFT
# ------------------------------------------------------------

def edge_to_adj(edge_index, n):

    row = edge_index[0].numpy()
    col = edge_index[1].numpy()

    data = np.ones(len(row))

    A = sp.coo_matrix((data, (row, col)), shape=(n, n)).tocsr()

    return A


def compute_gft(edge_index, n, k):

    A = edge_to_adj(edge_index, n)

    deg = np.asarray(A.sum(axis=1)).flatten()

    D = sp.diags(deg)

    L = D - A

    lam, U = eigsh(L, k=k, which="SM")

    return U.astype(np.float32), lam.astype(np.float32)


def split_bands(K, low=0.85, mid=0.10):

    k1 = int(K * low)

    k2 = int(K * (low + mid))

    return k1, k2


def reconstruct(V, U, k1, k2):

    V_hat = U.T @ V

    low = U[:, :k1] @ V_hat[:k1]

    mid = U[:, k1:k2] @ V_hat[k1:k2]

    high = U[:, k2:] @ V_hat[k2:]

    return low, mid, high


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

class MeshDataset(torch.utils.data.Dataset):

    def __init__(self, rows, edge_index, U, k1, k2):

        self.rows = rows
        self.edge_index = edge_index
        self.U = U
        self.k1 = k1
        self.k2 = k2

    def __len__(self):

        return len(self.rows)

    def __getitem__(self, i):

        cohort, sid, age, obj = self.rows[i]

        mesh = trimesh.load(obj, process=False)

        V = mesh.vertices.astype(np.float32)

        V = V - V.mean(0)

        low, mid, high = reconstruct(V, self.U, self.k1, self.k2)

        x = torch.from_numpy(low)

        y = torch.tensor(age, dtype=torch.float32)

        return Data(x=x, edge_index=self.edge_index, y=y)


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

class GNN(torch.nn.Module):

    def __init__(self, in_dim=3, hidden=64):

        super().__init__()

        self.c1 = SAGEConv(in_dim, hidden)
        self.c2 = SAGEConv(hidden, hidden)
        self.c3 = SAGEConv(hidden, hidden)

        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden * 2, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, data):

        x, ei, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.c1(x, ei))
        x = F.relu(self.c2(x, ei))
        x = F.relu(self.c3(x, ei))

        xm = global_mean_pool(x, batch)
        xM = global_max_pool(x, batch)

        x = torch.cat([xm, xM], 1)

        return self.head(x).squeeze(1)


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device):

    model.eval()

    ys = []
    ps = []

    for batch in loader:

        batch = batch.to(device)

        p = model(batch)

        ys.append(batch.y.cpu().numpy())
        ps.append(p.cpu().numpy())

    y = np.concatenate(ys)
    p = np.concatenate(ps)

    mae = np.mean(np.abs(p - y))

    rmse = np.sqrt(np.mean((p - y) ** 2))

    return mae, rmse


# ------------------------------------------------------------
# CV split
# ------------------------------------------------------------

def kfold(n, k=10):

    idx = np.arange(n)

    np.random.shuffle(idx)

    return np.array_split(idx, k)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--out_dir", default="results")

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=2)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    set_seed()

    txt = os.path.join(args.data_dir, "monkey_ID_age_sex.txt")
    obj_dir = os.path.join(args.data_dir, "input_obj")

    rows = parse_monkey_txt(txt, obj_dir)

    print("Samples:", len(rows))

    template = find_template(obj_dir)

    edge_index = build_edge_index(template)

    mesh = trimesh.load(template, process=False)

    U, lam = compute_gft(edge_index, mesh.vertices.shape[0], 256)

    k1, k2 = split_bands(U.shape[1])

    folds = kfold(len(rows))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = []

    for i in range(10):

        print("\nFOLD", i + 1)

        test_idx = folds[i]

        train_idx = np.concatenate([folds[j] for j in range(10) if j != i])

        train_rows = [rows[j] for j in train_idx]
        test_rows = [rows[j] for j in test_idx]

        train_ds = MeshDataset(train_rows, edge_index, U, k1, k2)
        test_ds = MeshDataset(test_rows, edge_index, U, k1, k2)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size)

        model = GNN().to(device)

        opt = torch.optim.Adam(model.parameters(), lr=3e-4)

        for e in range(args.epochs):

            model.train()

            for batch in train_loader:

                batch = batch.to(device)

                p = model(batch)

                loss = F.l1_loss(p, batch.y)

                opt.zero_grad()

                loss.backward()

                opt.step()

        mae, rmse = evaluate(model, test_loader, device)

        print("MAE", mae, "RMSE", rmse)

        results.append(mae)

    print("\nCV MAE:", np.mean(results), "+-", np.std(results))


if __name__ == "__main__":
    main()