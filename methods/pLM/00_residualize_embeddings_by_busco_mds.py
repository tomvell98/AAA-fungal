#!/usr/bin/env python3
"""
00_residualize_embeddings_by_busco_mds.py

Residualize embedding VECTORS against genome-wide (BUSCO) distances.

Inputs:
  --embeddings_npy : (n x d) OR (n x layers x d) etc. (use --layer if needed)
  --ids_txt        : one seq ID per line, matching embedding rows
  --busco_dist_csv : square distance matrix with row/col labels = IDs

Method:
  1) Align by shared IDs
  2) Compute MDS/PCoA coordinates from BUSCO distances (k dims)
  3) Fit linear regression E ~ intercept + X (least squares)
  4) Output residual embeddings E_resid (n x d), plus BUSCO coords

Outputs:
  --out_resid_npy
  --out_aligned_ids_txt
  --out_busco_coords_tsv
  --out_fit_txt
"""

import argparse
import os
import numpy as np
import pandas as pd

from scipy.spatial.distance import squareform
from sklearn.manifold import MDS

# --- reuse your layer selection logic (simplified) ---
def load_embeddings(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    arr = np.load(path)
    return arr

def pick_embedding_layer(arr, layer=None):
    w = np.squeeze(arr)
    if w.ndim == 2:
        return w
    if w.ndim == 3:
        if layer is None:
            raise ValueError("Embeddings are 3D; provide --layer.")
        # (n, L, d) or (L, n, d)
        if layer < w.shape[1]:
            sel = w[:, layer, :]
        elif layer < w.shape[0]:
            sel = w[layer, :, :]
        else:
            raise ValueError(f"Layer {layer} out of range for {w.shape}")
        sel = np.squeeze(sel)
        if sel.ndim != 2:
            raise ValueError(f"Selected slice is not 2D: {sel.shape}")
        return sel
    raise ValueError(f"Unsupported embedding array shape: {w.shape}")

def load_ids(ids_txt, n):
    ids = [ln.strip() for ln in open(ids_txt) if ln.strip()]
    if len(ids) != n:
        raise ValueError(f"ids_txt has {len(ids)} IDs, embeddings have {n} rows.")
    return [str(x) for x in ids]

def load_square_dist(path):
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    if df.shape[0] != df.shape[1]:
        raise ValueError(f"BUSCO matrix not square: {df.shape}")
    return df

def align_all(E, ids, busco_df):
    common = sorted(set(ids).intersection(set(busco_df.index)))
    if len(common) < 10:
        raise ValueError(f"Too few shared IDs after intersection: {len(common)}")

    id_to_i = {sid:i for i, sid in enumerate(ids)}
    idx = [id_to_i[sid] for sid in common]
    E2 = E[idx, :]
    B2 = busco_df.loc[common, common].values.astype(float)
    return E2, common, B2

def fit_linear_residuals(E, X):
    """
    E: (n x d), X: (n x k)
    Fit E ~ intercept + X  => residuals: E - (1,X)B
    """
    n = E.shape[0]
    X1 = np.column_stack([np.ones(n), X])  # (n x (k+1))
    # Solve for B in least squares: X1 B = E
    B, _, _, _ = np.linalg.lstsq(X1, E, rcond=None)  # B: (k+1 x d)
    E_hat = X1 @ B
    resid = E - E_hat
    return resid, B

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings_npy", required=True)
    ap.add_argument("--ids_txt", required=True)
    ap.add_argument("--busco_dist_csv", required=True)

    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--mds_dim", type=int, default=20)
    ap.add_argument("--mds_random_state", type=int, default=1)

    ap.add_argument("--out_resid_npy", required=True)
    ap.add_argument("--out_aligned_ids_txt", required=True)
    ap.add_argument("--out_busco_coords_tsv", required=True)
    ap.add_argument("--out_fit_txt", required=True)
    args = ap.parse_args()

    arr = load_embeddings(args.embeddings_npy)
    E = pick_embedding_layer(arr, layer=args.layer)
    ids = load_ids(args.ids_txt, E.shape[0])
    busco_df = load_square_dist(args.busco_dist_csv)

    E2, common_ids, B = align_all(E, ids, busco_df)

    # MDS on BUSCO distances (metric MDS)
    mds = MDS(
        n_components=args.mds_dim,
        dissimilarity="precomputed",
        random_state=args.mds_random_state,
        n_init=4,
        max_iter=300,
        normalized_stress="auto",
    )
    X = mds.fit_transform(B)  # (n x k)

    E_resid, coef = fit_linear_residuals(E2, X)

    np.save(args.out_resid_npy, E_resid)
    with open(args.out_aligned_ids_txt, "w") as f:
        for sid in common_ids:
            f.write(sid + "\n")

    coords_df = pd.DataFrame(X, index=common_ids, columns=[f"busco_mds{i+1}" for i in range(X.shape[1])])
    coords_df.to_csv(args.out_busco_coords_tsv, sep="\t")

    # simple fit summary
    with open(args.out_fit_txt, "w") as f:
        f.write(f"n\t{E_resid.shape[0]}\n")
        f.write(f"embedding_dim\t{E_resid.shape[1]}\n")
        f.write(f"mds_dim\t{X.shape[1]}\n")
        f.write(f"mds_stress\t{mds.stress_}\n")

    print("Wrote residual embeddings:", args.out_resid_npy)
    print("Wrote aligned IDs:", args.out_aligned_ids_txt)
    print("Wrote BUSCO MDS coords:", args.out_busco_coords_tsv)
    print("Wrote fit summary:", args.out_fit_txt)

if __name__ == "__main__":
    main()
