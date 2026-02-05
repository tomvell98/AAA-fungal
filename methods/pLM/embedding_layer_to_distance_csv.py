#!/usr/bin/env python3
"""
Convert a saved embedding array into a square distance matrix CSV.
Supports 2D arrays (n_sequences x embedding_dim) and layered 3D arrays
(n_sequences x n_layers x embedding_dim or n_layers x n_sequences x embedding_dim).
"""

import argparse
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def load_embeddings(path):
    """Load embeddings from a .npy file and report shape."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embedding file not found: {path}")
    arr = np.load(path)
    print(f"Loaded embeddings with shape {arr.shape}")
    return arr


def pick_embedding_layer(arr, layer=None, auto_squeeze=True):
    """Select a 2D (n_sequences x embedding_dim) view from an embedding array."""
    working = np.squeeze(arr) if auto_squeeze else arr

    if working.ndim == 2:
        return working

    if working.ndim == 3:
        if layer is None:
            raise ValueError("Embedding array has three dimensions; specify --layer to choose one.")

        # Try (n_sequences, n_layers, dim)
        if layer < working.shape[1]:
            selected = working[:, layer, :]
        # Try (n_layers, n_sequences, dim)
        elif layer < working.shape[0]:
            selected = working[layer, :, :]
        else:
            raise ValueError(f"Layer index {layer} is out of range for shape {working.shape}.")

        selected = np.squeeze(selected)
        if selected.ndim != 2:
            raise ValueError(f"Selected slice has shape {selected.shape}; expected 2D after choosing a layer.")
        return selected

    raise ValueError(f"Unsupported embedding shape {working.shape}; need a 2D array or a single selected layer.")


def load_headers(ids_file, n_items):
    """Load sequence IDs or generate generic labels."""
    if ids_file:
        with open(ids_file, "r") as handle:
            headers = [line.strip() for line in handle if line.strip()]
        if len(headers) != n_items:
            raise ValueError(
                f"ID count ({len(headers)}) does not match number of sequences ({n_items})."
            )
        return headers
    return [f"seq_{i:04d}" for i in range(n_items)]


def compute_distance_matrix(embeddings, metric):
    """Compute an all-vs-all distance matrix."""
    print(f"Calculating {metric} distances...")
    distmat = cdist(embeddings, embeddings, metric=metric)
    print(f"Distance matrix shape: {distmat.shape}")
    return distmat


def main():
    parser = argparse.ArgumentParser(
        description="Extract one embedding layer and write its distance matrix as CSV."
    )
    parser.add_argument("embedding_file", help="Path to .npy embeddings")
    parser.add_argument(
        "--ids",
        dest="ids_file",
        help="Optional text file with one sequence ID per line to label rows/columns.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        help="Layer index to select when the embedding array has multiple layers.",
    )
    parser.add_argument(
        "--metric",
        default="cosine",
        choices=["cosine", "euclidean", "manhattan", "correlation"],
        help="Distance metric for cdist.",
    )
    parser.add_argument(
        "--output_csv",
        help="Exact output CSV path. Overrides --output_dir if provided.",
    )
    parser.add_argument(
        "--output_dir",
        help="Directory for outputs (default: alongside the embedding file).",
    )
    parser.add_argument(
        "--no-squeeze",
        dest="auto_squeeze",
        action="store_false",
        help="Disable automatic removal of singleton dimensions before layer selection.",
    )
    parser.set_defaults(auto_squeeze=True)

    args = parser.parse_args()

    embeddings = load_embeddings(args.embedding_file)
    selected = pick_embedding_layer(
        embeddings, layer=args.layer, auto_squeeze=args.auto_squeeze
    )

    headers = load_headers(args.ids_file, selected.shape[0])
    distmat = compute_distance_matrix(selected, args.metric)

    base_name = os.path.splitext(os.path.basename(args.embedding_file))[0]
    layer_suffix = f"_layer{args.layer}" if args.layer is not None else ""

    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.embedding_file))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    output_csv = args.output_csv or os.path.join(
        out_dir, f"{base_name}{layer_suffix}_{args.metric}_distance_matrix.csv"
    )

    df = pd.DataFrame(distmat, index=headers, columns=headers)
    df.to_csv(output_csv)
    print(f"Wrote distance matrix to {output_csv}")


if __name__ == "__main__":
    main()
