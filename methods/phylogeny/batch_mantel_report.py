#!/usr/bin/env python3
"""Batch Mantel + regression + RF summary across enzyme matrices.

Finds enzyme distance matrices named full_clean1_mat_*.csv in the
trimmed_alignments folder, picks the corresponding tree (preferring
*_phylogeny_no_outliers_redo.treefile when present, otherwise
*_phylogeny_no_outliers.treefile), and compares each to the BUSCO matrix/tree.

Outputs a CSV summary with Mantel r/p/n, regression slope/intercept/R^2 and
95% CIs (taxon bootstrap), optional per-matrix normalisation (unit or z-score),
and RF distances.
"""

import argparse
import glob
import os
import sys
import subprocess
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from skbio.stats.distance import mantel

from mantel_from_csv import (
    load_distance_matrix,
    _upper_tri_vectors,
    _linear_regression_stats,
    _bootstrap_regression,
    _fit_scale,
    _normalise_distance_matrix,
    _load_tree,
    _prune_to_common_tips,
    _robinson_foulds,
)


def pick_tree(matrix_dir: str, gene: str) -> str | None:
    """Choose redo tree if present, else the standard no_outliers tree."""

    redo = os.path.join(matrix_dir, f"{gene}_phylogeny_no_outliers_redo.treefile")
    std = os.path.join(matrix_dir, f"{gene}_phylogeny_no_outliers.treefile")
    if os.path.exists(redo):
        return redo
    if os.path.exists(std):
        return std
    return None


def process_one(
    mat_path: str,
    busco_mat: str,
    tree1_path: str | None,
    tree2_path: str,
    permutations: int,
    method: str,
    alternative: str,
    reg_bootstrap: int,
    reg_ci: float,
    reg_bootstrap_mode: str,
    stratify_phylum: bool,
    stratify_phylum_pairs: bool,
    taxonomy_csv: str | None,
    tax_acc_col: str,
    tax_phy_col: str,
    deviation_perms: int,
    deviation_alpha: float,
    deviation_min_effect: float,
    deviation_fdr: bool,
    stratified_out_dir: str | None,
    script_dir: str,
    normalise: str,
    scale_matrix1: bool,
    scale_matrix1_with_intercept: bool,
    scale_min: float,
) -> dict:
    # Enforce per-process BLAS/NumPy threading limit
    thread_cap = os.environ.get("OMP_NUM_THREADS") or "1"
    os.environ["OMP_NUM_THREADS"] = thread_cap
    os.environ["OPENBLAS_NUM_THREADS"] = thread_cap
    os.environ["MKL_NUM_THREADS"] = thread_cap
    os.environ["NUMEXPR_NUM_THREADS"] = thread_cap
    gene = os.path.basename(mat_path).removeprefix("full_clean1_mat_").removesuffix(".csv")
    row: dict[str, object] = {
        "gene": gene,
        "matrix_path": mat_path,
        "tree1": tree1_path,
        "normalise": normalise,
    }

    try:
        dm1 = load_distance_matrix(mat_path)
        dm2 = load_distance_matrix(busco_mat)
    except Exception as e:
        row["error"] = f"load_error: {e}"
        return row

    common_labels = dm1.index.intersection(dm2.index)
    row["shared_labels"] = int(len(common_labels))
    if len(common_labels) < 3:
        row["error"] = "<3 shared labels"
        return row

    dm1_aligned = dm1.loc[common_labels, common_labels]
    dm2_aligned = dm2.loc[common_labels, common_labels]

    if scale_matrix1 and scale_matrix1_with_intercept:
        row["error"] = "scale_flags_conflict"
        return row

    if normalise != "none" and (scale_matrix1 or scale_matrix1_with_intercept):
        row["error"] = "normalise_scale_conflict"
        return row

    if normalise != "none":
        try:
            dm1_aligned, stats1 = _normalise_distance_matrix(dm1_aligned, mode=normalise)
            dm2_aligned, stats2 = _normalise_distance_matrix(dm2_aligned, mode=normalise)
            row.update(
                {
                    "matrix1_norm_max": stats1.get("max"),
                    "matrix1_norm_mean": stats1.get("mean"),
                    "matrix1_norm_std": stats1.get("std"),
                    "matrix2_norm_max": stats2.get("max"),
                    "matrix2_norm_mean": stats2.get("mean"),
                    "matrix2_norm_std": stats2.get("std"),
                }
            )
        except ValueError as e:
            row["error"] = f"normalise_error: {e}"
            return row

    if scale_matrix1 or scale_matrix1_with_intercept:
        slope_s, intercept_s = _fit_scale(
            dm1_aligned.values, dm2_aligned.values, with_intercept=scale_matrix1_with_intercept
        )
        scaled_vals = dm1_aligned.values * slope_s + intercept_s
        if scale_min is not None:
            scaled_vals = np.maximum(scaled_vals, scale_min)
        np.fill_diagonal(scaled_vals, 0.0)
        dm1_aligned = pd.DataFrame(scaled_vals, index=dm1_aligned.index, columns=dm1_aligned.columns)
        row["scale_slope"] = slope_s
        row["scale_intercept"] = intercept_s

    try:
        r, p_value, n = mantel(
            dm1_aligned.values,
            dm2_aligned.values,
            method=method,
            permutations=permutations,
            alternative=alternative,
        )
        row["mantel_r"] = float(r)
        row["mantel_p"] = float(p_value)
        row["mantel_n"] = int(n)
    except Exception as e:
        row["error"] = f"mantel_error: {e}"
        return row

    x, y = _upper_tri_vectors(dm1_aligned.values, dm2_aligned.values)
    slope, intercept, r_lin, r2 = _linear_regression_stats(x, y)
    row.update({
        "slope": slope,
        "intercept": intercept,
        "pearson_r": r_lin,
        "r2": r2,
    })

    if reg_bootstrap > 0:
        rng = np.random.default_rng(seed=42)
        ci = _bootstrap_regression(
            dm1_aligned.values,
            dm2_aligned.values,
            n_reps=reg_bootstrap,
            ci=reg_ci,
            rng=rng,
            mode=reg_bootstrap_mode,
        )
        row["slope_ci_low"], row["slope_ci_high"] = ci["slope_ci"]
        row["intercept_ci_low"], row["intercept_ci_high"] = ci["intercept_ci"]
        row["r_ci_low"], row["r_ci_high"] = ci["r_ci"]

    if tree1_path is None or not os.path.exists(tree1_path):
        row["rf"] = None
        row["rf_normalised"] = None
        row["shared_tips"] = None
        row.setdefault("error", "tree1_missing")
    else:
        try:
            tree1 = _load_tree(tree1_path)
            tree2 = _load_tree(tree2_path)
            pruned1, pruned2, common_tips = _prune_to_common_tips(tree1, tree2)
            row["shared_tips"] = len(common_tips)
            rf_distance = _robinson_foulds(pruned1, pruned2)
            denom = 2 * (len(common_tips) - 3)
            row["rf"] = rf_distance
            row["rf_normalised"] = rf_distance / denom if denom > 0 else None
        except Exception as e:
            row.setdefault("error", f"tree_error: {e}")

    # Optionally run stratified analysis via mantel_from_csv CLI to produce per-matrix CSV
    if stratified_out_dir and (stratify_phylum or stratify_phylum_pairs):
        os.makedirs(stratified_out_dir, exist_ok=True)
        out_path = os.path.join(stratified_out_dir, f"{gene}_stratified.csv")
        cmd = [
            sys.executable,
            os.path.join(script_dir, "mantel_from_csv.py"),
            mat_path,
            busco_mat,
        ]
        if tree1_path:
            cmd.extend(["--tree1", tree1_path, "--tree2", tree2_path])
        cmd.extend([
            "--method", method,
            "--permutations", str(permutations),
            "--alternative", alternative,
            "--linear-regression",
            "--reg-bootstrap", str(reg_bootstrap),
            "--reg-bootstrap-mode", reg_bootstrap_mode,
            "--reg-ci", str(reg_ci),
        ])
        cmd.extend(["--normalise", normalise])
        if scale_matrix1:
            cmd.append("--scale-matrix1")
        if scale_matrix1_with_intercept:
            cmd.append("--scale-matrix1-with-intercept")
        cmd.extend(["--scale-min", str(scale_min)])
        if stratify_phylum:
            cmd.append("--stratify-phylum")
        if stratify_phylum_pairs:
            cmd.append("--stratify-phylum-pairs")
        if taxonomy_csv:
            cmd.extend([
                "--taxonomy-csv", taxonomy_csv,
                "--taxonomy-accession-col", tax_acc_col,
                "--taxonomy-phylum-col", tax_phy_col,
            ])
        if deviation_perms > 0:
            cmd.extend([
                "--deviation-permutations", str(deviation_perms),
                "--deviation-alpha", str(deviation_alpha),
                "--deviation-min-effect", str(deviation_min_effect),
            ])
        if deviation_fdr:
            cmd.append("--deviation-fdr")
        cmd.extend(["--stratified-csv-out", out_path])

        try:
            # Keep threading confined per worker
            env = os.environ.copy()
            thread_cap_env = env.get("OMP_NUM_THREADS", "1")
            env["OMP_NUM_THREADS"] = thread_cap_env
            env["OPENBLAS_NUM_THREADS"] = thread_cap_env
            env["MKL_NUM_THREADS"] = thread_cap_env
            env["NUMEXPR_NUM_THREADS"] = thread_cap_env
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
            row["stratified_csv"] = out_path
        except subprocess.CalledProcessError as e:
            row.setdefault("error", f"stratified_error: {e}")

    return row


def main():
    parser = argparse.ArgumentParser(description="Batch Mantel/regression/RF summaries for enzyme matrices.")
    parser.add_argument(
        "--matrix-dir",
        required=True,
        help="Directory containing full_clean1_mat_*.csv and trees",
    )
    parser.add_argument(
        "--busco-matrix",
        required=True,
        help="BUSCO distance matrix path",
    )
    parser.add_argument(
        "--busco-tree",
        required=True,
        help="BUSCO tree path",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=999,
        help="Permutations for Mantel",
    )
    parser.add_argument(
        "--method",
        choices=["pearson", "spearman"],
        default="pearson",
        help="Correlation method for Mantel and regression reporting",
    )
    parser.add_argument(
        "--alternative",
        choices=["two-sided", "greater", "less"],
        default="two-sided",
        help="Alternative hypothesis for Mantel",
    )
    parser.add_argument(
        "--reg-bootstrap",
        type=int,
        default=500,
        help="Taxon bootstrap replicates for regression CIs",
    )
    parser.add_argument(
        "--reg-ci",
        type=float,
        default=0.95,
        help="Confidence level for regression CIs",
    )
    parser.add_argument(
        "--reg-bootstrap-mode",
        choices=["permute", "taxon"],
        default="taxon",
        help="Bootstrap mode for regression CIs (permute or taxon)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes",
    )
    parser.add_argument(
        "--omp-threads",
        type=int,
        default=1,
        help="Threads per worker for BLAS/NumPy (sets OMP_NUM_THREADS)",
    )
    parser.add_argument(
        "--output",
        default="batch_mantel_summary.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--stratify-phylum",
        action="store_true",
        help="Also run stratified per-phylum analysis via mantel_from_csv for each matrix",
    )
    parser.add_argument(
        "--stratify-phylum-pairs",
        action="store_true",
        help="Also run stratified per-phylum-pair analysis via mantel_from_csv for each matrix",
    )
    parser.add_argument(
        "--taxonomy-csv",
        default=None,
        help="Taxonomy CSV for stratified analyses",
    )
    parser.add_argument(
        "--taxonomy-accession-col",
        default="Accession",
        help="Accession column in taxonomy CSV",
    )
    parser.add_argument(
        "--taxonomy-phylum-col",
        default="Phylum",
        help="Phylum column in taxonomy CSV",
    )
    parser.add_argument(
        "--deviation-permutations",
        type=int,
        default=0,
        help="Label permutations for Δslope testing in stratified analyses",
    )
    parser.add_argument(
        "--deviation-alpha",
        type=float,
        default=0.05,
        help="Alpha for deviation testing in stratified analyses",
    )
    parser.add_argument(
        "--deviation-min-effect",
        type=float,
        default=0.0,
        help="Minimum |Δslope| to flag deviation in stratified analyses",
    )
    parser.add_argument(
        "--deviation-fdr",
        action="store_true",
        help="Apply BH FDR to stratified deviation tests",
    )
    parser.add_argument(
        "--scale-matrix1",
        action="store_true",
        help="Linearly scale matrix1 to BUSCO via slope (through origin).",
    )
    parser.add_argument(
        "--scale-matrix1-with-intercept",
        action="store_true",
        help="Scale matrix1 to BUSCO using slope+intercept fit (diagonal reset to 0).",
    )
    parser.add_argument(
        "--normalise",
        choices=["none", "unit", "zscore"],
        default="none",
        help=(
            "Normalise both matrices before analysis: unit divides by max off-diagonal distances; "
            "zscore standardises using off-diagonal mean/std; none leaves raw distances."
        ),
    )
    parser.add_argument(
        "--scale-min",
        type=float,
        default=0.0,
        help="Lower bound to clip scaled matrix1 distances (default: 0.0).",
    )
    parser.add_argument(
        "--stratified-out-dir",
        default=None,
        help="Directory to write per-matrix stratified CSV outputs (one file per matrix)",
    )
    args = parser.parse_args()

    if (args.stratify_phylum or args.stratify_phylum_pairs) and not args.taxonomy_csv:
        print("Stratified analyses require --taxonomy-csv", file=sys.stderr)
        sys.exit(1)

    if args.stratified_out_dir:
        try:
            os.makedirs(args.stratified_out_dir, exist_ok=True)
        except Exception as e:
            print(f"Cannot create stratified output directory {args.stratified_out_dir}: {e}", file=sys.stderr)
            sys.exit(1)

    if args.scale_matrix1 and args.scale_matrix1_with_intercept:
        print("Choose only one of --scale-matrix1 or --scale-matrix1-with-intercept.", file=sys.stderr)
        sys.exit(1)

    if args.normalise != "none" and (args.scale_matrix1 or args.scale_matrix1_with_intercept):
        print("Choose normalisation OR scaling, not both.", file=sys.stderr)
        sys.exit(1)

    patterns = ["full_clean1_mat_*.csv", "*_layer_*.csv"]
    mat_paths: list[str] = []
    for pat in patterns:
        mat_paths.extend(glob.glob(os.path.join(args.matrix_dir, pat)))
    mat_paths = sorted(set(mat_paths))
    if not mat_paths:
        print("No matrices found matching full_clean1_mat_*.csv or *_layer_*.csv", file=sys.stderr)
        sys.exit(1)

    # Set BLAS/NumPy threading for workers (per-process thread cap)
    thread_cap = str(args.omp_threads)
    os.environ["OMP_NUM_THREADS"] = thread_cap
    os.environ["OPENBLAS_NUM_THREADS"] = thread_cap
    os.environ["MKL_NUM_THREADS"] = thread_cap
    os.environ["NUMEXPR_NUM_THREADS"] = thread_cap

    rows: list[dict[str, object]] = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_gene = {}
        for mat_path in mat_paths:
            gene = os.path.basename(mat_path).removeprefix("full_clean1_mat_").removesuffix(".csv")
            tree1 = pick_tree(args.matrix_dir, gene)
            fut = executor.submit(
                process_one,
                mat_path,
                args.busco_matrix,
                tree1,
                args.busco_tree,
                args.permutations,
                args.method,
                args.alternative,
                args.reg_bootstrap,
                args.reg_ci,
                args.reg_bootstrap_mode,
                args.stratify_phylum,
                args.stratify_phylum_pairs,
                args.taxonomy_csv,
                args.taxonomy_accession_col,
                args.taxonomy_phylum_col,
                args.deviation_permutations,
                args.deviation_alpha,
                args.deviation_min_effect,
                args.deviation_fdr,
                args.stratified_out_dir,
                script_dir,
                args.normalise,
                args.scale_matrix1,
                args.scale_matrix1_with_intercept,
                args.scale_min,
            )
            future_to_gene[fut] = gene

        for fut in as_completed(future_to_gene):
            gene = future_to_gene[fut]
            try:
                row = fut.result()
                rows.append(row)
                print(f"Processed {gene}")
            except Exception as e:
                rows.append({"gene": gene, "error": f"worker_error: {e}"})
                print(f"Error processing {gene}: {e}", file=sys.stderr)

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"Wrote {args.output} with {len(df)} rows")


if __name__ == "__main__":
    main()
