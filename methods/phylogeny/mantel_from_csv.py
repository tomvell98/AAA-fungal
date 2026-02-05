#!/usr/bin/env python3
import argparse
import sys
import numpy as np
import pandas as pd
from skbio import TreeNode
from skbio.stats.distance import mantel


def _normalise_label(label: str) -> str:
    """Normalise genome IDs so the two matrices can be aligned.

    Examples
    --------
    - "GCA_000151175.1_ASM15117v1_genomic.fna" -> "GCA_000151175.1"
    - "GCA_000151175.1" -> "GCA_000151175.1"
    """
    if not isinstance(label, str):
        label = str(label)
    parts = label.split("_")
    if len(parts) >= 2 and parts[0] in {"GCA", "GCF"}:
        return f"{parts[0]}_{parts[1]}"
    return label


def load_distance_matrix(path: str) -> pd.DataFrame:
    """Load a distance matrix and normalise labels.

    Tries whitespace or comma separators. Expects:
    - First row: column labels (no leading row label).
    - Subsequent rows: first column is row label, remaining are distances.
    """

    def _try_read(sep):
        return pd.read_csv(path, sep=sep, header=0, index_col=0, engine="python")

    try:
        df = _try_read(r"\s+")
        if df.shape[1] == 0:
            raise ValueError("empty parse")
    except Exception:
        df = _try_read(",")

    if df.shape[0] != df.shape[1]:
        raise ValueError(
            f"{path} is not square: shape={df.shape}. "
            "Expected an n x n distance matrix with labels in first row/column."
        )

    # Normalise labels so that matrices using different naming schemes
    # (e.g. full filenames vs bare accessions) can still be aligned.
    norm_index = df.index.to_series().map(_normalise_label)
    norm_columns = df.columns.to_series().map(_normalise_label)

    df.index = norm_index
    df.columns = norm_columns

    return df


def _load_tree(path: str) -> TreeNode:
    """Load a Newick tree and normalise tip labels."""

    tree = TreeNode.read(path)
    for tip in tree.tips():
        if tip.name is None:
            raise ValueError(f"Tip without name encountered in {path}")
        tip.name = _normalise_label(tip.name)
    return tree


def _prune_to_common_tips(tree1: TreeNode, tree2: TreeNode):
    """Prune both trees to their shared tip set and return common labels."""

    tips1 = {tip.name for tip in tree1.tips()}
    tips2 = {tip.name for tip in tree2.tips()}
    common = tips1.intersection(tips2)

    if len(common) < 3:
        raise ValueError(
            f"Not enough overlapping tips between trees ({len(common)} found; need at least 3)."
        )

    pruned1 = tree1.shear(common)
    pruned2 = tree2.shear(common)
    return pruned1, pruned2, common


def _bipartitions(tree: TreeNode):
    """Return canonical bipartitions (splits) for an unrooted RF calculation."""

    leaves = {tip.name for tip in tree.tips()}
    splits = set()

    for node in tree.postorder(include_self=True):
        if node.is_tip():
            continue
        tip_names = {tip.name for tip in node.tips()}
        if 0 < len(tip_names) < len(leaves):
            part = frozenset(tip_names)
            # Use the smaller side of the split for canonical representation
            if len(part) > len(leaves) / 2:
                part = frozenset(leaves - tip_names)
            splits.add(part)
    return splits


def _robinson_foulds(tree1: TreeNode, tree2: TreeNode):
    """Compute unnormalised RF distance between two trees with identical tip sets."""

    splits1 = _bipartitions(tree1)
    splits2 = _bipartitions(tree2)
    return len(splits1.symmetric_difference(splits2))


def _upper_tri_vectors(mat1: np.ndarray, mat2: np.ndarray):
    """Return aligned upper-triangle vectors (k>0) from two square matrices."""

    rows, cols = np.triu_indices(mat1.shape[0], k=1)
    x = mat1[rows, cols]
    y = mat2[rows, cols]
    return x, y


def _linear_regression_stats(x: np.ndarray, y: np.ndarray):
    """Compute slope, intercept, Pearson r, and r^2 for y ~ slope*x + intercept."""

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    X = np.hstack([x, np.ones_like(x)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    slope = float(beta[0, 0])
    intercept = float(beta[1, 0])
    r = float(np.corrcoef(x[:, 0], y[:, 0])[0, 1])
    r2 = r * r
    return slope, intercept, r, r2


def _fit_scale(dm1: np.ndarray, dm2: np.ndarray, with_intercept: bool = False):
    """Fit scale (and optional intercept) to map dm1 -> dm2 on upper triangle.

    Returns slope, intercept.
    """

    x, y = _upper_tri_vectors(dm1, dm2)
    if with_intercept:
        slope, intercept, _, _ = _linear_regression_stats(x, y)
    else:
        denom = float(np.dot(x, x))
        slope = float(np.dot(x, y) / denom) if denom > 0 else 1.0
        intercept = 0.0
    return slope, intercept


def _normalise_distance_matrix(df: pd.DataFrame, mode: str):
    """Normalise a distance matrix (DataFrame) using off-diagonal values.

    Modes
    -----
    - "unit": divide by the maximum off-diagonal distance (maps to [0, 1]).
    - "zscore": subtract mean and divide by standard deviation of off-diagonal distances.
    - "none": return unchanged.
    """

    if mode == "none":
        return df, {}

    arr = df.values.astype(float)
    rows, cols = np.triu_indices(arr.shape[0], k=1)
    tri = arr[rows, cols]

    if tri.size == 0:
        raise ValueError("Matrix must be at least 2x2 to normalise")

    stats: dict[str, float] = {}

    if mode == "unit":
        max_val = float(np.max(tri))
        if max_val <= 0:
            raise ValueError("Cannot unit-scale; maximum off-diagonal distance is <= 0")
        arr_norm = arr / max_val
        stats["max"] = max_val
    elif mode == "zscore":
        mean = float(np.mean(tri))
        std = float(np.std(tri, ddof=0))
        if std <= 0:
            raise ValueError("Cannot z-score; off-diagonal standard deviation is zero")
        arr_norm = (arr - mean) / std
        stats["mean"] = mean
        stats["std"] = std
    else:
        raise ValueError(f"Unknown normalisation mode: {mode}")

    np.fill_diagonal(arr_norm, 0.0)
    df_norm = pd.DataFrame(arr_norm, index=df.index, columns=df.columns)
    return df_norm, stats


def _bootstrap_regression(dm1: np.ndarray, dm2: np.ndarray, n_reps: int, ci: float, rng: np.random.Generator, mode: str):
    """Bootstrap regression: permute labels (null) or resample taxa (taxon bootstrap)."""

    slopes: list[float] = []
    intercepts: list[float] = []
    rs: list[float] = []

    n = dm1.shape[0]

    for _ in range(n_reps):
        if mode == "permute":
            perm = rng.permutation(n)
            dm2_b = dm2[perm][:, perm]
            dm1_b = dm1
        else:  # taxon
            idx = rng.choice(n, size=n, replace=True)
            dm1_b = dm1[idx][:, idx]
            dm2_b = dm2[idx][:, idx]

        x, y = _upper_tri_vectors(dm1_b, dm2_b)
        slope, intercept, r, _ = _linear_regression_stats(x, y)
        slopes.append(slope)
        intercepts.append(intercept)
        rs.append(r)

    alpha = 1 - ci
    lower = alpha / 2
    upper = 1 - alpha / 2

    def pct(vals):
        return (np.percentile(vals, lower * 100), np.percentile(vals, upper * 100))

    return {
        "slope_ci": pct(slopes),
        "intercept_ci": pct(intercepts),
        "r_ci": pct(rs),
    }


def _taxon_bootstrap_regression(dm1: np.ndarray, dm2: np.ndarray, n_reps: int, ci: float, rng: np.random.Generator):
    """Taxon-resampling bootstrap for regression; returns CIs."""

    return _bootstrap_regression(dm1, dm2, n_reps=n_reps, ci=ci, rng=rng, mode="taxon")


def _taxon_bootstrap_slopes(dm1: np.ndarray, dm2: np.ndarray, n_reps: int, rng: np.random.Generator) -> np.ndarray:
    """Taxon bootstrap slopes only (faster when CIs already handled elsewhere)."""

    slopes: list[float] = []
    n = dm1.shape[0]
    for _ in range(n_reps):
        idx = rng.choice(n, size=n, replace=True)
        x, y = _upper_tri_vectors(dm1[idx][:, idx], dm2[idx][:, idx])
        slope, _, _, _ = _linear_regression_stats(x, y)
        slopes.append(slope)
    return np.asarray(slopes)


def _bootstrap_slope_delta(group_dm1: np.ndarray, group_dm2: np.ndarray, global_dm1: np.ndarray | None, global_dm2: np.ndarray | None, n_reps: int, ci: float, rng: np.random.Generator, global_slopes: np.ndarray | None = None):
    """Bootstrap slope difference between a group and the global set.

    If precomputed global_slopes are provided, they are reused to avoid recomputation.
    """

    group_slopes = _taxon_bootstrap_slopes(group_dm1, group_dm2, n_reps=n_reps, rng=rng)
    if global_slopes is None:
        if global_dm1 is None or global_dm2 is None:
            raise ValueError("global_dm1/global_dm2 required when global_slopes not supplied")
        global_slopes = _taxon_bootstrap_slopes(global_dm1, global_dm2, n_reps=n_reps, rng=rng)
    if len(global_slopes) != len(group_slopes):
        raise ValueError("global_slopes length must match n_reps")

    delta = group_slopes - global_slopes

    alpha = 1 - ci
    lower_q = alpha / 2
    upper_q = 1 - alpha / 2
    delta_ci = (
        np.percentile(delta, lower_q * 100),
        np.percentile(delta, upper_q * 100),
    )

    # Two-sided bootstrap p-value for delta != 0 with small-sample correction
    ge = (np.sum(delta >= 0) + 1) / (len(delta) + 1)
    le = (np.sum(delta <= 0) + 1) / (len(delta) + 1)
    p_val = 2 * min(ge, le)
    p_val = min(p_val, 1.0)

    return {
        "delta_ci": delta_ci,
        "p_value": p_val,
    }


def _bh_fdr(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values (monotone, capped at 1)."""

    m = len(pvals)
    if m == 0:
        return []
    order = np.argsort(pvals)
    adj = np.empty(m)
    cummin = 1.0
    # Walk from largest to smallest to enforce monotonicity
    for rank, idx in enumerate(order[::-1], start=1):
        p = pvals[idx]
        adj_val = p * m / (m - rank + 1)
        cummin = min(cummin, adj_val)
        adj[idx] = min(cummin, 1.0)
    return adj.tolist()


def _perm_delta_pvalue(phylum_labels: np.ndarray, target_phyla: set[str], global_slope: float, group_slope_obs: float, dm1: np.ndarray, dm2: np.ndarray, permutations: int, rng: np.random.Generator, min_n: int = 3):
    """Permutation test for Δslope: shuffle phylum labels, recompute group slope, compare to observed.

    Returns p-value and number of successful permutations (with enough taxa).
    """

    if permutations <= 0:
        return None, 0

    exceed = 0
    valid = 0

    delta_obs = group_slope_obs - global_slope

    for _ in range(permutations):
        perm_labels = rng.permutation(phylum_labels)
        mask = np.isin(perm_labels, list(target_phyla))
        if mask.sum() < min_n:
            continue
        x_p, y_p = _upper_tri_vectors(dm1[np.ix_(mask, mask)], dm2[np.ix_(mask, mask)])
        slope_perm, _, _, _ = _linear_regression_stats(x_p, y_p)
        delta_perm = slope_perm - global_slope
        valid += 1
        if abs(delta_perm) >= abs(delta_obs):
            exceed += 1

    if valid == 0:
        return None, 0

    p_val = (exceed + 1) / (valid + 1)
    return p_val, valid


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a Mantel test on two CSV distance matrices."
    )
    parser.add_argument("matrix1", help="Path to first distance matrix CSV")
    parser.add_argument("matrix2", help="Path to second distance matrix CSV")
    parser.add_argument(
        "-m", "--method",
        choices=["pearson", "spearman"],
        default="pearson",
        help="Correlation method (default: pearson)",
    )
    parser.add_argument(
        "-p", "--permutations",
        type=int,
        default=999,
        help="Number of permutations for significance testing (default: 999)",
    )
    parser.add_argument(
        "-a", "--alternative",
        choices=["two-sided", "greater", "less"],
        default="two-sided",
        help="Alternative hypothesis (default: two-sided)",
    )
    parser.add_argument(
        "--tree1",
        help="Optional path to first Newick tree for topology comparison",
    )
    parser.add_argument(
        "--tree2",
        help="Optional path to second Newick tree for topology comparison",
    )
    parser.add_argument(
        "--linear-regression",
        action="store_true",
        help="Also report slope/intercept/R^2 between flattened matrices",
    )
    parser.add_argument(
        "--reg-bootstrap",
        type=int,
        default=0,
        help="Number of label-permutation bootstraps for regression CIs (default: 0)",
    )
    parser.add_argument(
        "--reg-ci",
        type=float,
        default=0.95,
        help="Confidence level for regression CIs (default: 0.95)",
    )
    parser.add_argument(
        "--reg-bootstrap-mode",
        choices=["permute", "taxon"],
        default="permute",
        help="Bootstrap mode: permute (null) or taxon (resample taxa)",
    )
    parser.add_argument(
        "--deviation-alpha",
        type=float,
        default=0.05,
        help="Significance level for deviation tests on slopes (default: 0.05)",
    )
    parser.add_argument(
        "--deviation-min-effect",
        type=float,
        default=0.0,
        help="Minimum absolute slope difference vs global to flag deviation (default: 0.0)",
    )
    parser.add_argument(
        "--deviation-fdr",
        action="store_true",
        help="Apply Benjamini-Hochberg FDR correction to stratified deviation p-values",
    )
    parser.add_argument(
        "--deviation-permutations",
        type=int,
        default=0,
        help="Number of label permutations for Δslope significance testing (default: 0 = skip)",
    )
    parser.add_argument(
        "--scale-matrix1",
        action="store_true",
        help="Linearly scale matrix1 to matrix2 via slope (through origin) before analyses.",
    )
    parser.add_argument(
        "--scale-matrix1-with-intercept",
        action="store_true",
        help="Scale matrix1 to matrix2 using slope+intercept fit; keeps matrix1 diagonal reset to 0.",
    )
    parser.add_argument(
        "--normalise",
        choices=["none", "unit", "zscore"],
        default="none",
        help=(
            "Normalise both matrices before analysis: "
            "unit divides by max off-diagonal (maps to [0,1]); "
            "zscore standardises using off-diagonal mean/std; none leaves unchanged."
        ),
    )
    parser.add_argument(
        "--scale-min",
        type=float,
        default=0.0,
        help="Lower bound to clip scaled matrix1 distances (default: 0.0).",
    )
    parser.add_argument(
        "--stratified-csv-out",
        default=None,
        help="Optional path to write stratified results (phylum/phylum-pair) as CSV",
    )
    parser.add_argument(
        "--stratify-phylum",
        action="store_true",
        help="Compute per-phylum regressions and deviation vs global slope",
    )
    parser.add_argument(
        "--stratify-phylum-pairs",
        action="store_true",
        help="Compute per-phylum-pair regressions and deviation vs global slope",
    )
    parser.add_argument(
        "--taxonomy-csv",
        default=None,
        help="Taxonomy table with accession and Phylum columns",
    )
    parser.add_argument(
        "--taxonomy-accession-col",
        default="Accession",
        help="Column name for accession in taxonomy table (default: Accession)",
    )
    parser.add_argument(
        "--taxonomy-phylum-col",
        default="Phylum",
        help="Column name for phylum in taxonomy table (default: Phylum)",
    )
    args = parser.parse_args()

    if bool(args.tree1) ^ bool(args.tree2):
        print("Provide both --tree1 and --tree2 to compute a tree distance.", file=sys.stderr)
        sys.exit(1)

    taxonomy_df = None
    if args.stratify_phylum or args.stratify_phylum_pairs:
        if not args.taxonomy_csv:
            print("Stratified phylum analysis requires --taxonomy-csv", file=sys.stderr)
            sys.exit(1)
        try:
            taxonomy_df = pd.read_csv(args.taxonomy_csv)
        except Exception as e:
            print(f"Error loading taxonomy table: {e}", file=sys.stderr)
            sys.exit(1)

    try:
        dm1 = load_distance_matrix(args.matrix1)
        dm2 = load_distance_matrix(args.matrix2)
    except Exception as e:
        print(f"Error loading matrices: {e}", file=sys.stderr)
        sys.exit(1)

    if args.scale_matrix1 and args.scale_matrix1_with_intercept:
        print("Choose only one of --scale-matrix1 or --scale-matrix1-with-intercept.", file=sys.stderr)
        sys.exit(1)

    if args.normalise != "none" and (args.scale_matrix1 or args.scale_matrix1_with_intercept):
        print("Choose normalisation OR scaling, not both.", file=sys.stderr)
        sys.exit(1)

    # Align matrices on shared labels
    common_labels = dm1.index.intersection(dm2.index)
    if len(common_labels) < 3:
        print(
            f"Not enough overlapping labels between matrices "
            f"({len(common_labels)} found; need at least 3).",
            file=sys.stderr,
        )
        sys.exit(1)

    dm1_aligned = dm1.loc[common_labels, common_labels]
    dm2_aligned = dm2.loc[common_labels, common_labels]

    if args.normalise != "none":
        try:
            dm1_aligned, stats1 = _normalise_distance_matrix(dm1_aligned, mode=args.normalise)
            dm2_aligned, stats2 = _normalise_distance_matrix(dm2_aligned, mode=args.normalise)
        except ValueError as e:
            print(f"Normalisation error: {e}", file=sys.stderr)
            sys.exit(1)

        if args.normalise == "unit":
            print(
                "Normalised matrices by max off-diagonal: "
                f"matrix1 max={stats1['max']:.4g}, matrix2 max={stats2['max']:.4g}"
            )
        elif args.normalise == "zscore":
            print(
                "Normalised matrices by z-score (off-diagonal): "
                f"matrix1 mean={stats1['mean']:.4g}, sd={stats1['std']:.4g}; "
                f"matrix2 mean={stats2['mean']:.4g}, sd={stats2['std']:.4g}"
            )

    if args.scale_matrix1 or args.scale_matrix1_with_intercept:
        slope_s, intercept_s = _fit_scale(
            dm1_aligned.values, dm2_aligned.values, with_intercept=args.scale_matrix1_with_intercept
        )
        scaled_vals = dm1_aligned.values * slope_s + intercept_s
        if args.scale_min is not None:
            scaled_vals = np.maximum(scaled_vals, args.scale_min)
        np.fill_diagonal(scaled_vals, 0.0)
        dm1_aligned = pd.DataFrame(scaled_vals, index=dm1_aligned.index, columns=dm1_aligned.columns)
        print(f"Scaled matrix1 to matrix2: slope={slope_s:.4f}, intercept={intercept_s:.4f}, clip_min={args.scale_min}")

    print(f"Using {len(common_labels)} shared labels.")
    r, p_value, n = mantel(
        dm1_aligned.values,
        dm2_aligned.values,
        method=args.method,
        permutations=args.permutations,
        alternative=args.alternative,
    )

    print(f"Mantel r: {r:.4f}")
    print(f"p-value: {p_value:.4g}")
    print(f"Number of effective comparisons (n): {n}")

    global_slope_boot: np.ndarray | None = None

    # Always compute the global linear fit; only print if requested
    x, y = _upper_tri_vectors(dm1_aligned.values, dm2_aligned.values)
    slope, intercept, r_lin, r2 = _linear_regression_stats(x, y)

    if args.linear_regression:
        print("Linear regression on paired distances (y ~ slope*x + intercept):")
        print(f"  slope: {slope:.4f}")
        print(f"  intercept: {intercept:.4f}")
        print(f"  Pearson r: {r_lin:.4f}")
        print(f"  R^2: {r2:.4f}")

    if args.reg_bootstrap > 0:
        rng = np.random.default_rng(seed=42)
        ci = _bootstrap_regression(
            dm1_aligned.values,
            dm2_aligned.values,
            n_reps=args.reg_bootstrap,
            ci=args.reg_ci,
            rng=rng,
            mode=args.reg_bootstrap_mode,
        )
        if args.linear_regression:
            lower, upper = ci["slope_ci"]
            print(f"  slope CI ({args.reg_ci:.2f}): [{lower:.4f}, {upper:.4f}]")
            lower, upper = ci["intercept_ci"]
            print(f"  intercept CI ({args.reg_ci:.2f}): [{lower:.4f}, {upper:.4f}]")
            lower, upper = ci["r_ci"]
            print(f"  Pearson r CI ({args.reg_ci:.2f}): [{lower:.4f}, {upper:.4f}]")

    # Precompute global slope bootstraps once for deviation tests (taxon only)
    if args.reg_bootstrap > 0 and args.reg_bootstrap_mode == "taxon":
        rng_global = np.random.default_rng(seed=123)
        global_slope_boot = _taxon_bootstrap_slopes(
            dm1_aligned.values,
            dm2_aligned.values,
            n_reps=args.reg_bootstrap,
            rng=rng_global,
        )

    if (args.stratify_phylum or args.stratify_phylum_pairs) and taxonomy_df is not None:
        if args.taxonomy_accession_col not in taxonomy_df.columns or args.taxonomy_phylum_col not in taxonomy_df.columns:
            print(
                f"Taxonomy table missing required columns {args.taxonomy_accession_col} / {args.taxonomy_phylum_col}",
                file=sys.stderr,
            )
            sys.exit(1)

        tax_map = (
            taxonomy_df[[args.taxonomy_accession_col, args.taxonomy_phylum_col]]
            .assign(**{args.taxonomy_accession_col: lambda d: d[args.taxonomy_accession_col].astype(str).map(_normalise_label)})
            .groupby(args.taxonomy_accession_col)[args.taxonomy_phylum_col]
            .first()
        )

        phylum_to_labels: dict[str, list[str]] = {}
        for label in common_labels:
            phylum = tax_map.get(label)
            if pd.isna(phylum) or phylum is None:
                continue
            phylum_to_labels.setdefault(str(phylum), []).append(label)

        phylum_labels_arr = np.array([tax_map.get(lbl, np.nan) for lbl in common_labels], dtype=object)

        can_do_delta = args.reg_bootstrap > 0 and args.reg_bootstrap_mode == "taxon" and global_slope_boot is not None
        can_do_perm = args.deviation_permutations > 0

        stratified_records: list[dict] = []

        if args.stratify_phylum:
            print("\nPer-phylum regression vs BUSCO (taxon bootstrap for CIs and slope deltas):")

            phylum_rows = []
            for idx, (phylum, labels) in enumerate(sorted(phylum_to_labels.items())):
                if len(labels) < 3:
                    continue
                sub1 = dm1_aligned.loc[labels, labels].values
                sub2 = dm2_aligned.loc[labels, labels].values
                x_sub, y_sub = _upper_tri_vectors(sub1, sub2)
                slope_p, intercept_p, r_p, r2_p = _linear_regression_stats(x_sub, y_sub)

                ci = None
                if args.reg_bootstrap > 0:
                    rng_ci = np.random.default_rng(seed=1000 + idx)
                    ci = _taxon_bootstrap_regression(
                        sub1,
                        sub2,
                        n_reps=args.reg_bootstrap,
                        ci=args.reg_ci,
                        rng=rng_ci,
                    )

                deviation = None
                delta_ci = None
                if can_do_delta:
                    rng_delta = np.random.default_rng(seed=2000 + idx)
                    delta_stats = _bootstrap_slope_delta(
                        sub1,
                        sub2,
                        global_dm1=None,
                        global_dm2=None,
                        n_reps=args.reg_bootstrap,
                        ci=args.reg_ci,
                        rng=rng_delta,
                        global_slopes=global_slope_boot,
                    )
                    delta_ci = delta_stats["delta_ci"]

                perm_p = None
                perm_valid = 0
                if can_do_perm:
                    rng_perm = np.random.default_rng(seed=6000 + idx)
                    perm_p, perm_valid = _perm_delta_pvalue(
                        phylum_labels_arr,
                        {phylum},
                        slope,
                        slope_p,
                        dm1_aligned.values,
                        dm2_aligned.values,
                        permutations=args.deviation_permutations,
                        rng=rng_perm,
                    )

                effect = slope_p - slope
                deviation = {
                    "effect": effect,
                    "delta_ci": delta_ci,
                    "p_perm": perm_p,
                    "perm_valid": perm_valid,
                }

                phylum_rows.append(
                    {
                        "phylum": phylum,
                        "n": len(labels),
                        "slope": slope_p,
                        "r": r_p,
                        "r2": r2_p,
                        "ci": ci,
                        "deviation": deviation,
                    }
                )

            if args.deviation_fdr and can_do_perm:
                pvals = [row["deviation"]["p_perm"] for row in phylum_rows if row.get("deviation") is not None and row["deviation"].get("p_perm") is not None]
                adj = _bh_fdr(pvals)
                adj_iter = iter(adj)
                for row in phylum_rows:
                    if row.get("deviation") is None or row["deviation"].get("p_perm") is None:
                        continue
                    row["deviation"]["p_adj"] = next(adj_iter)

            for row in phylum_rows:
                msg = (
                    f"  {row['phylum']}: n={row['n']}, slope={row['slope']:.4f}, r={row['r']:.4f}, r2={row['r2']:.4f}"
                )
                if row["ci"]:
                    msg += f", slope CI ({args.reg_ci:.2f})=[{row['ci']['slope_ci'][0]:.4f}, {row['ci']['slope_ci'][1]:.4f}]"
                dev = row.get("deviation")
                if dev:
                    p_use = dev.get("p_adj", dev.get("p_perm"))
                    if dev.get("delta_ci") is not None:
                        msg += f", Δslope={dev['effect']:.4f}, ΔCI=[{dev['delta_ci'][0]:.4f}, {dev['delta_ci'][1]:.4f}]"
                    else:
                        msg += f", Δslope={dev['effect']:.4f}"
                    if dev.get("p_perm") is not None:
                        msg += f", perm p={dev['p_perm']:.4g}"
                    if "p_adj" in dev:
                        msg += f", q={dev['p_adj']:.4g}"
                    deviates_flag = (
                        p_use is not None
                        and abs(dev["effect"]) >= args.deviation_min_effect
                        and p_use <= args.deviation_alpha
                    )
                    msg += ", deviates? yes" if deviates_flag else ", deviates? no"
                    stratified_records.append(
                        {
                            "level": "phylum",
                            "group": row["phylum"],
                            "n": row["n"],
                            "slope": row["slope"],
                            "slope_ci_lower": row["ci"]["slope_ci"][0] if row["ci"] else None,
                            "slope_ci_upper": row["ci"]["slope_ci"][1] if row["ci"] else None,
                            "r": row["r"],
                            "r2": row["r2"],
                            "delta_slope": dev["effect"],
                            "delta_ci_lower": dev["delta_ci"][0] if dev.get("delta_ci") is not None else None,
                            "delta_ci_upper": dev["delta_ci"][1] if dev.get("delta_ci") is not None else None,
                            "perm_p": dev.get("p_perm"),
                            "q": dev.get("p_adj"),
                            "deviates": deviates_flag,
                            "permutations": args.deviation_permutations,
                            "perm_valid": dev.get("perm_valid"),
                            "alpha": args.deviation_alpha,
                            "min_effect": args.deviation_min_effect,
                        }
                    )
                print(msg)

        if args.stratify_phylum_pairs:
            phyla = sorted(phylum_to_labels.keys())
            print("\nPer-phylum-pair regression vs BUSCO (taxon bootstrap for CIs and slope deltas):")

            pair_rows = []
            for i, p1 in enumerate(phyla):
                for j, p2 in enumerate(phyla[i:]):
                    labels = [lbl for lbl in common_labels if tax_map.get(lbl) in {p1, p2}]
                    if len(labels) < 3:
                        continue
                    sub1 = dm1_aligned.loc[labels, labels].values
                    sub2 = dm2_aligned.loc[labels, labels].values
                    x_sub, y_sub = _upper_tri_vectors(sub1, sub2)
                    slope_p, intercept_p, r_p, r2_p = _linear_regression_stats(x_sub, y_sub)

                    ci = None
                    if args.reg_bootstrap > 0:
                        rng_ci = np.random.default_rng(seed=3000 + i * 100 + j)
                        ci = _taxon_bootstrap_regression(
                            sub1,
                            sub2,
                            n_reps=args.reg_bootstrap,
                            ci=args.reg_ci,
                            rng=rng_ci,
                        )

                    delta_ci = None
                    if can_do_delta:
                        rng_delta = np.random.default_rng(seed=4000 + i * 100 + j)
                        delta_stats = _bootstrap_slope_delta(
                            sub1,
                            sub2,
                            global_dm1=None,
                            global_dm2=None,
                            n_reps=args.reg_bootstrap,
                            ci=args.reg_ci,
                            rng=rng_delta,
                            global_slopes=global_slope_boot,
                        )
                        delta_ci = delta_stats["delta_ci"]

                    perm_p = None
                    perm_valid = 0
                    if can_do_perm:
                        rng_perm = np.random.default_rng(seed=8000 + i * 100 + j)
                        perm_p, perm_valid = _perm_delta_pvalue(
                            phylum_labels_arr,
                            {p1, p2},
                            slope,
                            slope_p,
                            dm1_aligned.values,
                            dm2_aligned.values,
                            permutations=args.deviation_permutations,
                            rng=rng_perm,
                        )

                    effect = slope_p - slope
                    deviation = {
                        "effect": effect,
                        "delta_ci": delta_ci,
                        "p_perm": perm_p,
                        "perm_valid": perm_valid,
                    }

                    pair_rows.append(
                        {
                            "pair": f"{p1}|{p2}" if p1 != p2 else p1,
                            "n": len(labels),
                            "slope": slope_p,
                            "r": r_p,
                            "r2": r2_p,
                            "ci": ci,
                            "deviation": deviation,
                        }
                    )

            if args.deviation_fdr and can_do_perm:
                pvals = [row["deviation"]["p_perm"] for row in pair_rows if row.get("deviation") is not None and row["deviation"].get("p_perm") is not None]
                adj = _bh_fdr(pvals)
                adj_iter = iter(adj)
                for row in pair_rows:
                    if row.get("deviation") is None or row["deviation"].get("p_perm") is None:
                        continue
                    row["deviation"]["p_adj"] = next(adj_iter)

            for row in pair_rows:
                msg = (
                    f"  {row['pair']}: n={row['n']}, slope={row['slope']:.4f}, r={row['r']:.4f}, r2={row['r2']:.4f}"
                )
                if row["ci"]:
                    msg += f", slope CI ({args.reg_ci:.2f})=[{row['ci']['slope_ci'][0]:.4f}, {row['ci']['slope_ci'][1]:.4f}]"
                dev = row.get("deviation")
                if dev:
                    p_use = dev.get("p_adj", dev.get("p_perm"))
                    if dev.get("delta_ci") is not None:
                        msg += f", Δslope={dev['effect']:.4f}, ΔCI=[{dev['delta_ci'][0]:.4f}, {dev['delta_ci'][1]:.4f}]"
                    else:
                        msg += f", Δslope={dev['effect']:.4f}"
                    if dev.get("p_perm") is not None:
                        msg += f", perm p={dev['p_perm']:.4g}"
                    if "p_adj" in dev:
                        msg += f", q={dev['p_adj']:.4g}"
                    deviates_flag = (
                        p_use is not None
                        and abs(dev["effect"]) >= args.deviation_min_effect
                        and p_use <= args.deviation_alpha
                    )
                    msg += ", deviates? yes" if deviates_flag else ", deviates? no"
                    stratified_records.append(
                        {
                            "level": "phylum_pair",
                            "group": row["pair"],
                            "n": row["n"],
                            "slope": row["slope"],
                            "slope_ci_lower": row["ci"]["slope_ci"][0] if row["ci"] else None,
                            "slope_ci_upper": row["ci"]["slope_ci"][1] if row["ci"] else None,
                            "r": row["r"],
                            "r2": row["r2"],
                            "delta_slope": dev["effect"],
                            "delta_ci_lower": dev["delta_ci"][0] if dev.get("delta_ci") is not None else None,
                            "delta_ci_upper": dev["delta_ci"][1] if dev.get("delta_ci") is not None else None,
                            "perm_p": dev.get("p_perm"),
                            "q": dev.get("p_adj"),
                            "deviates": deviates_flag,
                            "permutations": args.deviation_permutations,
                            "perm_valid": dev.get("perm_valid"),
                            "alpha": args.deviation_alpha,
                            "min_effect": args.deviation_min_effect,
                        }
                    )
                print(msg)

        if args.stratified_csv_out and stratified_records:
            pd.DataFrame(stratified_records).to_csv(args.stratified_csv_out, index=False)
            print(f"Wrote stratified results to {args.stratified_csv_out}")

    if args.tree1 and args.tree2:
        try:
            tree1 = _load_tree(args.tree1)
            tree2 = _load_tree(args.tree2)
            pruned1, pruned2, common_tips = _prune_to_common_tips(tree1, tree2)

            rf_distance = _robinson_foulds(pruned1, pruned2)
            denom = 2 * (len(common_tips) - 3)
            normalised_rf = rf_distance / denom if denom > 0 else None

            print(f"Using {len(common_tips)} shared tips for tree comparison.")
            print(f"Robinson-Foulds distance: {rf_distance}")
            if normalised_rf is not None:
                print(f"Normalised RF (0-1): {normalised_rf:.4f}")
            else:
                print("Normalised RF (0-1): n/a (need at least 4 shared tips)")
        except Exception as e:
            print(f"Error computing tree distance: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()