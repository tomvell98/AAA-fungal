#!/usr/bin/env python3
"""
hotspot_vs_conservation_mwu.py

Combine hotspot residue indices from TWO TSV files (union), compare their conservation
against non-hotspot residues using:

1) Mann–Whitney U test (hotspot vs all non-hotspot)  [descriptive; can be overpowered]
2) MWU + permutation/random-set test:
     - sample many size-matched sets from non-hotspot residues
     - compute MWU statistic and AUC effect size for each
     - report empirical p-value under the random-set null

Outputs:
  - a one-row CSV summary with effect sizes and p-values
  - (optional) a CSV of permutation distribution if requested

Assumptions:
- Hotspot TSVs contain at least one column with residue indices (1-based or 0-based).
- Conservation CSV contains residue index + conservation score.
- Indices refer to the same coordinate system (alignment column indices).
  If not, use --hotspot-index-base / --cons-index-base to reconcile.

Example:
python hotspot_vs_conservation_mwu.py \
  --hotspot_tsv_a PC1.top_vs_bottom_10pct.hotspots.tsv \
  --hotspot_tsv_b PC2.top_vs_bottom_10pct.hotspots.tsv \
  --cons_csv LYS20.conservation_by_column.csv \
  --hotspot_index_col aln_col_1based \
  --cons_index_col aln_col_1based \
  --cons_value_col conservation \
  --top_k 25 \
  --n_perms 10000 \
  --seed 1 \
  --out_prefix LYS20_hotspot_vs_cons
"""

import argparse
import sys
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


def _read_table(path: str) -> pd.DataFrame:
    # Robust separator detection (tsv/csv)
    if path.lower().endswith(".tsv") or path.lower().endswith(".tab"):
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _coerce_index_base(series: pd.Series, base_in: int, base_out: int) -> pd.Series:
    """Convert indices between 0-based and 1-based."""
    s = pd.to_numeric(series, errors="coerce")
    if base_in == base_out:
        return s
    if base_in == 1 and base_out == 0:
        return s - 1
    if base_in == 0 and base_out == 1:
        return s + 1
    raise ValueError(f"Unsupported base conversion: {base_in} -> {base_out}")


def auc_from_u(u: float, n1: int, n2: int) -> float:
    """AUC = U / (n1*n2). Interpretable as P(X_hotspot > X_other) + 0.5*P(ties)."""
    if n1 <= 0 or n2 <= 0:
        return float("nan")
    return float(u) / float(n1 * n2)


def rank_biserial_from_auc(auc: float) -> float:
    """Rank-biserial correlation: 2*AUC - 1."""
    if np.isnan(auc):
        return float("nan")
    return float(2.0 * auc - 1.0)


def mwu_u(x: np.ndarray, y: np.ndarray, alternative: str) -> float:
    """Return U statistic (SciPy returns U for x by default)."""
    res = mannwhitneyu(x, y, alternative=alternative, method="auto")
    return float(res.statistic)


def main() -> None:

    ap = argparse.ArgumentParser(
        description="Compare hotspot residue conservation against non-hotspots using Mann–Whitney U test and permutation testing."
    )
    ap.add_argument("--hotspot_tsv_a", required=True)
    ap.add_argument("--hotspot_tsv_b", required=True)
    ap.add_argument("--cons_csv", required=True)
    ap.add_argument("--cons_index_base", type=int, choices=[0, 1], default=1,
                    help="Index base of conservation CSV indices (default 1).")

    # Set your actual defaults here
    ap.add_argument("--hotspot_index_col", default="cons_res_index_1based",
                    help="Column in hotspot TSVs containing residue index (default: cons_res_index_1based).")
    ap.add_argument("--cons_index_col", default="ResSeq",
                    help="Column in conservation CSV containing residue index (default: ResSeq).")
    ap.add_argument("--cons_value_col", default="AvgPhylaEffectiveConservation",
                    help="Column in conservation CSV containing conservation values (default: AvgPhylaEffectiveConservation).")

    ap.add_argument("--top_k", type=int, default=25,
                    help="Take top-K rows from each hotspot TSV (default 25). Use <=0 for ALL rows.")
    ap.add_argument("--hotspot_index_base", type=int, choices=[0, 1], default=1,
                    help="Index base of hotspot TSV indices (default 1).")
    ap.add_argument("--n_perms", type=int, default=10000,
                    help="Number of random-set permutations (default 10000).")
    ap.add_argument("--seed", type=int, default=1,
                    help="Random seed (default 1).")
    ap.add_argument("--out_prefix", required=True,
                    help="Prefix for output files.")
    ap.add_argument("--alternative", choices=["two-sided", "greater", "less"], default="two-sided",
                    help="Alternative for MWU. 'greater' tests hotspot > comparison.")
    ap.add_argument("--min_pool", type=int, default=10,
                    help="Minimum size of non-hotspot pool required to run permutations (default 10).")
    ap.add_argument("--write_perm_dist", action="store_true",
                    help="If set, write permutation distribution CSV.")

    args = ap.parse_args()

    # --- Load hotspots ---
    hs_a = _read_table(args.hotspot_tsv_a)
    hs_b = _read_table(args.hotspot_tsv_b)

    if args.hotspot_index_col not in hs_a.columns:
        raise ValueError(f"hotspot_tsv_a missing column '{args.hotspot_index_col}'. Columns: {list(hs_a.columns)}")
    if args.hotspot_index_col not in hs_b.columns:
        raise ValueError(f"hotspot_tsv_b missing column '{args.hotspot_index_col}'. Columns: {list(hs_b.columns)}")

    if args.top_k and args.top_k > 0:
        hs_a = hs_a.head(args.top_k).copy()
        hs_b = hs_b.head(args.top_k).copy()

    hs_idx_a = _coerce_index_base(hs_a[args.hotspot_index_col], args.hotspot_index_base, args.cons_index_base)
    hs_idx_b = _coerce_index_base(hs_b[args.hotspot_index_col], args.hotspot_index_base, args.cons_index_base)

    # Prepare all hotspot sets
    hotspot_sets = {
        "union": set(pd.concat([hs_idx_a, hs_idx_b], ignore_index=True).dropna().astype(int).tolist()),
        "hotspot_tsv_a": set(hs_idx_a.dropna().astype(int).tolist()),
        "hotspot_tsv_b": set(hs_idx_b.dropna().astype(int).tolist()),
    }

    # --- Load conservation ---
    cons = _read_table(args.cons_csv)
    if args.cons_index_col not in cons.columns:
        raise ValueError(f"cons_csv missing column '{args.cons_index_col}'. Columns: {list(cons.columns)}")
    if args.cons_value_col not in cons.columns:
        raise ValueError(f"cons_csv missing column '{args.cons_value_col}'. Columns: {list(cons.columns)}")

    # Check for FreeSASA_ASA column
    has_asa = "FreeSASA_ASA" in cons.columns

    # Prepare value columns to analyze
    value_cols = [(args.cons_value_col, "conservation")]
    if has_asa:
        value_cols.append(("FreeSASA_ASA", "asa"))

    # Prepare summary rows
    summary_rows = []

    for hs_label, hotspot_set in hotspot_sets.items():
        for value_col, value_label in value_cols:
            # Prepare conservation/ASA table
            cons_sub = cons[[args.cons_index_col, value_col]].copy()
            cons_sub[args.cons_index_col] = pd.to_numeric(cons_sub[args.cons_index_col], errors="coerce")
            cons_sub[value_col] = pd.to_numeric(cons_sub[value_col], errors="coerce")
            cons_sub = cons_sub.dropna(subset=[args.cons_index_col, value_col]).copy()
            cons_sub[args.cons_index_col] = cons_sub[args.cons_index_col].astype(int)

            # Keep only indices that exist in conservation table
            all_indices = set(cons_sub[args.cons_index_col].tolist())
            hs_set = hotspot_set.intersection(all_indices)

            if len(hs_set) == 0:
                # Skip if no overlap
                continue

            # Build vectors
            hotspot_mask = cons_sub[args.cons_index_col].isin(hs_set)
            x_hot = cons_sub.loc[hotspot_mask, value_col].to_numpy(dtype=float)
            pool = cons_sub.loc[~hotspot_mask, [args.cons_index_col, value_col]].copy()

            n_hot = int(len(x_hot))
            n_pool = int(len(pool))

            if n_pool < max(args.min_pool, n_hot):
                continue

            y_all = pool[value_col].to_numpy(dtype=float)

            # --- MWU hotspot vs ALL non-hotspot (descriptive) ---
            mwu_all = mannwhitneyu(x_hot, y_all, alternative=args.alternative, method="auto")
            u_all = float(mwu_all.statistic)
            p_all = float(mwu_all.pvalue)
            auc_all = auc_from_u(u_all, n_hot, len(y_all))
            rbc_all = rank_biserial_from_auc(auc_all)

            # --- Permutation/random-set test: compare hotspot vs many size-matched random sets ---
            rng = np.random.default_rng(args.seed)
            pool_vals = y_all

            u_perm = np.empty(args.n_perms, dtype=float)
            auc_perm = np.empty(args.n_perms, dtype=float)

            for i in range(args.n_perms):
                samp = rng.choice(pool_vals, size=n_hot, replace=False)
                u = mwu_u(x_hot, samp, alternative=args.alternative)
                u_perm[i] = u
                auc_perm[i] = auc_from_u(u, n_hot, n_hot)

            # --- Proper label-permutation test: shuffle hotspot labels across all positions ---
            all_vals = cons_sub[value_col].to_numpy(dtype=float)
            n_all = len(all_vals)
            x_obs = x_hot
            y_obs = y_all
            u_obs = u_all

            u_labelperm = np.empty(args.n_perms, dtype=float)
            auc_labelperm = np.empty(args.n_perms, dtype=float)

            hot_bool = hotspot_mask.to_numpy(dtype=bool)

            for i in range(args.n_perms):
                perm_mask = np.zeros(n_all, dtype=bool)
                perm_hot_idx = rng.choice(n_all, size=n_hot, replace=False)
                perm_mask[perm_hot_idx] = True
                x_p = all_vals[perm_mask]
                y_p = all_vals[~perm_mask]
                u_p = mwu_u(x_p, y_p, alternative=args.alternative)
                u_labelperm[i] = u_p
                auc_labelperm[i] = auc_from_u(u_p, n_hot, n_all - n_hot)

            # empirical p-value
            if args.alternative == "two-sided":
                center = float(np.median(u_labelperm))
                dev_obs = abs(u_obs - center)
                dev_perm = np.abs(u_labelperm - center)
                p_perm = float((np.sum(dev_perm >= dev_obs) + 1) / (len(dev_perm) + 1))
            else:
                if args.alternative == "greater":
                    p_perm = float((np.sum(u_labelperm >= u_obs) + 1) / (len(u_labelperm) + 1))
                else:
                    p_perm = float((np.sum(u_labelperm <= u_obs) + 1) / (len(u_labelperm) + 1))

            # Summary row
            summary = {
                "hotspot_set": hs_label,
                "value_col": value_col,
                "value_label": value_label,
                "hotspot_tsv_a": args.hotspot_tsv_a,
                "hotspot_tsv_b": args.hotspot_tsv_b,
                "cons_csv": args.cons_csv,
                "hotspot_index_col": args.hotspot_index_col,
                "cons_index_col": args.cons_index_col,
                "top_k_per_file": args.top_k,
                "n_hotspot": n_hot,
                "n_nonhotspot_pool": n_pool,
                "alternative": args.alternative,
                "mwu_u_hot_vs_allnon": u_all,
                "mwu_p_hot_vs_allnon": p_all,
                "auc_hot_vs_allnon": auc_all,
                "rank_biserial_hot_vs_allnon": rbc_all,
                "labelperm_p": p_perm,
                "labelperm_n": args.n_perms,
                "labelperm_auc_median": float(np.median(auc_labelperm)),
                "labelperm_auc_mean": float(np.mean(auc_labelperm)),
            }
            summary_rows.append(summary)

    # Write summary
    out_sum = f"{args.out_prefix}.hotspot_vs_conservation.summary.csv"
    pd.DataFrame(summary_rows).to_csv(out_sum, index=False)
    print(f"Wrote: {out_sum}")

    # Optionally write permutation distributions for the last run (union/conservation)
    if args.write_perm_dist and len(summary_rows) > 0:
        out_dist = f"{args.out_prefix}.hotspot_vs_conservation.perm_dist.csv"
        pd.DataFrame({
            "u_labelperm": u_labelperm,
            "auc_labelperm": auc_labelperm
        }).to_csv(out_dist, index=False)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
