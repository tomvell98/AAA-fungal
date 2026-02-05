#!/usr/bin/env python3
"""
nonparametric_ancova_quade_style.py

Nonparametric ANCOVA-like test:
Does SS affect CONS after adjusting for SASA?

Method (Quade-style residualization):
  - Rank-transform CONS and SASA within each enzyme
  - Regress rank(CONS) ~ rank(SASA)
  - Test residuals across SS groups using Kruskal–Wallis
  - BH-FDR across enzymes

This avoids distributional assumptions and does NOT use permutations.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import rankdata, kruskal
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

def quade_style_ss_adjusted_test(
    df: pd.DataFrame,
    enzyme_col: str = "ENZYME",
    cons_col: str = "CONS",
    sasa_col: str = "SASA",
    ss_col: str = "SS",
) -> pd.DataFrame:
    rows = []

    for enz, sub in df.groupby(enzyme_col, sort=False):
        sub = sub[[cons_col, sasa_col, ss_col]].dropna().copy()

        # Need at least 2 SS groups
        if sub[ss_col].nunique() < 2 or len(sub) < 20:
            rows.append({"ENZYME": enz, "n": len(sub), "H": np.nan, "p": np.nan})
            continue

        # Rank-transform within enzyme
        sub["rCONS"] = rankdata(sub[cons_col].to_numpy(), method="average")
        sub["rSASA"] = rankdata(sub[sasa_col].to_numpy(), method="average")

        # Residualize rCONS against rSASA (rank-based adjustment for burial)
        X = sm.add_constant(sub["rSASA"].to_numpy())
        y = sub["rCONS"].to_numpy()
        model = sm.OLS(y, X).fit()
        sub["resid"] = model.resid

        # Kruskal–Wallis on residuals across SS groups
        ss_levels = sub[ss_col].astype("category").cat.categories
        groups = [sub.loc[sub[ss_col] == lvl, "resid"].to_numpy() for lvl in ss_levels]
        groups = [g for g in groups if len(g) > 0]

        if len(groups) < 2:
            rows.append({"ENZYME": enz, "n": len(sub), "H": np.nan, "p": np.nan})
            continue

        H, p = kruskal(*groups)
        rows.append({"ENZYME": enz, "n": len(sub), "H": H, "p": p})

    res = pd.DataFrame(rows)

    # BH-FDR across enzymes
    mask = res["p"].notna()
    res["q_BH"] = np.nan
    if mask.sum() > 0:
        res.loc[mask, "q_BH"] = multipletests(res.loc[mask, "p"].to_numpy(), method="fdr_bh")[1]

    return res


if __name__ == "__main__":
    import argparse
    import os
    from itertools import combinations
    from scipy.stats import mannwhitneyu

    parser = argparse.ArgumentParser(description="Quade-style nonparametric ANCOVA for secondary structure conservation adjusted for burial.")
    parser.add_argument("--inputs", nargs='+', required=True, help="Input CSV file(s), one per enzyme")
    parser.add_argument("--output", type=str, default="SS_effect_adjusted_for_SASA_quade_style.csv", help="Output CSV file")
    args = parser.parse_args()

    # Hard-coded columns
    CONS_COL = "AvgPhylaEffectiveConservation"
    SASA_COL = "FreeSASA_ASA"
    SS_COL = "SecStruct"

    def collapse_ss(ss):
        if pd.isna(ss) or str(ss).strip() == '':
            return "Loop"
        ss = str(ss).strip().upper()
        if ss in ["H", "G", "I"]:
            return "Helix"
        elif ss in ["B", "E"]:
            return "Sheet"
        else:
            return "Loop"

    all_omnibus = []
    all_posthoc = []
    all_omnibus_unadj = []
    all_posthoc_unadj = []
    all_counts = []

    for input_file in args.inputs:
        # Use filename (without extension) as enzyme name
        enzyme_name = os.path.splitext(os.path.basename(input_file))[0]
        df = pd.read_csv(input_file)
        df["SS"] = df[SS_COL].apply(collapse_ss)
        df["ENZYME"] = enzyme_name

        # Count of each SS category per enzyme
        counts = df["SS"].value_counts(dropna=False).reset_index()
        counts.columns = ["SS", "count"]
        counts["ENZYME"] = enzyme_name
        all_counts.append(counts)

        # --- Adjusted for SASA ---
        # Omnibus test
        res = quade_style_ss_adjusted_test(
            df,
            enzyme_col="ENZYME",
            cons_col=CONS_COL,
            sasa_col=SASA_COL,
            ss_col="SS",
        )
        res["ENZYME"] = enzyme_name  # ensure correct enzyme name
        all_omnibus.append(res)

        # Post-hoc pairwise tests on residuals
        sub = df[[CONS_COL, SASA_COL, "SS"]].dropna().copy()
        sub["rCONS"] = rankdata(sub[CONS_COL].to_numpy(), method="average")
        sub["rSASA"] = rankdata(sub[SASA_COL].to_numpy(), method="average")
        X = sm.add_constant(sub["rSASA"].to_numpy())
        y = sub["rCONS"].to_numpy()
        model = sm.OLS(y, X).fit()
        sub["resid"] = model.resid

        ss_levels = sub["SS"].astype("category").cat.categories
        pairs = list(combinations(ss_levels, 2))
        posthoc_rows = []
        for a, b in pairs:
            group_a = sub.loc[sub["SS"] == a, "resid"].to_numpy()
            group_b = sub.loc[sub["SS"] == b, "resid"].to_numpy()
            if len(group_a) > 0 and len(group_b) > 0:
                stat, p = mannwhitneyu(group_a, group_b, alternative="two-sided")
                median_a = np.median(group_a)
                median_b = np.median(group_b)
                if median_a > median_b:
                    direction = f"{a} > {b}"
                elif median_a < median_b:
                    direction = f"{a} < {b}"
                else:
                    direction = "no difference"
                posthoc_rows.append({
                    "ENZYME": enzyme_name,
                    "group1": a,
                    "group2": b,
                    "U": stat,
                    "p": p,
                    "direction": direction,
                    "median_group1": median_a,
                    "median_group2": median_b
                })
            else:
                posthoc_rows.append({
                    "ENZYME": enzyme_name,
                    "group1": a,
                    "group2": b,
                    "U": np.nan,
                    "p": np.nan,
                    "direction": "NA"
                })

        posthoc_df = pd.DataFrame(posthoc_rows)
        # FDR correction for posthoc p-values (within each enzyme)
        mask = posthoc_df["p"].notna()
        posthoc_df["q_BH"] = np.nan
        if mask.sum() > 0:
            posthoc_df.loc[mask, "q_BH"] = multipletests(posthoc_df.loc[mask, "p"].to_numpy(), method="fdr_bh")[1]
        all_posthoc.append(posthoc_df)

        # --- Unadjusted (raw conservation) ---
        # Omnibus Kruskal-Wallis on raw conservation
        sub_unadj = df[[CONS_COL, "SS"]].dropna().copy()
        ss_levels_unadj = sub_unadj["SS"].astype("category").cat.categories
        groups_unadj = [sub_unadj.loc[sub_unadj["SS"] == lvl, CONS_COL].to_numpy() for lvl in ss_levels_unadj]
        groups_unadj = [g for g in groups_unadj if len(g) > 0]
        if len(groups_unadj) >= 2:
            H_unadj, p_unadj = kruskal(*groups_unadj)
        else:
            H_unadj, p_unadj = np.nan, np.nan
        res_unadj = pd.DataFrame({
            "ENZYME": [enzyme_name],
            "n": [len(sub_unadj)],
            "H": [H_unadj],
            "p": [p_unadj],
            "q_BH": [p_unadj]  # only one test per enzyme
        })
        all_omnibus_unadj.append(res_unadj)

        # Post-hoc pairwise tests on raw conservation
        pairs_unadj = list(combinations(ss_levels_unadj, 2))
        posthoc_rows_unadj = []
        for a, b in pairs_unadj:
            group_a = sub_unadj.loc[sub_unadj["SS"] == a, CONS_COL].to_numpy()
            group_b = sub_unadj.loc[sub_unadj["SS"] == b, CONS_COL].to_numpy()
            if len(group_a) > 0 and len(group_b) > 0:
                stat, p = mannwhitneyu(group_a, group_b, alternative="two-sided")
                median_a = np.median(group_a)
                median_b = np.median(group_b)
                if median_a > median_b:
                    direction = f"{a} > {b}"
                elif median_a < median_b:
                    direction = f"{a} < {b}"
                else:
                    direction = "no difference"
                posthoc_rows_unadj.append({
                    "ENZYME": enzyme_name,
                    "group1": a,
                    "group2": b,
                    "U": stat,
                    "p": p,
                    "direction": direction,
                    "median_group1": median_a,
                    "median_group2": median_b
                })
            else:
                posthoc_rows_unadj.append({
                    "ENZYME": enzyme_name,
                    "group1": a,
                    "group2": b,
                    "U": np.nan,
                    "p": np.nan,
                    "direction": "NA"
                })
        posthoc_df_unadj = pd.DataFrame(posthoc_rows_unadj)
        mask_unadj = posthoc_df_unadj["p"].notna()
        posthoc_df_unadj["q_BH"] = np.nan
        if mask_unadj.sum() > 0:
            posthoc_df_unadj.loc[mask_unadj, "q_BH"] = multipletests(posthoc_df_unadj.loc[mask_unadj, "p"].to_numpy(), method="fdr_bh")[1]
        all_posthoc_unadj.append(posthoc_df_unadj)

    # Concatenate all results
    omnibus_all = pd.concat(all_omnibus, ignore_index=True)
    posthoc_all = pd.concat(all_posthoc, ignore_index=True)
    omnibus_all_unadj = pd.concat(all_omnibus_unadj, ignore_index=True)
    posthoc_all_unadj = pd.concat(all_posthoc_unadj, ignore_index=True)
    counts_all = pd.concat(all_counts, ignore_index=True)

    # Write all tables to the same output file, separated by blank lines
    with open(args.output, "w") as f:
        f.write("# Category counts per enzyme\n")
        counts_all.to_csv(f, index=False)
        f.write("\n# Omnibus test (SASA-adjusted)\n")
        omnibus_all.to_csv(f, index=False)
        f.write("\n# Post-hoc pairwise (SASA-adjusted)\n")
        posthoc_all.to_csv(f, index=False)
        f.write("\n# Omnibus test (raw conservation)\n")
        omnibus_all_unadj.to_csv(f, index=False)
        f.write("\n# Post-hoc pairwise (raw conservation)\n")
        posthoc_all_unadj.to_csv(f, index=False)
