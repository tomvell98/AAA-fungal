#!/usr/bin/env python3
"""
hotspot_entropy_matched_null.py

Test whether hotspot predictability is explained by "just variable sites" by building
an entropy-matched null.

Given:
  - MSA fasta (aligned)
  - umap_pca.csv with seq_id and PC columns
  - train/test IDs inferred from existing axis_score train/test CSVs (your pipeline outputs)
  - a hotspot TSV listing selected alignment columns (aln_col_1based)

We compute:
  - Observed test correlation between PC and an axis score built from the hotspot columns
  - Null distribution of test correlations for entropy-matched random column sets
  - Optional baselines: top-entropy set, random unmatched sets

Outputs:
  {out_prefix}.summary.csv
  {out_prefix}.null_dist.csv

Notes:
  - Uses TRAIN to fit per-column state effects and (optionally) per-column eta2 weights.
  - Evaluates on TEST (out-of-sample).
"""

import argparse
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Optional

AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY-")
AA_SET = set(AA_ALPHABET)

def read_msa_fasta(path: str) -> Tuple[List[str], List[str], int]:
    from Bio import SeqIO
    recs = list(SeqIO.parse(path, "fasta"))
    if not recs:
        raise ValueError("No sequences read from MSA.")
    ids = [r.id for r in recs]
    seqs = [str(r.seq).upper() for r in recs]
    L = len(seqs[0])
    if any(len(s) != L for s in seqs):
        raise ValueError("MSA sequences are not the same length.")
    return ids, seqs, L

def shannon_entropy(states: np.ndarray) -> float:
    # states: array of symbols (strings); NaNs removed upstream
    if states.size == 0:
        return np.nan
    cnt = Counter(states.tolist())
    n = float(states.size)
    p = np.array([c / n for c in cnt.values()], dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return q
    p_ok = p[ok]
    m = p_ok.size
    order = np.argsort(p_ok)
    ranked = p_ok[order]
    bh = ranked * m / (np.arange(1, m + 1))
    bh_mon = np.minimum.accumulate(bh[::-1])[::-1]
    bh_mon = np.clip(bh_mon, 0.0, 1.0)
    q_ok = np.empty_like(p_ok)
    q_ok[order] = bh_mon
    q[ok] = q_ok
    return q

def eta2_continuous(pc: np.ndarray, states: np.ndarray, min_group_size: int = 5) -> float:
    pc = np.asarray(pc, dtype=float)
    states = np.asarray(states, dtype=object)

    ok = np.isfinite(pc)
    pc = pc[ok]
    states = states[ok]
    if pc.size < 10:
        return np.nan

    groups = {}
    for v, s in zip(pc, states):
        groups.setdefault(s, []).append(float(v))
    groups = {k: v for k, v in groups.items() if len(v) >= min_group_size}
    if len(groups) < 2:
        return 0.0

    keep = set(groups.keys())
    mask = np.array([s in keep for s in states], dtype=bool)
    pc2 = pc[mask]
    st2 = states[mask]

    mu = float(np.mean(pc2))
    ss_total = float(np.sum((pc2 - mu) ** 2))
    if ss_total <= 1e-15:
        return 0.0

    ss_between = 0.0
    for k in keep:
        vals = pc2[st2 == k]
        if vals.size == 0:
            continue
        mu_k = float(np.mean(vals))
        ss_between += float(vals.size) * (mu_k - mu) ** 2
    return float(ss_between / ss_total)

def get_states_column(aln: List[str], j: int, gap_char: str, gap_mode: str) -> np.ndarray:
    """
    gap_mode:
      - "aa_only": gaps treated as missing (np.nan)
      - "exclude": gaps are a state '-'
    """
    col = np.array([s[j] if j < len(s) else gap_char for s in aln], dtype=object)
    col = np.array([c if c in AA_SET else gap_char for c in col], dtype=object)
    if gap_mode == "aa_only":
        out = col.astype(object)
        out[out == gap_char] = np.nan
        return out
    return col  # includes gaps as '-'

def corr_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]; y = y[ok]
    if x.size < 10:
        return np.nan
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x*x) * np.sum(y*y))
    if denom <= 0:
        return np.nan
    return float(np.sum(x*y) / denom)

def corr_spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        return np.nan
    return float(spearmanr(x[ok], y[ok]).correlation)

def fit_effects_and_score(
    msa_ids: List[str], msa_seqs: List[str], cols_1based: List[int],
    pc_map: Dict[str, float],
    train_ids: set, test_ids: set,
    gap_mode: str, gap_char: str,
    min_state_count: int, min_group_size: int,
    weight_mode: str
) -> Tuple[float, float]:
    """
    Build axis score from given columns:
      - Fit state effects on TRAIN
      - Optionally compute per-column eta2 on TRAIN and use as weights
      - Score TEST and return (pearson_r, spearman_rho) between axis_score and PC on TEST
    """
    # restrict to sequences present in pc_map and in msa
    msa_map = {sid: seq for sid, seq in zip(msa_ids, msa_seqs)}
    train_use = [sid for sid in train_ids if sid in msa_map and sid in pc_map]
    test_use  = [sid for sid in test_ids  if sid in msa_map and sid in pc_map]
    if len(train_use) < 50 or len(test_use) < 50:
        return (np.nan, np.nan)

    # precompute overall mean PC on TRAIN
    pc_train = np.array([pc_map[sid] for sid in train_use], dtype=float)
    overall = float(np.mean(pc_train))

    # Fit per-column effects dictionary and weights
    effects = {}  # (col_1based, state) -> centered mean
    weights = {}  # col_1based -> weight

    # build aligned train sequences list for convenience
    aln_train = [msa_map[sid] for sid in train_use]
    pc_train_vec = pc_train

    for c1 in cols_1based:
        j = int(c1) - 1
        if j < 0:
            continue
        # states on TRAIN at this column
        states = get_states_column(aln_train, j=j, gap_char=gap_char, gap_mode=gap_mode)

        # collapse rare AA states (ignore NaN)
        nonan = np.array([s for s in states.tolist() if not pd.isna(s)], dtype=object)
        if nonan.size == 0:
            continue
        cnt = Counter(nonan.tolist())
        states = np.array([np.nan if pd.isna(s) else (s if cnt[s] >= min_state_count else "X") for s in states], dtype=object)

        # compute eta2 weight if requested
        if weight_mode == "eta2":
            w = eta2_continuous(pc_train_vec, states, min_group_size=min_group_size)
            if not np.isfinite(w):
                w = 0.0
        else:
            w = 1.0
        weights[int(c1)] = float(w)

        # per-state centered mean PC effect
        for st in np.unique([s for s in states.tolist() if not pd.isna(s)]):
            vals = pc_train_vec[states == st]
            if vals.size < min_group_size:
                continue
            effects[(int(c1), str(st))] = float(np.mean(vals) - overall)

    if not weights:
        return (np.nan, np.nan)

    # Score TEST
    axis_scores = []
    pc_test = []

    for sid in test_use:
        seq = msa_map[sid]
        score = 0.0
        for c1, w in weights.items():
            j = c1 - 1
            if j >= len(seq):
                aa = gap_char
            else:
                aa = seq[j].upper()
                if aa not in AA_SET:
                    aa = gap_char

            if gap_mode == "aa_only" and aa == gap_char:
                continue
            st = aa if gap_mode != "aa_only" else aa  # aa_only just skipped gaps above

            # collapse rare states to X not feasible on test without counts; treat unseen as 0
            eff = effects.get((c1, st), None)
            if eff is None:
                eff = effects.get((c1, "X"), 0.0)  # if model had X
            score += float(w) * float(eff)

        axis_scores.append(float(score))
        pc_test.append(float(pc_map[sid]))

    axis_scores = np.array(axis_scores, dtype=float)
    pc_test = np.array(pc_test, dtype=float)

    return (corr_pearson(pc_test, axis_scores), corr_spearman(pc_test, axis_scores))

def entropy_table(
    msa_ids: List[str], msa_seqs: List[str], L: int,
    ids_subset: set,
    gap_mode: str, gap_char: str,
    drop_gappy_cols_gt: Optional[float]
) -> pd.DataFrame:
    msa_map = {sid: seq for sid, seq in zip(msa_ids, msa_seqs)}
    use_ids = [sid for sid in msa_ids if sid in ids_subset and sid in msa_map]
    aln = [msa_map[sid] for sid in use_ids]
    if len(aln) < 50:
        raise ValueError("Too few sequences to compute entropy.")

    rows = []
    for j in range(L):
        raw = np.array([s[j] if j < len(s) else gap_char for s in aln], dtype=object)
        raw = np.array([c if c in AA_SET else gap_char for c in raw], dtype=object)
        gap_frac = float(np.mean(raw == gap_char))
        if drop_gappy_cols_gt is not None and gap_frac > drop_gappy_cols_gt:
            continue

        states = get_states_column(aln, j=j, gap_char=gap_char, gap_mode=gap_mode)
        states = np.array([s for s in states.tolist() if not pd.isna(s)], dtype=object)
        H = shannon_entropy(states)
        rows.append({"aln_col_1based": j+1, "entropy": H, "gap_frac": gap_frac, "n_states": int(len(set(states.tolist())))})
    return pd.DataFrame(rows)

def sample_entropy_matched_sets(
    rng: np.random.Generator,
    candidate_df: pd.DataFrame,
    hotspot_cols: List[int],
    n_bins: int,
    n_samples: int
) -> List[List[int]]:
    # Bin by entropy quantiles on candidates
    cand = candidate_df.dropna(subset=["entropy"]).copy()
    qs = np.linspace(0, 1, n_bins+1)
    edges = np.quantile(cand["entropy"].values, qs)
    # make bins stable
    edges[0] -= 1e-9
    edges[-1] += 1e-9

    cand["bin"] = np.digitize(cand["entropy"].values, edges[1:-1], right=False)  # 0..n_bins-1

    hs = cand[cand["aln_col_1based"].isin(hotspot_cols)].copy()
    if hs.empty:
        raise ValueError("Hotspot columns not found in entropy table after filtering.")
    hs["bin"] = np.digitize(hs["entropy"].values, edges[1:-1], right=False)

    # how many hotspots per bin?
    need = hs["bin"].value_counts().to_dict()

    # precompute candidate pools per bin (exclude hotspot cols to avoid trivial re-picks)
    pools = {}
    hs_set = set(hotspot_cols)
    for b in range(n_bins):
        pool = cand[(cand["bin"] == b) & (~cand["aln_col_1based"].isin(hs_set))]["aln_col_1based"].astype(int).tolist()
        pools[b] = pool

    out = []
    K = len(hotspot_cols)

    for _ in range(n_samples):
        cols = []
        for b, k in need.items():
            pool = pools.get(int(b), [])
            if len(pool) < k:
                # if pool too small, allow sampling from all candidates in that bin including hotspots (rare)
                pool = cand[cand["bin"] == int(b)]["aln_col_1based"].astype(int).tolist()
            pick = rng.choice(pool, size=int(k), replace=False).tolist()
            cols.extend(pick)
        if len(cols) != K:
            # fallback: sample K from all candidates if binning failed
            cols = rng.choice(cand["aln_col_1based"].astype(int).values, size=K, replace=False).tolist()
        out.append([int(c) for c in cols])
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msa_fasta", required=True)
    ap.add_argument("--umap_pca_csv", required=True, help="e.g., ENZ.umap_pca.csv")
    ap.add_argument("--pc_col", required=True, choices=["PC1", "PC2"])
    ap.add_argument("--axis_train_csv", required=True, help="e.g., ENZ.PC1.axis_score_AA.split.train.csv")
    ap.add_argument("--axis_test_csv", required=True, help="e.g., ENZ.PC1.axis_score_AA.split.test.csv")
    ap.add_argument("--hotspot_tsv", required=True, help="e.g., ENZ.PC1.hotspots_AA.split.train_top25.tsv or stable.tsv")
    ap.add_argument("--hotspot_col", default="aln_col_1based")
    ap.add_argument("--top_k", type=int, default=25, help="If hotspot_tsv is not already topK, take head(top_k). Use <=0 for all.")
    ap.add_argument("--gap_mode", default="aa_only", choices=["aa_only", "exclude"])
    ap.add_argument("--gap_char", default="-")
    ap.add_argument("--drop_gappy_cols_gt", type=float, default=0.5)
    ap.add_argument("--min_state_count", type=int, default=5)
    ap.add_argument("--min_group_size", type=int, default=5)
    ap.add_argument("--weight_mode", default="eta2", choices=["eta2", "uniform"])
    ap.add_argument("--n_null", type=int, default=1000)
    ap.add_argument("--entropy_bins", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--also_top_entropy", action="store_true")
    ap.add_argument("--also_random_unmatched", action="store_true")
    ap.add_argument("--out_prefix", required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    msa_ids, msa_seqs, L = read_msa_fasta(args.msa_fasta)

    # PC map
    df = pd.read_csv(args.umap_pca_csv)
    if "seq_id" not in df.columns:
        raise ValueError("umap_pca_csv must have seq_id column.")
    if args.pc_col not in df.columns:
        raise ValueError(f"umap_pca_csv missing {args.pc_col}.")
    pc_map = dict(zip(df["seq_id"].astype(str).tolist(), df[args.pc_col].astype(float).tolist()))

    # Train/test IDs inferred from existing axis_score files (ensures same split)
    tr = pd.read_csv(args.axis_train_csv)
    te = pd.read_csv(args.axis_test_csv)
    train_ids = set(tr["seq_id"].astype(str).tolist())
    test_ids  = set(te["seq_id"].astype(str).tolist())

    # Hotspot columns
    hs = pd.read_csv(args.hotspot_tsv, sep="\t" if args.hotspot_tsv.endswith(".tsv") else ",")
    if args.hotspot_col not in hs.columns:
        raise ValueError(f"hotspot_tsv missing column {args.hotspot_col}.")
    if args.top_k and args.top_k > 0:
        hs = hs.head(int(args.top_k)).copy()
    hotspot_cols = hs[args.hotspot_col].astype(int).tolist()
    K = len(hotspot_cols)

    drop_gappy = args.drop_gappy_cols_gt
    if drop_gappy is not None and drop_gappy > 1.0:
        drop_gappy = None

    # Entropy table on TRAIN (same subset used to fit effects)
    ent_df = entropy_table(
        msa_ids=msa_ids, msa_seqs=msa_seqs, L=L,
        ids_subset=train_ids,
        gap_mode=args.gap_mode, gap_char=args.gap_char,
        drop_gappy_cols_gt=drop_gappy
    )

    # Candidate pool
    candidate_df = ent_df.copy()

    # Observed performance
    obs_r, obs_rho = fit_effects_and_score(
        msa_ids, msa_seqs, hotspot_cols,
        pc_map, train_ids, test_ids,
        args.gap_mode, args.gap_char,
        args.min_state_count, args.min_group_size,
        args.weight_mode
    )

    # Null: entropy-matched sets
    null_sets = sample_entropy_matched_sets(
        rng=rng, candidate_df=candidate_df, hotspot_cols=hotspot_cols,
        n_bins=int(args.entropy_bins), n_samples=int(args.n_null)
    )

    null_rows = []
    for i, cols in enumerate(null_sets, start=1):
        r, rho = fit_effects_and_score(
            msa_ids, msa_seqs, cols,
            pc_map, train_ids, test_ids,
            args.gap_mode, args.gap_char,
            args.min_state_count, args.min_group_size,
            args.weight_mode
        )
        null_rows.append({"i": i, "test_pearson_r": r, "test_spearman_rho": rho})

    null_df = pd.DataFrame(null_rows)

    # empirical p-values: how often null >= observed
    def emp_p(null_vals, obs):
        v = np.asarray(null_vals, dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0 or not np.isfinite(obs):
            return np.nan
        return float((np.sum(v >= obs) + 1) / (v.size + 1))

    p_r = emp_p(null_df["test_pearson_r"].values, obs_r)
    p_rho = emp_p(null_df["test_spearman_rho"].values, obs_rho)

    # z-scores
    def zscore(null_vals, obs):
        v = np.asarray(null_vals, dtype=float)
        v = v[np.isfinite(v)]
        if v.size < 5 or not np.isfinite(obs):
            return np.nan
        mu = float(np.mean(v))
        sd = float(np.std(v, ddof=1))
        if sd <= 0:
            return np.nan
        return float((obs - mu) / sd)

    z_r = zscore(null_df["test_pearson_r"].values, obs_r)
    z_rho = zscore(null_df["test_spearman_rho"].values, obs_rho)

    # Optional: top-entropy baseline
    topent_r = topent_rho = np.nan
    if args.also_top_entropy:
        top_cols = candidate_df.sort_values("entropy", ascending=False).head(K)["aln_col_1based"].astype(int).tolist()
        topent_r, topent_rho = fit_effects_and_score(
            msa_ids, msa_seqs, top_cols,
            pc_map, train_ids, test_ids,
            args.gap_mode, args.gap_char,
            args.min_state_count, args.min_group_size,
            args.weight_mode
        )

    # Optional: random unmatched baseline
    rand_r = rand_rho = np.nan
    if args.also_random_unmatched:
        cand_cols = candidate_df["aln_col_1based"].astype(int).values
        cols = rng.choice(cand_cols, size=K, replace=False).tolist()
        rand_r, rand_rho = fit_effects_and_score(
            msa_ids, msa_seqs, cols,
            pc_map, train_ids, test_ids,
            args.gap_mode, args.gap_char,
            args.min_state_count, args.min_group_size,
            args.weight_mode
        )

    summary = pd.DataFrame([{
        "pc_col": args.pc_col,
        "K": K,
        "gap_mode": args.gap_mode,
        "drop_gappy_cols_gt": args.drop_gappy_cols_gt,
        "weight_mode": args.weight_mode,
        "obs_test_pearson_r": obs_r,
        "obs_test_spearman_rho": obs_rho,
        "null_n": int(len(null_df)),
        "emp_p_pearson": p_r,
        "emp_p_spearman": p_rho,
        "z_pearson": z_r,
        "z_spearman": z_rho,
        "top_entropy_test_pearson_r": topent_r,
        "top_entropy_test_spearman_rho": topent_rho,
        "random_unmatched_test_pearson_r": rand_r,
        "random_unmatched_test_spearman_rho": rand_rho,
        "hotspot_tsv": args.hotspot_tsv,
        "axis_train_csv": args.axis_train_csv,
        "axis_test_csv": args.axis_test_csv
    }])

    out_sum = f"{args.out_prefix}.summary.csv"
    out_null = f"{args.out_prefix}.null_dist.csv"
    summary.to_csv(out_sum, index=False)
    null_df.to_csv(out_null, index=False)

    print("Wrote:", out_sum)
    print("Wrote:", out_null)
    print(f"Observed test r={obs_r:.4f} (emp_p={p_r:.4g}, z={z_r:.3f})")
    print(f"Observed test rho={obs_rho:.4f} (emp_p={p_rho:.4g}, z={z_rho:.3f})")

if __name__ == "__main__":
    main()
