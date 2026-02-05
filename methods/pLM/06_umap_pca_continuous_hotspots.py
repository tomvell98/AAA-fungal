#!/usr/bin/env python3
"""
06_umap_pca_continuous_hotspots.py

Robust "continuous hotspots" pipeline on residualized embeddings.

Adds three critical upgrades:
  (1) Out-of-sample (OOS) hotspot discovery:
      - train/test split or k-fold CV for hotspot selection and axis-score evaluation.
  (2) Explicit gap handling:
      - gap_mode=exclude   : drop gappy columns and treat gaps as missing (default)
      - gap_mode=indel     : treat GAP vs NON-GAP as a binary "indel hotspot" channel
      - gap_mode=aa_only   : ignore gaps in state definition; compute eta2 among non-gap residues only
      Optional: run both AA and INDEL channels.
  (3) Bootstrap stability:
      - bootstrap re-samples (optionally stratified by Phylum) and records how often a column
        appears in topK/topM -> "stable hotspots".

Core steps:
  1) Load distance matrix -> HDBSCAN + UMAP (visualization)
  2) Load residualized embeddings -> PCA -> PC scores per sequence
  3) Trait association: eta^2(PC ~ trait)
  4) Continuous hotspots: eta^2(PC ~ residue_state_at_col) per MSA column (with permutations + BH)
  5) Axis score from topK hotspots discovered on TRAIN only, evaluated on TEST (OOS)
  6) Bootstrap stability & "stable" hotspot outputs

Inputs:
  --dist_csv, --emb_npy, --ids_txt, --msa_fasta, --metadata_csv, --out_prefix

Outputs (key):
  {out_prefix}.umap_pca.csv
  {out_prefix}.clusters.csv
  {out_prefix}.pca_variance.csv/.txt
  {out_prefix}.pc_trait_eta2.csv/.txt

Hotspots:
  {out_prefix}.{PC}.hotspots_{channel}.tsv
  {out_prefix}.{PC}.hotspots_{channel}_top{K}.tsv
  {out_prefix}.{PC}.hotspots_{channel}_stable.tsv
  {out_prefix}.{PC}.hotspots_{channel}_bootstrap_freq.tsv

Axis score:
  {out_prefix}.{PC}.axis_score_{channel}.train.csv
  {out_prefix}.{PC}.axis_score_{channel}.test.csv
  {out_prefix}.axis_score_summary.csv/.txt

Notes:
  - "channel" is one of: AA, INDEL (if enabled).
  - Permutation p-values shuffle PC values across sequences (within the evaluated set).
  - BH-FDR is applied per PC per channel across tested columns.
"""

import argparse
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Optional


AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY-")  # include gap
AA_TO_I = {a: i for i, a in enumerate(AA_ALPHABET)}


# ---------------- IO helpers ----------------

def load_square_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    if df.shape[0] != df.shape[1]:
        raise ValueError(f"{path} is not square: {df.shape}")
    if not (df.index == df.columns).all():
        if set(df.index) == set(df.columns):
            df = df.loc[df.index, df.index]
        else:
            raise ValueError("Row/column labels differ; cannot interpret as square distance matrix.")
    return df


def load_ids_txt(path: str) -> List[str]:
    ids = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if s:
                ids.append(s)
    if not ids:
        raise ValueError("ids_txt is empty.")
    return ids


def load_embeddings_npy(path: str) -> np.ndarray:
    arr = np.load(path)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected embeddings to be 2D (n,d). Got shape {arr.shape}")
    return arr.astype(float)


def read_msa_fasta(path: str) -> Tuple[List[str], List[str], int]:
    from Bio import SeqIO
    records = list(SeqIO.parse(path, "fasta"))
    if not records:
        raise ValueError("No sequences read from MSA.")
    ids = [r.id for r in records]
    seqs = [str(r.seq).upper() for r in records]
    L = len(seqs[0])
    if any(len(s) != L for s in seqs):
        raise ValueError("MSA sequences are not the same length.")
    return ids, seqs, L


def read_metadata_csv(path: str, id_col: str = "Accession") -> pd.DataFrame:
    meta = pd.read_csv(path, sep=",")
    if id_col not in meta.columns:
        raise ValueError(f"Metadata missing id_col '{id_col}'. Columns: {list(meta.columns)}")
    meta = meta.rename(columns={id_col: "seq_id"})
    meta["seq_id"] = meta["seq_id"].astype(str)
    return meta


# ---------------- Category collapsing / cleaning ----------------

def _clean_cat(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return np.nan
    if s == "0":
        return np.nan
    return s


def collapse_rare_categories(s: pd.Series, min_count: int, other_label="Other") -> pd.Series:
    """Keep categories with count >= min_count, collapse others to Other. NaN stays NaN."""
    if min_count <= 1:
        return s
    counts = s.value_counts(dropna=True)
    keep = set(counts[counts >= min_count].index.astype(str))

    def _map(v):
        if pd.isna(v):
            return np.nan
        v = str(v)
        return v if v in keep else other_label

    return s.apply(_map)


def collapse_growth_form(s: pd.Series) -> pd.Series:
    """
    Collapse Growth_form_template to:
      Filamentous mycelium, Yeast-like, Thallus photosynthetic, Zoosporic, Unicellular non-yeast, Other
    """
    def _map(v):
        v = _clean_cat(v)
        if pd.isna(v):
            return np.nan
        vlow = v.lower()
        if "filamentous" in vlow:
            return "Filamentous mycelium"
        if "yeast" in vlow:
            return "Yeast-like"
        if "thallus" in vlow or "photosynthetic" in vlow:
            return "Thallus photosynthetic"
        if "zoosporic" in vlow or "chytrid" in vlow or "rhizomycelial" in vlow or "plasmodium" in vlow:
            return "Zoosporic"
        if "unicellular" in vlow:
            return "Unicellular non-yeast"
        return "Other"
    return s.apply(_map)


def lifestyle_macro_guild(s: pd.Series) -> pd.Series:
    """
    Optional biological reduction:
      Saprotroph, Plant pathogen, Animal/Protist parasite, Symbiotroph/Endophyte, Mycoparasite, Other
    """
    sap = {
        "soil_saprotroph", "wood_saprotroph", "litter_saprotroph", "dung_saprotroph",
        "pollen_saprotroph", "nectar/tap_saprotroph", "unspecified_saprotroph"
    }
    plant_path = {"plant_pathogen", "unspecified_pathotroph", "sooty_mold"}
    animal_prot = {"animal_parasite", "protistan_parasite", "animal-associated", "arthropod-associated", "algal_parasite"}
    symb = {"ectomycorrhizal", "arbuscular_mycorrhizal", "lichenized", "foliar_endophyte", "root_endophyte",
            "epiphyte", "moss_symbiont", "animal_endosymbiont"}
    myco = {"mycoparasite", "lichen_parasite"}

    def _map(v):
        v = _clean_cat(v)
        if pd.isna(v):
            return np.nan
        if v in sap:
            return "Saprotroph"
        if v in plant_path:
            return "Plant pathogen"
        if v in animal_prot:
            return "Animal/Protist parasite"
        if v in symb:
            return "Symbiotroph/Endophyte"
        if v in myco:
            return "Mycoparasite"
        return "Other"

    return s.apply(_map)


# ---------------- Plotting helpers ----------------

def _save_fig(fig, base_path_no_ext: str, fmt: str):
    if fmt in ("png", "both"):
        fig.savefig(base_path_no_ext + ".png", dpi=300)
    if fmt in ("pdf", "both"):
        fig.savefig(base_path_no_ext + ".pdf")


def plot_2d_by_cluster(df: pd.DataFrame, xcol: str, ycol: str, out_prefix: str, fmt: str,
                       title: str, point_size: float, alpha: float, label_centroids: bool = False):
    import matplotlib.pyplot as plt
    noise = df[df["cluster"] == -1]
    non_noise = df[df["cluster"] != -1]

    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    if len(noise) > 0:
        ax.scatter(noise[xcol], noise[ycol], s=point_size, alpha=0.25, label="noise (-1)")

    for c in sorted(non_noise["cluster"].unique()):
        sub = non_noise[non_noise["cluster"] == c]
        ax.scatter(sub[xcol], sub[ycol], s=point_size, alpha=alpha, label=f"cluster {c} (n={len(sub)})")

        if label_centroids and len(sub) > 0:
            ax.text(float(sub[xcol].mean()), float(sub[ycol].mean()), str(c),
                    fontsize=10, ha="center", va="center")

    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title(title)
    ax.legend(markerscale=2, fontsize=8, frameon=False, loc="best")
    ax.grid(False)
    fig.tight_layout()

    _save_fig(fig, out_prefix, fmt)
    plt.close(fig)


def plot_2d_by_category(df: pd.DataFrame, xcol: str, ycol: str, catcol: str, out_prefix: str, fmt: str,
                        title: str, point_size: float, alpha: float, na_label="NA", max_legend: int = 18):
    import matplotlib.pyplot as plt
    if catcol not in df.columns:
        return

    plot_df = df.copy()
    plot_df[catcol] = plot_df[catcol].where(plot_df[catcol].notna(), na_label).astype(str)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    df_na = plot_df[plot_df[catcol] == na_label]
    df_ok = plot_df[plot_df[catcol] != na_label]

    if len(df_na) > 0:
        ax.scatter(df_na[xcol], df_na[ycol], s=point_size, alpha=0.25, label=f"{na_label} (n={len(df_na)})")

    cats = sorted(df_ok[catcol].unique())
    if len(cats) > max_legend:
        for cat in cats:
            sub = df_ok[df_ok[catcol] == cat]
            ax.scatter(sub[xcol], sub[ycol], s=point_size, alpha=alpha)
        ax.text(0.01, 0.99, f"{catcol}: {len(cats)} categories (legend suppressed)",
                transform=ax.transAxes, ha="left", va="top", fontsize=9)
    else:
        for cat in cats:
            sub = df_ok[df_ok[catcol] == cat]
            ax.scatter(sub[xcol], sub[ycol], s=point_size, alpha=alpha, label=f"{cat} (n={len(sub)})")
        ax.legend(markerscale=2, fontsize=8, frameon=False, loc="best")

    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title(title)
    ax.grid(False)
    fig.tight_layout()

    _save_fig(fig, out_prefix, fmt)
    plt.close(fig)


def plot_axis_score_scatter(score_df: pd.DataFrame, pc_col: str, out_prefix: str, fmt: str,
                            title: str, point_size: float, alpha: float,
                            annotate: Optional[str] = None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    ax.scatter(score_df[pc_col].values, score_df["axis_score"].values, s=point_size, alpha=alpha)
    ax.set_xlabel(pc_col)
    ax.set_ylabel("Axis score")
    ax.set_title(title)
    ax.grid(False)
    if annotate:
        ax.text(0.02, 0.98, annotate, transform=ax.transAxes, ha="left", va="top", fontsize=10)
    fig.tight_layout()
    _save_fig(fig, out_prefix, fmt)
    plt.close(fig)


# ---------------- PCA variance output ----------------

def save_pca_variance(pca, out_prefix: str):
    var = pca.explained_variance_ratio_.astype(float)
    df = pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(len(var))],
        "explained_variance_ratio": var,
        "explained_variance_percent": 100.0 * var,
        "cumulative_percent": 100.0 * np.cumsum(var)
    })
    df.to_csv(f"{out_prefix}.pca_variance.csv", index=False)

    with open(f"{out_prefix}.pca_variance.txt", "w") as f:
        for i, v in enumerate(var, start=1):
            f.write(f"PC{i}\t{v:.6f}\t{100.0*v:.3f}%\n")
        f.write(f"CUM_PC2\t{100.0*np.sum(var[:2]):.3f}%\n")
    print(f"Wrote: {out_prefix}.pca_variance.csv")
    print(f"Wrote: {out_prefix}.pca_variance.txt")


# ---------------- Effect size: eta^2 for PC ~ category ----------------

def eta_squared_pc_category(df: pd.DataFrame, pc_col: str, cat_col: str,
                            permutations: int = 0, random_state: int = 1,
                            drop_na: bool = True, min_group_size: int = 2) -> dict:
    """
    eta^2 = SS_between / SS_total for PC ~ category.
    Optional permutation p-value: shuffle category labels.
    """
    sub = df[[pc_col, cat_col]].copy()
    if drop_na:
        sub = sub[sub[cat_col].notna()].copy()

    x = sub[pc_col].astype(float).values
    g = sub[cat_col].astype(str).values
    n_used = int(len(x))
    if n_used < 3:
        return {"pc": pc_col, "trait": cat_col, "n_used": n_used, "n_levels": 0,
                "eta2": np.nan, "eta2_p": np.nan, "smallest_group": 0}

    groups: Dict[str, List[float]] = {}
    for val, lab in zip(x, g):
        groups.setdefault(lab, []).append(float(val))

    groups = {k: v for k, v in groups.items() if len(v) >= min_group_size}
    if len(groups) < 2:
        return {"pc": pc_col, "trait": cat_col, "n_used": n_used, "n_levels": int(len(groups)),
                "eta2": 0.0, "eta2_p": np.nan,
                "smallest_group": int(min((len(v) for v in groups.values()), default=0))}

    keep = set(groups.keys())
    mask = np.array([lab in keep for lab in g], dtype=bool)
    x2 = x[mask]
    g2 = g[mask]
    n2 = int(len(x2))

    overall_mean = float(np.mean(x2))
    ss_total = float(np.sum((x2 - overall_mean) ** 2))
    if ss_total <= 0:
        eta2 = 0.0
    else:
        ss_between = 0.0
        for lab in keep:
            vals = x2[g2 == lab]
            if len(vals) == 0:
                continue
            mu = float(np.mean(vals))
            ss_between += float(len(vals)) * (mu - overall_mean) ** 2
        eta2 = float(ss_between / ss_total)

    out = {
        "pc": pc_col,
        "trait": cat_col,
        "n_used": n2,
        "n_levels": int(len(keep)),
        "eta2": eta2,
        "eta2_p": np.nan,
        "smallest_group": int(min(len(x2[g2 == lab]) for lab in keep)) if keep else 0
    }

    if permutations and permutations > 0 and len(keep) >= 2:
        rng = np.random.default_rng(random_state)
        perm_vals = []
        for _ in range(permutations):
            g_shuf = rng.permutation(g2)
            overall = float(np.mean(x2))
            ss_tot = float(np.sum((x2 - overall) ** 2))
            if ss_tot <= 0:
                perm_vals.append(0.0)
                continue
            ss_b = 0.0
            for lab in keep:
                vals = x2[g_shuf == lab]
                if len(vals) == 0:
                    continue
                mu = float(np.mean(vals))
                ss_b += float(len(vals)) * (mu - overall) ** 2
            perm_vals.append(float(ss_b / ss_tot))

        perm_vals = np.asarray(perm_vals, dtype=float)
        out["eta2_p"] = float((np.sum(perm_vals >= eta2) + 1) / (len(perm_vals) + 1))

    return out


# ---------------- Multiple testing: BH-FDR ----------------

def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg FDR q-values.
    pvals: array of p-values (nan allowed)
    Returns q-values array aligned to pvals.
    """
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


# ---------------- Splits / resampling ----------------

def train_test_split_ids(ids: List[str], test_frac: float, rng: np.random.Generator,
                         stratify_labels: Optional[pd.Series] = None) -> Tuple[List[str], List[str]]:
    ids = list(ids)
    if stratify_labels is None:
        perm = rng.permutation(len(ids))
        n_test = int(round(test_frac * len(ids)))
        test_idx = set(perm[:n_test].tolist())
        test = [ids[i] for i in range(len(ids)) if i in test_idx]
        train = [ids[i] for i in range(len(ids)) if i not in test_idx]
        return train, test

    # stratified split by label (e.g., Phylum), ignoring NaNs
    labels = stratify_labels.reindex(ids)
    label_to_ids = defaultdict(list)
    for sid, lab in labels.items():
        if pd.isna(lab):
            label_to_ids["__NA__"].append(sid)
        else:
            label_to_ids[str(lab)].append(sid)

    train, test = [], []
    for lab, lab_ids in label_to_ids.items():
        lab_ids = list(lab_ids)
        perm = rng.permutation(len(lab_ids))
        n_test = int(round(test_frac * len(lab_ids)))
        test.extend([lab_ids[i] for i in perm[:n_test]])
        train.extend([lab_ids[i] for i in perm[n_test:]])
    return train, test


def bootstrap_resample_ids(ids: List[str], rng: np.random.Generator,
                           sample_frac: float = 1.0,
                           stratify_labels: Optional[pd.Series] = None) -> List[str]:
    n = len(ids)
    n_samp = int(round(sample_frac * n))
    if stratify_labels is None:
        idx = rng.integers(0, n, size=n_samp)
        return [ids[i] for i in idx]

    labels = stratify_labels.reindex(ids)
    label_to_ids = defaultdict(list)
    for sid, lab in labels.items():
        key = "__NA__" if pd.isna(lab) else str(lab)
        label_to_ids[key].append(sid)

    out = []
    for lab, lab_ids in label_to_ids.items():
        lab_ids = list(lab_ids)
        m = len(lab_ids)
        m_samp = max(1, int(round(sample_frac * m)))
        idx = rng.integers(0, m, size=m_samp)
        out.extend([lab_ids[i] for i in idx])
    return out


# ---------------- Continuous hotspots helpers ----------------

def _collapse_rare_states(states: np.ndarray, min_count: int, other_label: str = "X") -> np.ndarray:
    if min_count <= 1:
        return states
    cnt = Counter(states.tolist())
    return np.array([s if cnt[s] >= min_count else other_label for s in states], dtype=object)


def eta2_continuous(pc: np.ndarray, states: np.ndarray, min_group_size: int = 2) -> float:
    pc = np.asarray(pc, dtype=float)
    states = np.asarray(states, dtype=object)

    ok = np.isfinite(pc)
    pc = pc[ok]
    states = states[ok]
    if pc.size < 3:
        return np.nan

    levels = {}
    for v, s in zip(pc, states):
        levels.setdefault(s, []).append(float(v))
    levels = {k: v for k, v in levels.items() if len(v) >= min_group_size}
    if len(levels) < 2:
        return 0.0

    keep = set(levels.keys())
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


def get_column_states_for_channel(aln: List[str], j: int, gap_char: str,
                                  channel: str,  # "AA" or "INDEL"
                                  gap_mode: str  # exclude|indel|aa_only
                                  ) -> np.ndarray:
    """
    Returns per-seq "state" at column j for a specific channel.
    - AA channel:
        * gap_mode=exclude : gaps kept as '-' state, but downstream may drop gappy columns; useful for occupancy patterns
        * gap_mode=aa_only : gaps treated as missing -> state = np.nan for gaps (removed in eta2_continuous via mask)
    - INDEL channel:
        state = "GAP" or "NONGAP" (always)
    """
    col = np.array([s[j] if j < len(s) else gap_char for s in aln], dtype=object)
    col = np.array([ch if ch in AA_TO_I else gap_char for ch in col], dtype=object)

    if channel == "INDEL":
        return np.array(["GAP" if ch == gap_char else "NONGAP" for ch in col], dtype=object)

    # channel == "AA"
    if gap_mode == "aa_only":
        # treat gaps as missing
        out = col.astype(object)
        out[out == gap_char] = np.nan
        return out
    else:
        # include gaps as a legitimate state
        return col


def continuous_hotspots(msa_ids: List[str], msa_seqs: List[str], L: int,
                        pc_df: pd.DataFrame, pc_col: str,
                        seq_ids_subset: Optional[set],
                        channel: str,
                        gap_mode: str,
                        gap_char: str,
                        drop_gappy_cols_gt: Optional[float],
                        min_state_count: int,
                        min_group_size: int,
                        permutations: int,
                        rng: np.random.Generator,
                        verbose_every: int = 0) -> pd.DataFrame:
    """
    Compute eta^2 per column for a given PC and channel on a specified subset of sequences (train/test/bootstrap).
    """
    pc_map = dict(zip(pc_df["seq_id"].astype(str).tolist(), pc_df[pc_col].astype(float).tolist()))

    use_idx = [i for i, sid in enumerate(msa_ids) if sid in pc_map and (seq_ids_subset is None or sid in seq_ids_subset)]
    if len(use_idx) < 50:
        raise ValueError(f"Too few sequences available for hotspots ({pc_col}, {channel}): {len(use_idx)}")

    ids = [msa_ids[i] for i in use_idx]
    aln = [msa_seqs[i] for i in use_idx]
    pc = np.array([pc_map[sid] for sid in ids], dtype=float)

    rows = []
    for j in range(L):
        # gap fraction is computed from raw AA column (including gaps)
        raw_col = np.array([s[j] if j < len(s) else gap_char for s in aln], dtype=object)
        raw_col = np.array([ch if ch in AA_TO_I else gap_char for ch in raw_col], dtype=object)
        gap_frac = float(np.mean(raw_col == gap_char))

        if drop_gappy_cols_gt is not None and gap_frac > drop_gappy_cols_gt:
            continue

        states = get_column_states_for_channel(aln, j, gap_char=gap_char, channel=channel, gap_mode=gap_mode)

        # collapse rare states (ignore NaN in counting)
        if channel == "AA":
            st_nonan = np.array([s for s in states.tolist() if not pd.isna(s)], dtype=object)
            if st_nonan.size == 0:
                continue
            st_collapsed = states.copy()
            collapsed = _collapse_rare_states(st_nonan, min_count=min_state_count, other_label="X")
            # map back: build a dict from original -> collapsed for nonan
            # easiest: recompute counts on st_nonan then apply rule directly to full states
            cnt = Counter(st_nonan.tolist())
            st_collapsed = np.array([np.nan if pd.isna(s) else (s if cnt[s] >= min_state_count else "X") for s in states], dtype=object)
            states = st_collapsed
        else:
            # INDEL has only GAP/NONGAP; no collapse needed
            pass

        eta2_obs = eta2_continuous(pc, states, min_group_size=min_group_size)

        p_perm = np.nan
        if permutations and permutations > 0:
            perm_vals = np.zeros(permutations, dtype=float)
            for b in range(permutations):
                pc_shuf = rng.permutation(pc)
                perm_vals[b] = eta2_continuous(pc_shuf, states, min_group_size=min_group_size)
            p_perm = float((np.sum(perm_vals >= eta2_obs) + 1) / (permutations + 1))

        # counts for audit
        st_nonan = [s for s in states.tolist() if not pd.isna(s)]
        cnt = Counter(st_nonan)
        rows.append({
            "pc": pc_col,
            "channel": channel,
            "aln_col_1based": j + 1,
            "eta2": float(eta2_obs),
            "p_perm": p_perm,
            "gap_frac": gap_frac,
            "n_seq": int(len(pc)),
            "n_states_raw": int(len(cnt)),
            "top_state": cnt.most_common(1)[0][0] if cnt else "NA",
            "top_state_n": int(cnt.most_common(1)[0][1]) if cnt else 0,
        })

        if verbose_every and (j + 1) % verbose_every == 0:
            print(f"[{pc_col}][{channel}] processed col {j+1}/{L}")

    out = pd.DataFrame(rows)
    out["q_bh"] = bh_fdr(out["p_perm"].values if "p_perm" in out.columns else np.full(len(out), np.nan))
    out["minuslog10p"] = -np.log10(np.clip(out["p_perm"].astype(float).values, 1e-300, 1.0))
    out["minuslog10q"] = -np.log10(np.clip(out["q_bh"].astype(float).values, 1e-300, 1.0))
    out = out.sort_values(["eta2", "q_bh"], ascending=[False, True]).reset_index(drop=True)
    return out


# ---------------- Axis score: train effects -> score any set ----------------

def fit_site_effects_from_hotspots(msa_ids: List[str], msa_seqs: List[str], L: int,
                                   pc_df: pd.DataFrame, pc_col: str,
                                   train_ids: set,
                                   hotspots_df: pd.DataFrame,
                                   top_k: int,
                                   weight_col: str,
                                   channel: str,
                                   gap_mode: str,
                                   gap_char: str,
                                   min_state_count: int,
                                   min_group_size: int) -> pd.DataFrame:
    """
    Fit per-site, per-state centered mean PC on TRAIN only, for topK hotspot columns.

    Returns a long dataframe:
      aln_col_1based, state, weight, effect_centered_mean_pc, n_state
    """
    pc_map = dict(zip(pc_df["seq_id"].astype(str).tolist(), pc_df[pc_col].astype(float).tolist()))
    use_idx = [i for i, sid in enumerate(msa_ids) if sid in pc_map and sid in train_ids]
    if len(use_idx) < 50:
        raise ValueError(f"Too few TRAIN sequences for site effects: {pc_col} {channel} n={len(use_idx)}")

    ids = [msa_ids[i] for i in use_idx]
    aln = [msa_seqs[i] for i in use_idx]
    pc = np.array([pc_map[sid] for sid in ids], dtype=float)
    overall_mean = float(np.mean(pc))

    top = hotspots_df.head(int(top_k)).copy()
    cols = top["aln_col_1based"].astype(int).values
    weights = top[weight_col].astype(float).values

    rows = []
    for c1, w in zip(cols, weights):
        j = int(c1) - 1
        if j < 0 or j >= L:
            continue
        states = get_column_states_for_channel(aln, j, gap_char=gap_char, channel=channel, gap_mode=gap_mode)

        # collapse rare states for AA
        if channel == "AA":
            st_nonan = np.array([s for s in states.tolist() if not pd.isna(s)], dtype=object)
            if st_nonan.size == 0:
                continue
            cnt = Counter(st_nonan.tolist())
            states = np.array([np.nan if pd.isna(s) else (s if cnt[s] >= min_state_count else "X") for s in states], dtype=object)

        for st in np.unique([s for s in states.tolist() if not pd.isna(s)]):
            vals = pc[states == st]
            if vals.size < min_group_size:
                continue
            mu = float(np.mean(vals)) - overall_mean
            rows.append({
                "pc": pc_col,
                "channel": channel,
                "aln_col_1based": j + 1,
                "state": st,
                "weight": float(w),
                "effect_centered_mean_pc": float(mu),
                "n_state": int(vals.size),
            })

    eff = pd.DataFrame(rows)
    return eff


def score_sequences_with_site_effects(msa_ids: List[str], msa_seqs: List[str], L: int,
                                      pc_df: pd.DataFrame, pc_col: str,
                                      target_ids: set,
                                      site_effects_df: pd.DataFrame,
                                      channel: str,
                                      gap_mode: str,
                                      gap_char: str) -> pd.DataFrame:
    """
    Use fitted site effects (from TRAIN) to score sequences in target_ids.
    For each site, contribution = weight * effect(state). If state unseen, contribution = 0.

    Returns seq_id, PC, axis_score, n_sites_used.
    """
    pc_map = dict(zip(pc_df["seq_id"].astype(str).tolist(), pc_df[pc_col].astype(float).tolist()))
    msa_map = {sid: seq for sid, seq in zip(msa_ids, msa_seqs)}

    eff = site_effects_df.copy()
    if eff.empty:
        return pd.DataFrame(columns=["seq_id", pc_col, "axis_score", "n_sites_used"])

    # build lookup: (col,state)->(weight,effect). weight is per-site (constant across states), but stored per row
    # if multiple rows share same col with different weights (shouldn't), take first weight.
    weight_by_col = eff.groupby("aln_col_1based")["weight"].first().to_dict()
    effect_lookup = {}
    for _, r in eff.iterrows():
        effect_lookup[(int(r["aln_col_1based"]), str(r["state"]))] = float(r["effect_centered_mean_pc"])

    out_rows = []
    for sid in target_ids:
        if sid not in msa_map or sid not in pc_map:
            continue
        seq = msa_map[sid]
        score = 0.0
        used = 0

        for c1, w in weight_by_col.items():
            j = int(c1) - 1
            if j < 0 or j >= L:
                continue

            aa = seq[j] if j < len(seq) else gap_char
            aa = aa if aa in AA_TO_I else gap_char

            if channel == "INDEL":
                st = "GAP" if aa == gap_char else "NONGAP"
            else:
                # AA channel
                if gap_mode == "aa_only" and aa == gap_char:
                    continue
                st = aa

            key = (int(c1), st)
            effv = effect_lookup.get(key, None)
            if effv is None:
                # if AA: allow unseen -> 0; if INDEL: should be present
                continue
            score += float(w) * float(effv)
            used += 1

        out_rows.append({
            "seq_id": sid,
            pc_col: float(pc_map[sid]),
            "axis_score": float(score),
            "n_sites_used": int(used),
        })

    return pd.DataFrame(out_rows)


def axis_score_stats(score_df: pd.DataFrame, pc_col: str) -> dict:
    from scipy.stats import spearmanr, pearsonr
    x = score_df[pc_col].astype(float).values
    y = score_df["axis_score"].astype(float).values
    if len(x) < 10:
        return {"n": int(len(x)), "pearson_r": np.nan, "pearson_p": np.nan,
                "spearman_rho": np.nan, "spearman_p": np.nan}
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    return {"n": int(len(x)), "pearson_r": float(pr), "pearson_p": float(pp),
            "spearman_rho": float(sr), "spearman_p": float(sp)}


# ---------------- Bootstrap stability ----------------

def hotspot_bootstrap_stability(msa_ids: List[str], msa_seqs: List[str], L: int,
                                pc_df: pd.DataFrame, pc_col: str,
                                base_ids: List[str],
                                channel: str,
                                gap_mode: str,
                                gap_char: str,
                                drop_gappy_cols_gt: Optional[float],
                                min_state_count: int,
                                min_group_size: int,
                                permutations: int,
                                bootstrap_n: int,
                                bootstrap_sample_frac: float,
                                bootstrap_rank: str,
                                stable_top_m: int,
                                rng: np.random.Generator,
                                stratify_labels: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    For each bootstrap replicate:
      - resample sequences with replacement
      - compute hotspots and take topM (rank by bootstrap_rank)
      - count frequency of each column in topM
    Returns a df: aln_col_1based, freq_in_topM (0..1), count_in_topM, bootstrap_n
    """
    freq = Counter()

    base_ids = list(base_ids)
    base_set = set(base_ids)

    # speed: permutations during bootstrap can be extremely expensive. You can set hotspot_permutations=0 for stability.
    for b in range(bootstrap_n):
        samp_ids = bootstrap_resample_ids(base_ids, rng=rng, sample_frac=bootstrap_sample_frac, stratify_labels=stratify_labels)
        samp_set = set(samp_ids)

        hot = continuous_hotspots(
            msa_ids=msa_ids, msa_seqs=msa_seqs, L=L,
            pc_df=pc_df[["seq_id", pc_col]].copy(),
            pc_col=pc_col,
            seq_ids_subset=samp_set,
            channel=channel,
            gap_mode=gap_mode,
            gap_char=gap_char,
            drop_gappy_cols_gt=drop_gappy_cols_gt,
            min_state_count=min_state_count,
            min_group_size=min_group_size,
            permutations=permutations,
            rng=rng,
            verbose_every=0
        )

        if hot.empty:
            continue

        if bootstrap_rank == "q_bh":
            hot = hot.sort_values(["q_bh", "eta2"], ascending=[True, False])
        else:
            hot = hot.sort_values(["eta2", "q_bh"], ascending=[False, True])

        top = hot.head(int(stable_top_m))
        for c1 in top["aln_col_1based"].astype(int).tolist():
            freq[int(c1)] += 1

        if (b + 1) % max(1, bootstrap_n // 10) == 0:
            print(f"[bootstrap {pc_col} {channel}] {b+1}/{bootstrap_n}")

    rows = []
    for c1, ct in freq.items():
        rows.append({
            "pc": pc_col,
            "channel": channel,
            "aln_col_1based": int(c1),
            "count_in_topM": int(ct),
            "bootstrap_n": int(bootstrap_n),
            "freq_in_topM": float(ct / bootstrap_n) if bootstrap_n > 0 else np.nan
        })
    out = pd.DataFrame(rows).sort_values("freq_in_topM", ascending=False).reset_index(drop=True)
    return out


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dist_csv", required=True)
    ap.add_argument("--emb_npy", required=True)
    ap.add_argument("--ids_txt", required=True)
    ap.add_argument("--msa_fasta", required=True)
    ap.add_argument("--metadata_csv", required=True)
    ap.add_argument("--out_prefix", required=True)

    # HDBSCAN
    ap.add_argument("--min_cluster_size", type=int, default=40)
    ap.add_argument("--min_samples", type=int, default=5)
    ap.add_argument("--cluster_selection_epsilon", type=float, default=0.0)

    # UMAP
    ap.add_argument("--umap_neighbors", type=int, default=30)
    ap.add_argument("--umap_min_dist", type=float, default=0.1)
    ap.add_argument("--random_state", type=int, default=1)

    # PCA
    ap.add_argument("--pca_components", type=int, default=10)

    # Plotting
    ap.add_argument("--plot_format", default="png", choices=["png", "pdf", "both", "none"])
    ap.add_argument("--point_size", type=float, default=6.0)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--label_centroids", action="store_true")

    # Trait collapsing / selection
    ap.add_argument("--lifestyle_min_count", type=int, default=50)
    ap.add_argument("--lifestyle_use_macro_guilds", action="store_true")
    ap.add_argument("--include_growth_form", action="store_true")
    ap.add_argument("--include_localization", action="store_true")
    ap.add_argument("--enzyme_name", default=None)

    # Trait association (eta^2)
    ap.add_argument("--eta2_permutations", type=int, default=0)
    ap.add_argument("--eta2_drop_na", action="store_true")
    ap.add_argument("--eta2_min_group_size", type=int, default=2)

    # Hotspot controls
    ap.add_argument("--gap_mode", default="exclude", choices=["exclude", "indel", "aa_only"],
                    help="How to handle gaps for AA hotspots.")
    ap.add_argument("--indel_channel", action="store_true",
                    help="Also compute an explicit INDEL channel (GAP vs NONGAP).")
    ap.add_argument("--hotspot_gap_char", default="-")
    ap.add_argument("--hotspot_drop_gappy_cols_gt", type=float, default=0.5,
                    help="Drop columns with gap_frac > this. Set >1 to disable.")
    ap.add_argument("--hotspot_min_state_count", type=int, default=5)
    ap.add_argument("--hotspot_min_group_size", type=int, default=5)
    ap.add_argument("--hotspot_permutations", type=int, default=0)

    # OOS evaluation
    ap.add_argument("--hotspot_oos_mode", default="split", choices=["split", "kfold", "none"],
                    help="Discover hotspots on train, evaluate axis score on held-out data.")
    ap.add_argument("--hotspot_test_frac", type=float, default=0.30)
    ap.add_argument("--hotspot_kfolds", type=int, default=5)
    ap.add_argument("--stratify_by_phylum", action="store_true",
                    help="Stratify train/test and bootstrap by Phylum if available.")

    # Axis score
    ap.add_argument("--hotspot_top_k", type=int, default=25)
    ap.add_argument("--hotspot_weight_col", default="eta2", choices=["eta2", "minuslog10q", "minuslog10p"])

    # Bootstrap stability
    ap.add_argument("--bootstrap_n", type=int, default=200,
                    help="Bootstrap replicates for hotspot stability (0 disables).")
    ap.add_argument("--bootstrap_sample_frac", type=float, default=1.0)
    ap.add_argument("--bootstrap_rank", default="eta2", choices=["eta2", "q_bh"],
                    help="Ranking used to decide topM in each bootstrap replicate.")
    ap.add_argument("--stable_top_m", type=int, default=50,
                    help="In each bootstrap replicate, count presence in topM (default 50).")
    ap.add_argument("--stable_min_freq", type=float, default=0.5,
                    help="Stable hotspot threshold: freq_in_topM >= this (default 0.5).")

    args = ap.parse_args()
    rng = np.random.default_rng(args.random_state)

    # ---------------- Load distance matrix ----------------
    dist_df = load_square_csv(args.dist_csv)
    dist_ids = dist_df.index.tolist()
    D = dist_df.values.astype(float)

    # ---------------- HDBSCAN ----------------
    import hdbscan
    clusterer = hdbscan.HDBSCAN(
        metric="precomputed",
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        cluster_selection_epsilon=args.cluster_selection_epsilon
    )
    labels = clusterer.fit_predict(D)
    probs = getattr(clusterer, "probabilities_", np.full(len(dist_ids), np.nan))
    outlier = getattr(clusterer, "outlier_scores_", np.full(len(dist_ids), np.nan))

    pd.DataFrame({
        "seq_id": dist_ids,
        "cluster": labels,
        "probability": probs,
        "outlier_score": outlier
    }).to_csv(f"{args.out_prefix}.clusters.csv", index=False)
    print(f"Wrote: {args.out_prefix}.clusters.csv")

    # ---------------- UMAP ----------------
    import umap
    reducer = umap.UMAP(
        metric="precomputed",
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        random_state=args.random_state
    )
    umap_xy = reducer.fit_transform(D)

    # ---------------- Load embeddings and align to dist IDs ----------------
    emb_ids = load_ids_txt(args.ids_txt)
    emb = load_embeddings_npy(args.emb_npy)
    if emb.shape[0] != len(emb_ids):
        raise ValueError(f"emb_npy n={emb.shape[0]} but ids_txt n={len(emb_ids)}")

    emb_idx = {sid: i for i, sid in enumerate(emb_ids)}
    missing = [sid for sid in dist_ids if sid not in emb_idx]
    if missing:
        raise ValueError(f"{len(missing)} distance-matrix IDs not found in embeddings ids_txt. Example: {missing[:5]}")
    emb_reordered = np.vstack([emb[emb_idx[sid], :] for sid in dist_ids])

    # ---------------- PCA ----------------
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(args.pca_components, emb_reordered.shape[1]))
    pcs = pca.fit_transform(emb_reordered)
    save_pca_variance(pca, args.out_prefix)

    # ---------------- Metadata ----------------
    meta = read_metadata_csv(args.metadata_csv, id_col="Accession").set_index("seq_id")

    # Master df
    df = pd.DataFrame({
        "seq_id": dist_ids,
        "cluster": labels,
        "umap1": umap_xy[:, 0],
        "umap2": umap_xy[:, 1],
        "PC1": pcs[:, 0],
        "PC2": pcs[:, 1],
    })

    # Attach traits if present
    for col in ["Phylum", "primary_lifestyle", "Growth_form_template"]:
        if col in meta.columns:
            df[col] = [meta.loc[sid, col] if sid in meta.index else np.nan for sid in dist_ids]
            df[col] = df[col].apply(_clean_cat)

    # Lifestyle reduction
    if "primary_lifestyle" in df.columns:
        if args.lifestyle_use_macro_guilds:
            df["Lifestyle_reduced"] = lifestyle_macro_guild(df["primary_lifestyle"])
        else:
            df["Lifestyle_reduced"] = collapse_rare_categories(df["primary_lifestyle"],
                                                               min_count=args.lifestyle_min_count,
                                                               other_label="Other")

    # Growth form reduction
    if args.include_growth_form and "Growth_form_template" in df.columns:
        df["GrowthForm_reduced"] = collapse_growth_form(df["Growth_form_template"])

    # Enzyme-specific localization
    if args.include_localization:
        enzyme = args.enzyme_name
        if enzyme is None:
            enzyme = str(args.out_prefix).split(".")[0].split("_")[0]
        candidate = f"{enzyme}_SEC"
        if candidate in meta.columns:
            df["Localization"] = [meta.loc[sid, candidate] if sid in meta.index else np.nan for sid in dist_ids]
            df["Localization"] = df["Localization"].apply(_clean_cat)
        else:
            print(f"[warn] --include_localization set but column '{candidate}' not found in metadata.")

    df.to_csv(f"{args.out_prefix}.umap_pca.csv", index=False)
    print(f"Wrote: {args.out_prefix}.umap_pca.csv")

    # ---------------- Plots ----------------
    if args.plot_format != "none":
        plot_2d_by_cluster(
            df, "umap1", "umap2",
            out_prefix=f"{args.out_prefix}.umap_clusters",
            fmt=args.plot_format,
            title="UMAP colored by HDBSCAN cluster",
            point_size=args.point_size, alpha=args.alpha, label_centroids=args.label_centroids
        )
        for cat, name in [("Phylum", "Phylum"),
                          ("Lifestyle_reduced", "Lifestyle"),
                          ("GrowthForm_reduced", "GrowthForm"),
                          ("Localization", "Localization")]:
            if cat in df.columns:
                plot_2d_by_category(
                    df, "umap1", "umap2", cat,
                    out_prefix=f"{args.out_prefix}.umap_{name}",
                    fmt=args.plot_format,
                    title=f"UMAP colored by {name}",
                    point_size=args.point_size, alpha=args.alpha
                )

        plot_2d_by_cluster(
            df, "PC1", "PC2",
            out_prefix=f"{args.out_prefix}.pca_clusters",
            fmt=args.plot_format,
            title="PCA (PC1 vs PC2) colored by HDBSCAN cluster",
            point_size=args.point_size, alpha=args.alpha, label_centroids=args.label_centroids
        )
        if "Phylum" in df.columns:
            plot_2d_by_category(
                df, "PC1", "PC2", "Phylum",
                out_prefix=f"{args.out_prefix}.pca_Phylum",
                fmt=args.plot_format,
                title="PCA (PC1 vs PC2) colored by Phylum",
                point_size=args.point_size, alpha=args.alpha
            )

    # ---------------- Trait association ----------------
    trait_cols = []
    for c in ["Phylum", "Lifestyle_reduced", "GrowthForm_reduced", "Localization"]:
        if c in df.columns:
            trait_cols.append(c)

    eta_records = []
    for trait in trait_cols:
        for comp in ["PC1", "PC2"]:
            rec = eta_squared_pc_category(
                df=df,
                pc_col=comp,
                cat_col=trait,
                permutations=args.eta2_permutations,
                random_state=args.random_state,
                drop_na=args.eta2_drop_na,
                min_group_size=args.eta2_min_group_size
            )
            eta_records.append(rec)

    eta_df = pd.DataFrame(eta_records)
    eta_df.to_csv(f"{args.out_prefix}.pc_trait_eta2.csv", index=False)
    with open(f"{args.out_prefix}.pc_trait_eta2.txt", "w") as f:
        for _, r in eta_df.iterrows():
            f.write(
                f"{r['pc']}\ttrait={r['trait']}\tn_used={int(r['n_used'])}\t"
                f"n_levels={int(r['n_levels'])}\teta2={float(r['eta2']):.4f}\t"
                f"smallest_group={int(r['smallest_group'])}"
            )
            if args.eta2_permutations and args.eta2_permutations > 0:
                f.write(f"\teta2_p={float(r['eta2_p']):.4g}")
            f.write("\n")
    print(f"Wrote: {args.out_prefix}.pc_trait_eta2.csv")
    print(f"Wrote: {args.out_prefix}.pc_trait_eta2.txt")

    # ---------------- Read MSA ----------------
    msa_ids, msa_seqs, L = read_msa_fasta(args.msa_fasta)
    df_pc = df[["seq_id", "PC1", "PC2"]].copy()

    # Interpret gappy drop flag
    drop_gappy = args.hotspot_drop_gappy_cols_gt
    if drop_gappy is not None and drop_gappy > 1.0:
        drop_gappy = None

    # Choose stratification labels
    strat_labels = None
    if args.stratify_by_phylum and "Phylum" in df.columns:
        strat_labels = df.set_index("seq_id")["Phylum"]

    # Channels to run
    channels = ["AA"]
    if args.indel_channel:
        channels.append("INDEL")

    # OOS split config
    all_ids = df["seq_id"].astype(str).tolist()
    all_id_set = set(all_ids)

    summary_rows = []

    # If kfold: define folds on IDs once
    folds = []
    if args.hotspot_oos_mode == "kfold":
        k = int(args.hotspot_kfolds)
        if k < 2:
            raise ValueError("--hotspot_kfolds must be >=2 for kfold mode.")
        # stratified-ish: group by label and assign round-robin
        if strat_labels is None:
            perm = rng.permutation(len(all_ids))
            all_perm = [all_ids[i] for i in perm]
            folds = [set() for _ in range(k)]
            for i, sid in enumerate(all_perm):
                folds[i % k].add(sid)
        else:
            lab_series = strat_labels.reindex(all_ids)
            lab_to_ids = defaultdict(list)
            for sid, lab in lab_series.items():
                key = "__NA__" if pd.isna(lab) else str(lab)
                lab_to_ids[key].append(sid)
            folds = [set() for _ in range(k)]
            for lab, ids_lab in lab_to_ids.items():
                ids_lab = list(ids_lab)
                perm = rng.permutation(len(ids_lab))
                for i, idx in enumerate(perm):
                    folds[i % k].add(ids_lab[idx])

    for pc_col in ["PC1", "PC2"]:
        for channel in channels:
            print(f"\n=== {pc_col} | channel={channel} | gap_mode={args.gap_mode} ===")

            # Decide OOS mode
            if args.hotspot_oos_mode == "none":
                train_ids = set(all_ids)
                test_ids = set()
                splits = [("nosplit", train_ids, test_ids)]
            elif args.hotspot_oos_mode == "split":
                train_list, test_list = train_test_split_ids(
                    all_ids, test_frac=float(args.hotspot_test_frac), rng=rng, stratify_labels=strat_labels
                )
                train_ids, test_ids = set(train_list), set(test_list)
                splits = [("split", train_ids, test_ids)]
            else:
                # kfold: each fold is test; rest is train
                splits = []
                for i, test in enumerate(folds):
                    train = all_id_set - set(test)
                    splits.append((f"fold{i+1}", set(train), set(test)))

            # Aggregate CV results if kfold
            fold_stats = []

            # Also compute bootstrap stability on the FULL ID set (or train set in split mode)
            # We'll do stability on the "base set" used for discovery (train in split; all in none; train per fold is too much).
            base_for_bootstrap = all_ids
            if args.hotspot_oos_mode == "split":
                base_for_bootstrap = list(train_ids)

            # ---- Bootstrap stability (optional) ----
            boot_df = pd.DataFrame()
            if args.bootstrap_n and args.bootstrap_n > 0:
                print(f"Bootstrap stability: n={args.bootstrap_n}, topM={args.stable_top_m}")
                boot_df = hotspot_bootstrap_stability(
                    msa_ids=msa_ids, msa_seqs=msa_seqs, L=L,
                    pc_df=df_pc, pc_col=pc_col,
                    base_ids=base_for_bootstrap,
                    channel=channel,
                    gap_mode=args.gap_mode,
                    gap_char=args.hotspot_gap_char,
                    drop_gappy_cols_gt=drop_gappy,
                    min_state_count=args.hotspot_min_state_count,
                    min_group_size=args.hotspot_min_group_size,
                    permutations=0,  # recommend 0 for speed; stability doesn't need perms
                    bootstrap_n=int(args.bootstrap_n),
                    bootstrap_sample_frac=float(args.bootstrap_sample_frac),
                    bootstrap_rank=str(args.bootstrap_rank),
                    stable_top_m=int(args.stable_top_m),
                    rng=rng,
                    stratify_labels=strat_labels if args.stratify_by_phylum else None
                )
                boot_out = f"{args.out_prefix}.{pc_col}.hotspots_{channel}.bootstrap_freq.tsv"
                boot_df.to_csv(boot_out, sep="\t", index=False)
                print(f"Wrote: {boot_out}")

            # Run each split/fold
            for split_name, tr_ids, te_ids in splits:
                # ---- Discover hotspots on TRAIN ----
                hot_train = continuous_hotspots(
                    msa_ids=msa_ids, msa_seqs=msa_seqs, L=L,
                    pc_df=df_pc[["seq_id", pc_col]].copy(),
                    pc_col=pc_col,
                    seq_ids_subset=tr_ids,
                    channel=channel,
                    gap_mode=args.gap_mode,
                    gap_char=args.hotspot_gap_char,
                    drop_gappy_cols_gt=drop_gappy,
                    min_state_count=args.hotspot_min_state_count,
                    min_group_size=args.hotspot_min_group_size,
                    permutations=int(args.hotspot_permutations),
                    rng=rng,
                    verbose_every=200 if split_name in ("split", "nosplit") else 0
                )

                out_hot_all = f"{args.out_prefix}.{pc_col}.hotspots_{channel}.{split_name}.train.tsv"
                hot_train.to_csv(out_hot_all, sep="\t", index=False)

                topk = int(args.hotspot_top_k)
                hot_top = hot_train.head(topk).copy()
                out_hot_top = f"{args.out_prefix}.{pc_col}.hotspots_{channel}.{split_name}.train_top{topk}.tsv"
                hot_top.to_csv(out_hot_top, sep="\t", index=False)

                # ---- Stable hotspot subset (if bootstrap exists) ----
                stable_df = pd.DataFrame()
                if not boot_df.empty:
                    stable = boot_df[boot_df["freq_in_topM"] >= float(args.stable_min_freq)].copy()
                    stable = stable.sort_values("freq_in_topM", ascending=False).reset_index(drop=True)

                    # attach train hotspot stats for these cols (if present)
                    stable_cols = set(stable["aln_col_1based"].astype(int).tolist())
                    hot_sub = hot_train[hot_train["aln_col_1based"].astype(int).isin(stable_cols)].copy()
                    hot_sub = hot_sub.merge(stable[["aln_col_1based", "freq_in_topM"]], on="aln_col_1based", how="left")
                    hot_sub = hot_sub.sort_values(["freq_in_topM", "eta2"], ascending=[False, False]).reset_index(drop=True)
                    stable_df = hot_sub

                    stable_out = f"{args.out_prefix}.{pc_col}.hotspots_{channel}.stable.tsv"
                    # write once (independent of split) to keep clean; but ensure we don't overwrite across folds:
                    if split_name in ("split", "nosplit", "fold1"):
                        stable_df.to_csv(stable_out, sep="\t", index=False)
                        print(f"Wrote: {stable_out}")

                # ---- Fit site effects on TRAIN and score TRAIN/TEST ----
                wcol = args.hotspot_weight_col
                if wcol not in hot_top.columns:
                    raise ValueError(f"Requested --hotspot_weight_col {wcol} not in hotspot table columns.")

                site_eff = fit_site_effects_from_hotspots(
                    msa_ids=msa_ids, msa_seqs=msa_seqs, L=L,
                    pc_df=df_pc[["seq_id", pc_col]].copy(),
                    pc_col=pc_col,
                    train_ids=tr_ids,
                    hotspots_df=hot_top,
                    top_k=topk,
                    weight_col=wcol,
                    channel=channel,
                    gap_mode=args.gap_mode,
                    gap_char=args.hotspot_gap_char,
                    min_state_count=args.hotspot_min_state_count,
                    min_group_size=args.hotspot_min_group_size
                )

                site_eff_out = f"{args.out_prefix}.{pc_col}.axis_score_{channel}.{split_name}.site_effects_train.tsv"
                site_eff.to_csv(site_eff_out, sep="\t", index=False)

                # Score train set
                train_scores = score_sequences_with_site_effects(
                    msa_ids=msa_ids, msa_seqs=msa_seqs, L=L,
                    pc_df=df_pc[["seq_id", pc_col]].copy(),
                    pc_col=pc_col,
                    target_ids=tr_ids,
                    site_effects_df=site_eff,
                    channel=channel,
                    gap_mode=args.gap_mode,
                    gap_char=args.hotspot_gap_char
                )
                train_out = f"{args.out_prefix}.{pc_col}.axis_score_{channel}.{split_name}.train.csv"
                train_scores.to_csv(train_out, index=False)

                tr_stats = axis_score_stats(train_scores, pc_col=pc_col)

                # Score test set (if exists)
                te_stats = {"n": 0, "pearson_r": np.nan, "pearson_p": np.nan,
                            "spearman_rho": np.nan, "spearman_p": np.nan}
                if len(te_ids) > 0:
                    test_scores = score_sequences_with_site_effects(
                        msa_ids=msa_ids, msa_seqs=msa_seqs, L=L,
                        pc_df=df_pc[["seq_id", pc_col]].copy(),
                        pc_col=pc_col,
                        target_ids=te_ids,
                        site_effects_df=site_eff,
                        channel=channel,
                        gap_mode=args.gap_mode,
                        gap_char=args.hotspot_gap_char
                    )
                    test_out = f"{args.out_prefix}.{pc_col}.axis_score_{channel}.{split_name}.test.csv"
                    test_scores.to_csv(test_out, index=False)
                    te_stats = axis_score_stats(test_scores, pc_col=pc_col)

                    if args.plot_format != "none" and split_name in ("split", "nosplit"):
                        annot = (f"{split_name} | topK={topk}, w={wcol}\n"
                                 f"TRAIN r={tr_stats['pearson_r']:.3f}, ρ={tr_stats['spearman_rho']:.3f}\n"
                                 f"TEST  r={te_stats['pearson_r']:.3f}, ρ={te_stats['spearman_rho']:.3f}")
                        plot_axis_score_scatter(
                            score_df=test_scores,
                            pc_col=pc_col,
                            out_prefix=f"{args.out_prefix}.{pc_col}.axis_score_{channel}.{split_name}.TEST_scatter",
                            fmt=args.plot_format,
                            title=f"{pc_col}: axis score ({channel}) on TEST",
                            point_size=max(4.0, args.point_size * 0.8),
                            alpha=min(0.65, args.alpha),
                            annotate=annot
                        )

                row = {
                    "pc": pc_col,
                    "channel": channel,
                    "split": split_name,
                    "gap_mode": args.gap_mode,
                    "top_k": topk,
                    "weight": wcol,
                    "n_train": int(tr_stats["n"]),
                    "train_pearson_r": tr_stats["pearson_r"],
                    "train_pearson_p": tr_stats["pearson_p"],
                    "train_spearman_rho": tr_stats["spearman_rho"],
                    "train_spearman_p": tr_stats["spearman_p"],
                    "n_test": int(te_stats["n"]),
                    "test_pearson_r": te_stats["pearson_r"],
                    "test_pearson_p": te_stats["pearson_p"],
                    "test_spearman_rho": te_stats["spearman_rho"],
                    "test_spearman_p": te_stats["spearman_p"],
                }
                summary_rows.append(row)

                if args.hotspot_oos_mode == "kfold":
                    fold_stats.append(row)

            # If kfold, write a fold summary table for this PC/channel
            if args.hotspot_oos_mode == "kfold":
                fold_df = pd.DataFrame(fold_stats)
                fold_out = f"{args.out_prefix}.{pc_col}.axis_score_{channel}.kfold_summary.csv"
                fold_df.to_csv(fold_out, index=False)
                print(f"Wrote: {fold_out}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{args.out_prefix}.axis_score_summary.csv", index=False)

    with open(f"{args.out_prefix}.axis_score_summary.txt", "w") as f:
        for _, r in summary_df.iterrows():
            f.write(
                f"{r['pc']}\tchannel={r['channel']}\tsplit={r['split']}\tgap_mode={r['gap_mode']}\t"
                f"topK={int(r['top_k'])}\tw={r['weight']}\t"
                f"train_r={float(r['train_pearson_r']):.4f}\ttrain_rho={float(r['train_spearman_rho']):.4f}\t"
                f"test_r={float(r['test_pearson_r']):.4f}\ttest_rho={float(r['test_spearman_rho']):.4f}\n"
            )

    print(f"Wrote: {args.out_prefix}.axis_score_summary.csv")
    print(f"Wrote: {args.out_prefix}.axis_score_summary.txt")


if __name__ == "__main__":
    main()
