#!/usr/bin/env python3
"""
05_annotate_correlations_with_consensus.py

Annotate correlation results with positions in an aligned consensus sequence.
Similar to hotspots annotation but works with correlation coefficients (eta2).

Assumptions:
- The consensus sequence is aligned and includes gaps, and has the same length as the MSA.
- Correlation TSV has a column named 'aln_col_1based' and 'eta2' column.

Outputs:
- annotated correlations TSV with added columns:
  - consensus_id
  - cons_char                : consensus residue/gap at that alignment column
  - cons_res_index_1based    : ungapped residue index in consensus (NA if cons_char is '-')
  - cons_window              : small consensus context window around position (optional)
- optional residue-level track TSV:
  - residue_index            : 1-based residue index in ungapped consensus
  - track_score              : aggregated correlation score at that residue
"""

import argparse
import pandas as pd
from Bio import SeqIO
import numpy as np
from typing import Dict, Optional, Tuple


def load_consensus_aligned(consensus_fasta: str, consensus_id: Optional[str] = None) -> Tuple[str, str]:
    recs = list(SeqIO.parse(consensus_fasta, "fasta"))
    if not recs:
        raise ValueError("No records found in consensus_fasta.")

    if consensus_id is None:
        if len(recs) != 1:
            raise ValueError("consensus_fasta has multiple sequences; provide --consensus_id.")
        seq = str(recs[0].seq).upper()
        cid = recs[0].id
        return cid, seq

    for r in recs:
        if r.id == consensus_id:
            return r.id, str(r.seq).upper()

    raise ValueError(f"Consensus ID '{consensus_id}' not found in {consensus_fasta}")


def build_alncol_to_cons_res_index(cons_aln_seq: str) -> Tuple[Dict[int, Optional[int]], int]:
    """
    Map alignment columns (1-based) -> consensus ungapped residue index (1-based) or None if gap.
    Returns: (mapping, n_residues_ungapped)
    """
    mapping: Dict[int, Optional[int]] = {}
    res_i = 0
    for col_i, ch in enumerate(cons_aln_seq, start=1):
        if ch != "-":
            res_i += 1
            mapping[col_i] = res_i
        else:
            mapping[col_i] = None
    return mapping, res_i


def consensus_window(cons_aln_seq: str, col_1based: int, half_window: int = 10) -> str:
    i = col_1based - 1
    lo = max(0, i - half_window)
    hi = min(len(cons_aln_seq), i + half_window + 1)
    return cons_aln_seq[lo:hi]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--correlations_tsv", required=True, help="Correlations TSV with aln_col_1based and eta2")
    ap.add_argument("--consensus_fasta", required=True, help="FASTA containing aligned consensus")
    ap.add_argument("--consensus_id", default=None, help="If FASTA has multiple seqs, pick this ID")
    ap.add_argument("--out_tsv", required=True, help="Annotated correlations TSV output")

    # Rank/selection controls
    ap.add_argument(
        "--top_n",
        type=int,
        default=25,
        help="Keep only top N positions after sorting by eta2."
    )
    ap.add_argument(
        "--drop_consensus_gaps",
        action="store_true",
        help="Drop rows where the consensus has a gap at that alignment column."
    )

    # Optional consensus context
    ap.add_argument("--add_window", action="store_true", help="Add consensus window context")
    ap.add_argument("--window_half", type=int, default=10)

    # Optional residue-level track output
    ap.add_argument(
        "--out_track_tsv",
        default=None,
        help="Optional output TSV with per-residue track_score for structure mapping."
    )
    ap.add_argument(
        "--agg",
        default="max",
        choices=["max", "sum"],
        help="How to aggregate multiple positions mapping to the same residue."
    )

    args = ap.parse_args()

    cid, cons_aln = load_consensus_aligned(args.consensus_fasta, args.consensus_id)
    col_to_res, n_res = build_alncol_to_cons_res_index(cons_aln)

    corr = pd.read_csv(args.correlations_tsv, sep="\t")
    if "aln_col_1based" not in corr.columns:
        raise ValueError("correlations_tsv must contain column 'aln_col_1based'")
    if "eta2" not in corr.columns:
        raise ValueError("correlations_tsv must contain column 'eta2'")

    # Sort by eta2 (descending)
    corr = corr.sort_values("eta2", ascending=False).copy()

    # Pre-annotate with consensus chars to identify gaps BEFORE filtering
    L = len(cons_aln)
    cons_chars_temp = []
    for c in corr["aln_col_1based"].astype(int).tolist():
        if c < 1 or c > L:
            cons_chars_temp.append("-")
        else:
            cons_chars_temp.append(cons_aln[c - 1])
    corr["_cons_char_temp"] = cons_chars_temp

    # Filter out gap rows and take top N ungapped
    if args.drop_consensus_gaps:
        corr = corr[corr["_cons_char_temp"] != "-"].copy()
    
    corr = corr.head(args.top_n).copy()
    corr = corr.drop(columns=["_cons_char_temp"])

    # Now annotate remaining rows with full consensus mapping
    cons_chars = []
    cons_res_idx = []
    cons_windows = []

    for c in corr["aln_col_1based"].astype(int).tolist():
        if c < 1 or c > L:
            cons_chars.append("NA")
            cons_res_idx.append(pd.NA)
            if args.add_window:
                cons_windows.append("NA")
            continue

        ch = cons_aln[c - 1]
        cons_chars.append(ch)
        cons_res_idx.append(col_to_res[c] if ch != "-" else pd.NA)
        if args.add_window:
            cons_windows.append(consensus_window(cons_aln, c, args.window_half))

    corr.insert(len(corr.columns), "consensus_id", cid)
    corr.insert(len(corr.columns), "cons_char", cons_chars)
    corr.insert(len(corr.columns), "cons_res_index_1based", cons_res_idx)
    if args.add_window:
        corr.insert(len(corr.columns), "cons_window", cons_windows)

    corr.to_csv(args.out_tsv, sep="\t", index=False)
    print(f"Wrote annotated correlations: {args.out_tsv}")

    # Optional: build residue-level track
    if args.out_track_tsv:
        scores = np.zeros(n_res + 1, dtype=float)  # 1-based

        for _, row in corr.iterrows():
            res = row["cons_res_index_1based"]
            if pd.isna(res):
                continue
            res = int(res)
            val = float(row["eta2"])
            if args.agg == "max":
                scores[res] = max(scores[res], val)
            else:
                scores[res] += val

        track_df = pd.DataFrame({
            "residue_index": list(range(1, n_res + 1)),
            "track_score": [scores[i] for i in range(1, n_res + 1)]
        })
        track_df.to_csv(args.out_track_tsv, sep="\t", index=False)
        print(f"Wrote residue track: {args.out_track_tsv}")


if __name__ == "__main__":
    main()