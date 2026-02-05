#!/usr/bin/env python3
"""
calculate_jsd_alignment.py

Calculate per-position Jensen-Shannon divergence (JSD) for a multiple sequence alignment (FASTA format).

Usage:
    python calculate_jsd_alignment.py alignment.fasta [--output jsd_per_position.txt]

Output:
    Tab-separated file with position and JSD value.
"""
import sys
import argparse
from collections import Counter, defaultdict
import numpy as np
from math import log2
import csv

def parse_fasta(filename):
    seqs = []
    headers = []
    with open(filename) as f:
        seq = ''
        header = None
        for line in f:
import os
import sys
import argparse
                if seq:
                    seqs.append(seq)
                    headers.append(header)
                    seq = ''
                header = line[1:].strip()
            else:
                seq += line.strip()
        if seq:
            seqs.append(seq)
            headers.append(header)
    return headers, seqs

def shannon_entropy(p):
    return -sum(x * log2(x) for x in p if x > 0)

def jsd(p, q):
    m = 0.5 * (p + q)
    return shannon_entropy(m) - 0.5 * (shannon_entropy(p) + shannon_entropy(q))

def get_freqs(column, alphabet):
    counts = Counter(column)
    total = sum(counts[a] for a in alphabet)
    return np.array([counts[a] / total if total > 0 else 0 for a in alphabet])

def consensus_for_indices(indices, seqs, alphabet):
def parse_args():
    parser = argparse.ArgumentParser(description='Calculate JSD for alignments with flexible paths.')
    parser.add_argument('--alignment', required=True, help='Alignment file path')
    parser.add_argument('--output', required=True, help='Output file path')
    # Add more arguments as needed
    return parser.parse_args()

    aln_len = len(seqs[0])
    consensus = []
    for i in range(aln_len):
        column = [seqs[idx][i] for idx in indices]
        if not column:
            consensus.append('-')
            continue
        counts = Counter(column)
        max_count = max(counts.values())
        max_residues = [res for res, cnt in counts.items() if cnt == max_count]
        consensus.append(sorted(max_residues)[0])
    return ''.join(consensus)

def consensus_of_consensuses(phylum_to_indices, seqs, alphabet, phyla):
    aln_len = len(seqs[0])
    # Get per-phylum consensuses
    phylum_consensuses = []
    for phylum in phyla:
        indices = phylum_to_indices[phylum]
        if not indices:
            continue
        consensus = []
        for i in range(aln_len):
            column = [seqs[idx][i] for idx in indices]
            if not column:
                consensus.append('-')
                continue
            counts = Counter(column)
            max_count = max(counts.values())
            max_residues = [res for res, cnt in counts.items() if cnt == max_count]
            consensus.append(sorted(max_residues)[0])
        phylum_consensuses.append(consensus)
    # Now, consensus of consensuses
    consensus = []
    for i in range(aln_len):
        column = [cons[i] for cons in phylum_consensuses]
        if not column:
            consensus.append('-')
            continue
        counts = Counter(column)
        max_count = max(counts.values())
        max_residues = [res for res, cnt in counts.items() if cnt == max_count]
        consensus.append(sorted(max_residues)[0])
    return ''.join(consensus)


def load_accession_to_phylum(csv_path):
    acc2phylum = {}
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            acc = row['Accession'].strip()
            phylum = row['Phylum'].strip()
            acc2phylum[acc] = phylum
    return acc2phylum

def extract_accession(header):
    # Accession is assumed to be the first word in the header
    return header.split()[0]

def main():
    parser = argparse.ArgumentParser(description="Calculate per-position conservation and JSD for an alignment, segmented by phylum.")
    parser.add_argument('alignment', help='Input alignment in FASTA format')
    parser.add_argument('--output', default=None, help='Output file (default: stdout)')
    parser.add_argument('--alphabet', default=None, help='Alphabet to use (default: inferred from alignment)')
    parser.add_argument('--accession_csv', required=True, help='CSV file mapping accession to phylum')
    parser.add_argument('--consensus_output', default=None, help='Output file for consensus sequences per phylum (default: consensus_per_phylum.fasta)')
    parser.add_argument('--alpha', type=float, default=1.0, help='Coverage penalty exponent alpha (default: 1.0)')
    parser.add_argument('--output_dir', default=None, help='Directory to write all output files')
    args = parser.parse_args()

    headers, seqs = parse_fasta(args.alignment)
    if not seqs:
    args = parse_args()
    alignment_path = args.alignment
    output_path = args.output
    # ...existing code using alignment_path/output_path instead of hardcoded paths...
        print('No sequences found in alignment.', file=sys.stderr)
        sys.exit(1)
    aln_len = len(seqs[0])
    for s in seqs:
        if len(s) != aln_len:
            print('All sequences must be the same length.', file=sys.stderr)
            sys.exit(1)

    if args.alphabet:
        alphabet = list(args.alphabet)
    else:
        alphabet = sorted(set(''.join(seqs)))

    # Load accession to phylum mapping
    acc2phylum = load_accession_to_phylum(args.accession_csv)

    # Group sequences by phylum
    phylum_to_indices = defaultdict(list)
    for idx, header in enumerate(headers):
        acc = extract_accession(header)
        phylum = acc2phylum.get(acc, None)
        if phylum:
            phylum_to_indices[phylum].append(idx)

    phyla = sorted(phylum_to_indices.keys())

    # For conservation, use all symbols including gap
    max_entropy = log2(len(alphabet))  # e.g., 21 for 20 aa + '-'
    alpha = args.alpha

    # Prepare output directory
    import os
    output_dir = args.output_dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Calculate consensus of consensuses sequence (across phyla)
    global_consensus = consensus_of_consensuses(phylum_to_indices, seqs, alphabet, phyla)
    aln_len_cons = len(global_consensus)
    kingdom_csv_outfile = os.path.join(output_dir, 'consensus_Kingdom_conservation.csv') if output_dir else 'consensus_Kingdom_conservation.csv'
    # Prepare header
    phylum_cols = []
    for phylum in phyla:
        phylum_cols.extend([f'{phylum}_Conservation', f'{phylum}_Coverage', f'{phylum}_EffectiveConservation'])
    avg_eff_col = ['AvgPhylaEffectiveConservation']
    delta_cols = [f'{phylum}_DeltaEffCons' for phylum in phyla]
    header = ['Position', 'Consensus', 'Conservation', 'Coverage', 'EffectiveConservation'] + avg_eff_col + phylum_cols + delta_cols
    with open(kingdom_csv_outfile, 'w') as csv_out:
        csv_out.write(','.join(header) + '\n')
        pos_counter = 1
        for i in range(aln_len_cons):
            consensus_res = global_consensus[i]
            if consensus_res == '-':
                continue
            column = [seq[i] for seq in seqs]
            n_total = len(column)
            n_consensus = sum(1 for aa in column if aa == consensus_res)
            counts = Counter(column)
            p_gap = counts.get('-', 0) / n_total if n_total > 0 else 0
            coverage = 1 - p_gap
            C_raw = n_consensus / n_total if n_total > 0 else 0
            eff_cons = C_raw * (coverage ** alpha)

            # Per-phylum conservation for this position, and collect eff_cons for averaging
            phylum_eff_cons_list = []
            phylum_cons_vals = []
            phylum_cov_vals = []
            for phylum in phyla:
                indices = phylum_to_indices[phylum]
                if not indices:
                    phylum_cons_vals.append('')
                    phylum_cov_vals.append('')
                    phylum_eff_cons_list.append(None)
                    continue
                phylum_column = [seqs[idx][i] for idx in indices]
                n_phylum = len(phylum_column)
                phylum_counts = Counter(phylum_column)
                p_phylum = np.array([phylum_counts[a] / n_phylum if n_phylum > 0 else 0 for a in alphabet])
                H_phylum = shannon_entropy(p_phylum)
                C_raw_phylum = 1 - (H_phylum / max_entropy) if max_entropy > 0 else 0
                p_gap_phylum = phylum_counts.get('-', 0) / n_phylum if n_phylum > 0 else 0
                coverage_phylum = 1 - p_gap_phylum
                eff_cons_phylum = C_raw_phylum * (coverage_phylum ** alpha)
                phylum_cons_vals.append(f'{C_raw_phylum:.6f}')
                phylum_cov_vals.append(f'{coverage_phylum:.3f}')
                phylum_eff_cons_list.append(eff_cons_phylum)
            # Compute average effective conservation (ignore None)
            eff_cons_values = [v for v in phylum_eff_cons_list if v is not None]
            if len(eff_cons_values) > 0:
                avg_eff_cons = sum(eff_cons_values) / len(eff_cons_values)
            else:
                avg_eff_cons = ''
            # Compute delta for each phylum
            delta_vals = []
            for v in phylum_eff_cons_list:
                if v is not None:
                    delta_vals.append(f'{avg_eff_cons - v:.6f}')
                else:
                    delta_vals.append('')
            row = [pos_counter, consensus_res, f'{C_raw:.6f}', f'{coverage:.3f}', f'{eff_cons:.6f}', f'{avg_eff_cons:.6f}' if avg_eff_cons != '' else '']
            # Add per-phylum conservation, coverage, eff cons
            for c, cov, eff in zip(phylum_cons_vals, phylum_cov_vals, phylum_eff_cons_list):
                row.extend([c, cov, f'{eff:.6f}' if eff is not None else ''])
            # Add delta columns
            row.extend(delta_vals)
            csv_out.write(','.join(map(str, row)) + '\n')
            pos_counter += 1
    # (Removed duplicate argument parsing and setup)

    # Calculate conservation, coverage, and effective conservation for each position and phylum
    cons_table = []  # Each row: [pos, cons_phylum1, coverage_phylum1, eff_cons_phylum1, ...]
    for i in range(aln_len):
        row = [i+1]
        for phylum in phyla:
            indices = phylum_to_indices[phylum]
            if not indices:
                row.extend(['', '', ''])
                continue
            column = [seqs[idx][i] for idx in indices]
            n_total = len(column)
            counts = Counter(column)
            p = np.array([counts[a] / n_total if n_total > 0 else 0 for a in alphabet])
            # Shannon entropy including gaps
            H = shannon_entropy(p)
            C_raw = 1 - (H / max_entropy) if max_entropy > 0 else 0
            p_gap = counts.get('-', 0) / n_total if n_total > 0 else 0
            coverage = 1 - p_gap
            eff_cons = C_raw * (coverage ** alpha)
            row.extend([f'{C_raw:.6f}', f'{coverage:.3f}', f'{eff_cons:.6f}'])
        cons_table.append(row)

    # Output header
    main_output_path = args.output
    if output_dir and main_output_path:
        main_output_path = os.path.join(output_dir, os.path.basename(main_output_path))
    out = open(main_output_path, 'w') if main_output_path else sys.stdout
    header_cols = []
    for phylum in phyla:
        header_cols.extend([f'{phylum}_cons', f'{phylum}_coverage', f'{phylum}_eff_cons'])
    print('Position\t' + '\t'.join(header_cols), file=out)
    for row in cons_table:
        print('\t'.join(str(x) for x in row), file=out)
    if out is not sys.stdout:
        out.close()

    # Output consensus sequences for each phylum and global consensus, with and without gaps
    consensus_outfile_nogap = args.consensus_output or 'consensus_per_phylum_nogap.fasta'
    consensus_outfile_withgap = 'consensus_per_phylum_withgap.fasta'
    if output_dir:
        consensus_outfile_nogap = os.path.join(output_dir, os.path.basename(consensus_outfile_nogap))
        consensus_outfile_withgap = os.path.join(output_dir, os.path.basename(consensus_outfile_withgap))

    # Calculate true overall consensus (from all sequences)
    def overall_consensus(seqs, alphabet):
        aln_len = len(seqs[0])
        consensus = []
        for i in range(aln_len):
            column = [seq[i] for seq in seqs]
            if not column:
                consensus.append('-')
                continue
            counts = Counter(column)
            max_count = max(counts.values())
            max_residues = [res for res, cnt in counts.items() if cnt == max_count]
            consensus.append(sorted(max_residues)[0])
        return ''.join(consensus)

    # Write consensus with gaps
    with open(consensus_outfile_withgap, 'w') as cons_out:
        # Consensus of consensuses (with gaps)
        global_consensus = consensus_of_consensuses(phylum_to_indices, seqs, alphabet, phyla)
        cons_out.write(f'>CONSENSUS_OF_CONSENSUSES\n{global_consensus}\n')
        # True overall consensus (with gaps)
        all_consensus = overall_consensus(seqs, alphabet)
        cons_out.write(f'>CONSENSUS_ALL_SEQUENCES\n{all_consensus}\n')
        # Per-phylum consensus (with gaps)
        for phylum in phyla:
            indices = phylum_to_indices[phylum]
            if not indices:
                continue
            consensus_seq = consensus_for_indices(indices, seqs, alphabet)
            cons_out.write(f'>{phylum}\n{consensus_seq}\n')

    # Write consensus without gaps
    with open(consensus_outfile_nogap, 'w') as cons_out:
        # Consensus of consensuses (no gaps)
        global_consensus_nogap = global_consensus.replace('-', '')
        cons_out.write(f'>CONSENSUS_OF_CONSENSUSES\n{global_consensus_nogap}\n')
        # True overall consensus (no gaps)
        all_consensus_nogap = all_consensus.replace('-', '')
        cons_out.write(f'>CONSENSUS_ALL_SEQUENCES\n{all_consensus_nogap}\n')
        # Per-phylum consensus (no gaps)
        for phylum in phyla:
            indices = phylum_to_indices[phylum]
            if not indices:
                continue
            consensus_seq = consensus_for_indices(indices, seqs, alphabet)
            consensus_seq_nogap = consensus_seq.replace('-', '')
            cons_out.write(f'>{phylum}\n{consensus_seq_nogap}\n')

    # Output per-phylum consensus with mapped conservation (CSV for PyMOL mapping, gaps removed from consensus)
    for phylum in phyla:
        indices = phylum_to_indices[phylum]
        if not indices:
            continue
        consensus_seq = consensus_for_indices(indices, seqs, alphabet)
        consensus_seq_nogap = consensus_seq.replace('-', '')
        csv_outfile = f'consensus_{phylum}_conservation.csv'
        if output_dir:
            csv_outfile = os.path.join(output_dir, os.path.basename(csv_outfile))
        with open(csv_outfile, 'w') as csv_out:
            csv_out.write('Position,Consensus,Conservation,Coverage,EffectiveConservation\n')
            pos_counter = 1
            for i in range(aln_len):
                consensus_res = consensus_seq[i]
                if consensus_res == '-':
                    continue
                column = [seqs[idx][i] for idx in indices]
                n_total = len(column)
                counts = Counter(column)
                p = np.array([counts[a] / n_total if n_total > 0 else 0 for a in alphabet])
                H = shannon_entropy(p)
                C_raw = 1 - (H / max_entropy) if max_entropy > 0 else 0
                p_gap = counts.get('-', 0) / n_total if n_total > 0 else 0
                coverage = 1 - p_gap
                eff_cons = C_raw * (coverage ** alpha)
                csv_out.write(f'{pos_counter},{consensus_res},{C_raw:.6f},{coverage:.3f},{eff_cons:.6f}\n')
                pos_counter += 1

if __name__ == '__main__':
    main()
