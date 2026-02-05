
import matplotlib.pyplot as plt
from Bio import AlignIO
from Bio.SeqRecord import SeqRecord
import random
import csv
import sys
import argparse

domain_palette = [
    "#0a16aa", "#ff3f68", "#fff123", "#ff8845", "#910491",
    "#ffc21f", "#d50080", "#000c7d", "#c7c7c7"
]
domain_colors = {}  # Will be filled dynamically

def parse_tsv(tsv_file):
    """Parse TSV file and return domain predictions as dict: seq_id -> [(domain, start, end)]"""
    domain_predictions = {}
    with open(tsv_file) as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) < 9:
                continue
            seq_id = row[0]
            pfam_id = row[4]
            domain_name = row[5] if row[5] else pfam_id
            try:
                start = int(row[6])
                end = int(row[7])
            except ValueError:
                continue
            if seq_id not in domain_predictions:
                domain_predictions[seq_id] = []
            domain_predictions[seq_id].append((domain_name, start, end))
    return domain_predictions

# --- FUNCTIONS ---
def get_domain_color(domain):
    if domain not in domain_colors:
        # Assign from palette if available, else random
        idx = len(domain_colors)
        if idx < len(domain_palette):
            domain_colors[domain] = domain_palette[idx]
        else:
            domain_colors[domain] = (random.random(), random.random(), random.random())
    return domain_colors[domain]

def map_domains_to_alignment(aligned_seq, domains):
    """Map domain positions to alignment, accounting for gaps and non-domain residues."""
    mapping = ['none'] * len(aligned_seq)
    seq_pos = 0
    pos_map = []  # alignment index -> sequence position (or None for gap)
    for i, aa in enumerate(aligned_seq):
        if aa == '-':
            mapping[i] = 'gap'
            pos_map.append(None)
        else:
            seq_pos += 1
            pos_map.append(seq_pos)
    return mapping, pos_map

def plot_domain_boxes(alignment, domain_predictions, output_svg="domain_plot.svg"):
    fig, ax = plt.subplots(figsize=(15, len(alignment) * 0.45))
    all_domains = set()
    for domains in domain_predictions.values():
        for domain, _, _ in domains:
            all_domains.add(domain)
    # Assign colors for all domains in sorted order for determinism
    for idx, domain in enumerate(sorted(all_domains)):
        if domain not in domain_colors:
            if idx < len(domain_palette):
                domain_colors[domain] = domain_palette[idx]
            else:
                domain_colors[domain] = (random.random(), random.random(), random.random())
    # Colors for gaps and non-domain residues
    gap_color = 'gray'
    box_height = 0.4
    backbone_y = box_height / 2
    y_start = 0.7  # Start plotting first sequence above x-axis
    backbone_linewidth = 2
    desired_order = [
        "Ascomycota",
        "Basidiomycota",
        "Mucoromycota",
        "Zoopagomycota",
        "Chytridiomycota",
        "Blastocladiomycota",
        "CONSENSUS_OF_CONSENSUSES"
    ]
    # Build a dict for quick lookup
    record_dict = {record.id: record for record in alignment}
    label_x = -6  # Move labels further left
    n_seqs = sum(1 for seq_id in reversed(desired_order) if seq_id in record_dict and seq_id != "CONSENSUS_ALL_SEQUENCES")
    # Start plotting from y=0 (bottom)
    for plot_idx, seq_id in enumerate([sid for sid in reversed(desired_order) if sid in record_dict and sid != "CONSENSUS_ALL_SEQUENCES"]):
        y = y_start + plot_idx
        record = record_dict[seq_id]
        aligned_seq = str(record.seq)
        domains = domain_predictions.get(seq_id, [])
        mapping, pos_map = map_domains_to_alignment(aligned_seq, domains)
        # Draw continuous backbone line for the whole sequence
        ax.plot([0, len(mapping)], [y+backbone_y, y+backbone_y], color='black', linewidth=backbone_linewidth, zorder=1)
        # Draw solid domain blocks
        for domain, start, end in domains:
            domain_indices = [i for i, p in enumerate(pos_map) if p is not None and start <= p <= end]
            if domain_indices:
                left = domain_indices[0]
                right = domain_indices[-1] + 1
                ax.add_patch(plt.Rectangle((left, y), right-left, box_height, color=get_domain_color(domain), zorder=3))
        # Overlay gap boxes
        for i, val in enumerate(mapping):
            if val == 'gap':
                ax.add_patch(plt.Rectangle((i, y), 1, box_height, color=gap_color, zorder=4))
        ax.text(label_x, y + backbone_y, seq_id, va='center', ha='right')
    ax.set_xlim(label_x, len(alignment[0]))
    ax.set_ylim(0, y_start + n_seqs - 1 + box_height + 0.5)  # Sequences close to x-axis
    ax.set_yticks([])
    # Show only 10 equally spaced alignment positions
    aln_len = len(alignment[0])
    num_ticks = 10
    tick_positions = [int(round(i * (aln_len - 1) / (num_ticks - 1))) for i in range(num_ticks)]
    tick_labels = [str(pos + 1) for pos in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Alignment Position")
    # Draw vertical lines at each tick position
    for x in tick_positions:
        ax.axvline(x=x, color='gray', linestyle='--', linewidth=1, zorder=0)
    # Draw horizontal x-axis line at y=0
    ax.axhline(y=0, color='black', linewidth=1.5, zorder=0)
    # Remove plot box (spines)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Legend
    legend_handles = []
    import matplotlib.patches as mpatches
    for domain in sorted(all_domains):
        legend_handles.append(mpatches.Patch(color=get_domain_color(domain), label=domain))
    legend_handles.append(mpatches.Patch(color='black', label="Unannotated"))
    legend_handles.append(mpatches.Patch(color=gap_color, label="Alignment gap"))
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.08)  # Increase top and bottom margins
    plt.savefig(output_svg, format="svg")
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description='Plot domains on sequence alignments.')
    parser.add_argument('--alignment', required=True, help='Alignment file (FASTA or Clustal)')
    parser.add_argument('--tsv', required=True, help='Pfam/Interpro TSV file')
    parser.add_argument('--output', default='domain_plot.svg', help='Output SVG file')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    alignment_file = args.alignment
    tsv_file = args.tsv
    output_svg = args.output
    domain_predictions = parse_tsv(tsv_file)
    alignment = list(AlignIO.read(alignment_file, "fasta"))
    plot_domain_boxes(alignment, domain_predictions, output_svg)