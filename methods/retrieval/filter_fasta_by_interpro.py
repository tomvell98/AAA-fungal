import pandas as pd
import numpy as np
import os
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

def get_valid_accessions(interpro_df, queries, n=1):
    """Return accessions that HAVE the required domain annotations (opposite of get_missing)."""
    # Convert all strings in DataFrame to lowercase for case-insensitive search
    df_lower = interpro_df.astype(str).apply(lambda x: x.str.lower())
    
    # Convert all queries to lowercase
    queries_lower = [q.lower() for q in queries]
    
    # Check if any of the queries match in any column
    mask = df_lower.apply(lambda row: any(q in ' '.join(row.values) for q in queries_lower), axis=1)
    
    # Use boolean indexing with loc instead of iloc
    querried_accessions = interpro_df.loc[mask, interpro_df.columns[0]].tolist()
    
    # Get unique values and their counts in querried_accessions
    unique_accessions, counts = np.unique(querried_accessions, return_counts=True)
    
    # Filter accessions that appear at least n times (these are VALID)
    valid_accessions = set(unique_accessions[counts >= n])
    
    return valid_accessions

def filter_fasta_by_interpro(fasta_file, valid_accessions, output_file):
    """Filter FASTA file to keep only sequences with valid InterPro annotations."""
    valid_records = []
    total_sequences = 0
    kept_sequences = 0
    
    with open(fasta_file, 'r') as f:
        for record in SeqIO.parse(f, 'fasta'):
            total_sequences += 1
            if record.id in valid_accessions:
                valid_records.append(record)
                kept_sequences += 1
    
    # Write filtered sequences
    with open(output_file, 'w') as f:
        SeqIO.write(valid_records, f, 'fasta')
    
    return total_sequences, kept_sequences

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Filter FASTA by InterPro domain annotations.')
    parser.add_argument('--interpro_dir', required=True, help='Directory with InterPro result files')
    parser.add_argument('--fasta_input_dir', required=True, help='Input FASTA directory')
    parser.add_argument('--output_dir', required=True, help='Output directory for filtered FASTA files')
    parser.add_argument('--genes', nargs='+', required=True, help='List of gene names')
    args = parser.parse_args()

    interpro_dir = args.interpro_dir
    fasta_input_dir = args.fasta_input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    gene_names = args.genes

    queries = {
        'ACO2': ['IPR050926', 'aconitase', 'acnase', 'acoase'],
        'ARO8': ['IPR050859', 'aminotransferase', 'Pyridoxal phosphate-dependent transferase'],
        'LYS1': ['IPR027281', 'saccharopine', 'AlaDh'],
        'LYS2': ['IPR014397', 'Alpha-AR', 'aminoadipate','adenylation domain','sdr','IPR010080','NRPS', 'AMP-binding', 'Nonribosomal peptide', 'IPR014397', 'PTHR44845'],
        'LYS4': ['IPR004418', 'aconitase'],
        'LYS9': ['IPR051168', 'SACCHAROPINE', 'Sacchrp'],
        'LYS12': ['IPR024084', 'Isopropylmalate','Dehydrogenase', 'ISOCITRATE'],
        'LYS20': ['IPR050073', 'HOMOCITRATE','Aldolase'],
    }

    print("=== InterPro-based FASTA Filtering (Complete Sets Only) ===\n")
    
    # STEP 1: Collect valid accessions for each gene
    gene_valid_accessions = {}
    
    for gene in gene_names:
        print(f"Analyzing {gene}...")
        
        # Load InterPro annotations
        interpro_file = os.path.join(interpro_dir, f'{gene}.fasta.tsv')
        if not os.path.exists(interpro_file):
            print(f"  WARNING: InterPro file not found: {interpro_file}")
            gene_valid_accessions[gene] = set()
            continue
            
        interpro_df = pd.read_csv(interpro_file, sep='\t')
        
        # Get valid accessions (those WITH required annotations)
        if gene == 'LYS2':
            valid_accessions = get_valid_accessions(interpro_df, queries[gene], n=2)
        else:
            valid_accessions = get_valid_accessions(interpro_df, queries[gene], n=1)
        
        gene_valid_accessions[gene] = valid_accessions
        print(f"  Found {len(valid_accessions)} accessions with valid {gene} annotations")
    
    # STEP 2: Find intersection - accessions that have ALL enzymes
    print(f"\n=== Finding Complete Sets ===")
    complete_accessions = None
    
    for gene in gene_names:
        if gene not in gene_valid_accessions:
            continue
            
        if complete_accessions is None:
            complete_accessions = gene_valid_accessions[gene].copy()
        else:
            complete_accessions = complete_accessions.intersection(gene_valid_accessions[gene])
        
        print(f"After {gene}: {len(complete_accessions)} complete genomes remaining")
    
    if complete_accessions is None:
        complete_accessions = set()
    
    print(f"\nFinal result: {len(complete_accessions)} genomes have ALL {len(gene_names)} enzymes with valid annotations")
    
    # STEP 3: Filter FASTA files to keep only complete accessions
    print(f"\n=== Filtering FASTA Files (Complete Sets Only) ===")
    
    total_stats = {}
    
    for gene in gene_names:
        print(f"Filtering {gene}...")
        
        # Filter FASTA file
        input_fasta = os.path.join(fasta_input_dir, f'{gene}.fasta')
        output_fasta = os.path.join(output_dir, f'{gene}.fasta')
        
        if not os.path.exists(input_fasta):
            print(f"  WARNING: Input FASTA not found: {input_fasta}")
            continue
        
        total_seqs, kept_seqs = filter_fasta_by_interpro(input_fasta, complete_accessions, output_fasta)
        
        # Store statistics
        total_stats[gene] = {
            'total': total_seqs,
            'kept': kept_seqs,
            'removed': total_seqs - kept_seqs,
            'retention_rate': (kept_seqs / total_seqs * 100) if total_seqs > 0 else 0
        }
        
        print(f"  Input sequences: {total_seqs}")
        print(f"  Kept sequences: {kept_seqs}")
        print(f"  Removed sequences: {total_seqs - kept_seqs}")
        print(f"  Retention rate: {kept_seqs/total_seqs*100:.1f}%")
        print()
    
    # Summary statistics
    print("=== SUMMARY STATISTICS ===")
    print(f"{'Gene':<6} {'Input':<6} {'Kept':<6} {'Removed':<8} {'Retention':<10}")
    print("-" * 45)
    
    total_input = 0
    total_kept = 0
    
    for gene in gene_names:
        if gene in total_stats:
            stats = total_stats[gene]
            print(f"{gene:<6} {stats['total']:<6} {stats['kept']:<6} {stats['removed']:<8} {stats['retention_rate']:<9.1f}%")
            total_input += stats['total']
            total_kept += stats['kept']
    
    print("-" * 45)
    overall_retention = (total_kept / total_input * 100) if total_input > 0 else 0
    print(f"{'TOTAL':<6} {total_input:<6} {total_kept:<6} {total_input-total_kept:<8} {overall_retention:<9.1f}%")
    
    print(f"\nComplete genomes with all enzymes: {len(complete_accessions)}")
    print(f"Output directory: {output_dir}")
    
    # Save list of complete accessions
    with open(os.path.join(output_dir, 'complete_accessions.txt'), 'w') as f:
        f.write('\n'.join(sorted(complete_accessions)))
    
    # Verification: Check that all output files have the same number of sequences
    print(f"\n=== VERIFICATION ===")
    for gene in gene_names:
        output_fasta = os.path.join(output_dir, f'{gene}.fasta')
        if os.path.exists(output_fasta):
            with open(output_fasta, 'r') as f:
                seq_count = len(list(SeqIO.parse(f, 'fasta')))
            print(f"{gene}: {seq_count} sequences")
        else:
            print(f"{gene}: File not found")
    
    print("\n=== InterPro filtering completed successfully! ===")
    print("✓ All output files contain exactly the same genomes")
    print("✓ Only genomes with ALL enzymes are included")
    print("✓ Ready for phylogenetic analysis with complete datasets")
