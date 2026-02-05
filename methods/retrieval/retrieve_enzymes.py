import pandas as pd
import os
from tqdm import tqdm

# Define maximum sequence length limits for each enzyme
# Maximum allowed sequence lengths for each gene (in amino acids):
# ACO2: 1000, LYS1: 1100, LYS2: 2000, ARO8: 800, LYS12: 1000, LYS20: 1000, LYS9: 1000, LYS4: 1500
# Add bigg numbers for no length filtering!!
MAX_LENGTHS = {
    'ACO2': 10000,
    'LYS1': 10000,
    'LYS2': 10000,
    'ARO8': 10000,
    'LYS12': 10000,
    'LYS20': 10000,
    'LYS9': 10000,
    'LYS4': 10000
}

def process_results(result_path, name, gene_dict):

    df = pd.read_csv(result_path, sep='\t', header=None, names=['Gene', 'Accession', 'E', 'Seq'])

    # Instead of taking the lowest E-value match directly, process each gene separately
    genes = df['Gene'].unique()

    for gene in genes:
        # Get all results for this gene, sorted by E-value (ascending)
        gene_results = df[df['Gene'] == gene].sort_values('E')
        
        # Find the first result that satisfies the length limit
        found_valid_seq = False
        for _, row in gene_results.iterrows():
            seq = row['Seq']
            if len(seq) <= MAX_LENGTHS.get(gene, float('inf')):
                fasta_string = f">{name}\n{seq}\n"
                gene_dict[gene].append(fasta_string)
                found_valid_seq = True
                break
                
        if not found_valid_seq:
            min_e_seq_len = len(gene_results.iloc[0]['Seq'])
            print(f"Skipping {name} {gene}: all sequences exceed maximum length {MAX_LENGTHS[gene]}, " 
                  f"shortest is {min_e_seq_len} (E-value: {gene_results.iloc[0]['E']})")

    return gene_dict

def check_gene_counts(gene_dict):
    counts = {gene: len(sequences) for gene, sequences in gene_dict.items()}
    max_count = max(counts.values())
    
    missing = {}
    for gene, count in counts.items():
        if count < max_count:
            missing[gene] = max_count - count
    
    if missing:
        print("\nInconsistent gene counts found:")
        print(f"Maximum sequences found: {max_count}")
        for gene, missing_count in missing.items():
            print(f"{gene}: missing {missing_count} sequences")
    else:
        print(f"\nAll genes have {max_count} sequences")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Retrieve enzyme sequences from protein results.')
    parser.add_argument('--parent_dir', required=True, help='Parent directory containing protein subfolders')
    parser.add_argument('--result_dir', required=True, help='Directory to write filtered FASTA files')
    parser.add_argument('--tax_file', required=True, help='CSV file with valid accessions')
    parser.add_argument('--genes', nargs='+', default=['LYS1', 'LYS20', 'ACO2', 'LYS4','LYS12','ARO8', 'LYS2', 'LYS9'], help='List of gene names')
    args = parser.parse_args()

    PARENT_DIR = args.parent_dir
    RESULT_DIR = args.result_dir
    TAX_FILE = args.tax_file
    gene_list = args.genes

    valid_accessions = pd.read_csv(TAX_FILE)['Accession'].tolist()

    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    gene_dict = {gene: [] for gene in gene_list}
    accession_genes = {}

    subdirs = next(os.walk(PARENT_DIR))[1]
    for subdir in tqdm(subdirs, desc="Processing subdirectories"):
        if subdir not in valid_accessions:
            continue
        subdir_path = os.path.join(PARENT_DIR, subdir)
        result_path = os.path.join(subdir_path, 'results.tsv')
        if os.path.exists(result_path):
            name = os.path.basename(subdir_path)
            original_counts = {gene: len(sequences) for gene, sequences in gene_dict.items()}
            gene_dict = process_results(result_path, name, gene_dict)
            accession_genes[name] = []
            for gene, sequences in gene_dict.items():
                if len(sequences) > original_counts[gene]:
                    accession_genes[name].append(gene)
        else:
            print(f'No results for {subdir_path}')

    all_genes = list(gene_dict.keys())
    complete_accessions = [acc for acc, genes in accession_genes.items() if set(genes) == set(all_genes)]
    print(f"\nFound {len(complete_accessions)} accessions with all {len(all_genes)} genes")

    filtered_gene_dict = {gene: [] for gene in all_genes}
    for gene, sequences in gene_dict.items():
        for sequence in sequences:
            acc_name = sequence.strip().split('\n')[0][1:]
            if acc_name in complete_accessions:
                filtered_gene_dict[gene].append(sequence)

    for gene, sequences in filtered_gene_dict.items():
        with open(os.path.join(RESULT_DIR, f'{gene}.fasta'), 'w') as f:
            f.writelines(sequences)

    check_gene_counts(filtered_gene_dict)

if __name__ == '__main__':
    main()


