import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import pairwise_distances
import os
import pandas as pd
from Bio import SeqIO
import multiprocessing
import pickle
from adjustText import adjust_text  # Add this import

# Define phylum colors using ColorBrewer
PHYLUM_COLORS = {
    'Ascomycota': '#000c7d',    # Blue
    'Basidiomycota': '#910491',       # Violet
    'Mucoromycota': '#ff3f68',     # Pink
    'Zoopagomycota': '#ff8845',    # Purple
    'Chytridiomycota': '#fff123',  # Orange
    'Blastocladiomycota': '#d50080',# Yellow
    'Cryptomycota': '#fff123'      # Brown
}

# # Add special organisms for labeling
# SPECIAL_ORGANISMS = {
#     "GCF_000146045.2": "Saccharomyces cerevisiae",
#     "GCA_000230395.2": "Aspergillus niger",
#     "GCF_000002655.1": "Aspergillus fumigatus",
#     "GCF_000182895.1": "Coprinopsis cinerea",
#     "GCF_000149305.1": "Rhizopus delemar",
#     "GCF_028827035.1": "Penicillium chrysogenum"
# }
# Add special organisms for labeling
SPECIAL_ORGANISMS = {
    "GCF_025024165": "K. alabastrina", #Zoopagomycota cluster 0
    "GCF_000149305": "R. delemar", #Mucoromycota cluster 2
    "GCF_000146045": "S. cerevisiae", #Ascomycota cluster 1
    "GCF_900519145": "U. hordei", #Basidiomycota cluster 3
    "GCA_015847185": "M. alpina", #Mucoromycota cluster 4
    "GCF_000523455": "B. oryzae", #Ascomycota cluster 5
    "GCF_000149205": "A. nidulans", #Ascomycota cluster 6
    "GCF_000300555": "A. bisporus", #Basidiomycota cluster 7
    "GCF_000182925": "N. crassa", #Ascomycota cluster 8
    "GCF_000002655": "A. fumigatus", #Ascomycota cluster 8
    "GCF_000143535": "B. cinerea", #Ascomycota cluster 8
    "GCF_000240135": "F. graminearum", #Ascomycota cluster 8
    "GCF_001720155": "C. wingfieldii", #Basidiomycota cluster 9
}

def load_embeddings(embedding_file):
    """Load embeddings from saved numpy file."""
    embeddings = np.load(embedding_file)
    print(f"Loaded embedding array shape: {embeddings.shape}")
    
    # Generate sequential IDs since we don't have protein IDs anymore
    protein_ids = [f"protein_{i+1}" for i in range(len(embeddings))]
    
    return embeddings, protein_ids

def load_taxa_info(taxa_file):
    """Load taxa information from CSV file."""
    df = pd.read_csv(taxa_file)
    # Strip version numbers from accessions to match FASTA processing
    df['Accession'] = df['Accession'].apply(lambda x: x.split('.')[0])
    return dict(zip(df.Accession, df.Phylum))


# New function to get accessions from ID file
def get_accessions_from_id_file(id_file):
    """Get accessions in order from ID file."""
    with open(id_file) as f:
        return [line.strip().split('.')[0] for line in f]

def add_organism_labels(ax, umap_coords, accessions, fontsize=6):
    """Add labels with lines pointing to special organisms."""
    texts = []
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    
    for i, acc in enumerate(accessions):
        base_acc = acc.split('.')[0]
        if base_acc in SPECIAL_ORGANISMS:
            x, y = umap_coords[i, 0], umap_coords[i, 1]
            
            # Create text annotation without arrow initially
            text = ax.text(x, y, SPECIAL_ORGANISMS[base_acc],
                         fontsize=fontsize,
                         ha='center',
                         va='bottom',
                         color='black',
                         zorder=5)
            texts.append(text)
    
    # Adjust text positions to avoid overlaps with arrows for all labels
    adjust_text(texts,
               ax=ax,
               arrowprops=dict(arrowstyle='-', color='black', lw=0.5, alpha=0.7),
               expand_points=(1.5, 1.5),
               force_points=(0.1, 0.1),
               force_text=(0.5, 0.5),
               lim=500)  # Increase iterations for better placement

def create_umap_plot(embeddings, protein_ids, output_dir, gene_name, phyla=None, ax=None, save_format='png', fasta_accessions=None):
    """Create and save UMAP plot of embeddings."""
    print(f"Input embedding matrix shape: {embeddings.shape}")
    
    if len(embeddings.shape) != 2:
        raise ValueError(f"Expected 2D embedding matrix, got shape {embeddings.shape}")
    
    similarity_matrix = cosine_similarity(embeddings)
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    
    umap = UMAP(
        n_components=2,
        metric='precomputed',
        random_state=42,
        min_dist=0.1,
        n_neighbors=100
    )
    
    distance_matrix = 1 - similarity_matrix
    # Ensure no negative distances (clip to [0, 2] since cosine distance should be in this range)
    distance_matrix = np.clip(distance_matrix, 0, 2)
    umap_coords = umap.fit_transform(distance_matrix)
    
    # If no axis provided, create new figure
    if ax is None:
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
    
    if phyla is not None:
        for phylum in PHYLUM_COLORS.keys():
            if phylum in set(phyla):
                mask = [p == phylum for p in phyla]
                ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1], 
                        alpha=0.6, label=phylum, 
                        color=PHYLUM_COLORS[phylum],
                        s=7)
        # Only add legend for the last subplot when plotting separately
        if ax.is_last_row():
            ax.legend(bbox_to_anchor=(1.05, 1), 
                     loc='upper left',
                     fontsize=12,
                     markerscale=2,
                     frameon=True)
        
        # Remove axis ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    else:
        ax.scatter(umap_coords[:, 0], umap_coords[:, 1], alpha=0.6, s=10)
    
    ax.set_title(f'{gene_name}')
    
    # Add labels for special organisms if accessions are provided
    if fasta_accessions is not None:
        add_organism_labels(ax, umap_coords, fasta_accessions, fontsize=8 if ax is None else 6)
    
    # Only save if we're not using subplots
    if ax is None:
        os.makedirs(output_dir, exist_ok=True)
        if save_format == 'png':
            plt.savefig(os.path.join(output_dir, f'embeddings_umap_{gene_name.lower()}.png'), 
                       dpi=300, bbox_inches='tight')
        elif save_format == 'fig':
            with open(os.path.join(output_dir, f'embeddings_umap_{gene_name.lower()}.fig.pickle'), 'wb') as f:
                pickle.dump(plt.gcf(), f)
        plt.close()

def main(gene_name, use_subplots=False, fig=None, ax=None, save_format='png', layer=12):
    # Allow override via function arguments or environment variables
    embedding_file = os.environ.get('EMBEDDING_FILE', f'{gene_name}_layer_{layer}_embeddings.npy')
    taxa_file = os.environ.get('TAXA_FILE', 'deduplicated_genomes.csv')
    id_file = os.environ.get('ID_FILE', f'{gene_name}_layer_{layer}_ids.txt')
    output_dir = os.environ.get('OUTPUT_DIR', f'plots/layer_{layer}')
    # Allow override via function arguments
    if hasattr(main, 'embedding_file') and main.embedding_file:
        embedding_file = main.embedding_file
    if hasattr(main, 'taxa_file') and main.taxa_file:
        taxa_file = main.taxa_file
    if hasattr(main, 'id_file') and main.id_file:
        id_file = main.id_file
    if hasattr(main, 'output_dir') and main.output_dir:
        output_dir = main.output_dir
    if hasattr(main, 'save_format') and main.save_format:
        save_format = main.save_format
    # Check if required files exist
    if not os.path.exists(embedding_file):
        print(f"Warning: Embedding file not found for {gene_name}: {embedding_file}")
        return False
    if not os.path.exists(id_file):
        print(f"Warning: ID file not found for {gene_name}: {id_file}")
        return False
    if not os.path.exists(taxa_file):
        print(f"Warning: Taxa file not found: {taxa_file}")
        return False

    try:
        taxa_dict = load_taxa_info(taxa_file)
        accessions = get_accessions_from_id_file(id_file)
        phyla = [taxa_dict[acc] for acc in accessions if acc in taxa_dict]

        print("Loading embeddings...")
        embeddings, protein_ids = load_embeddings(embedding_file)
        print(f"Loaded {len(protein_ids)} sequences with embedding shape: {embeddings.shape}")

        print("Creating UMAP plot...")
        create_umap_plot(embeddings, protein_ids, output_dir, gene_name,
                        phyla=phyla, ax=ax, save_format=save_format,
                        fasta_accessions=accessions)  # Pass accessions for labeling

        if not use_subplots:
            ext = 'png' if save_format == 'png' else 'fig.pickle'
            print(f"Plot saved to {output_dir}/embeddings_umap_{gene_name.lower()}_layer_{layer}.{ext}")

        return True

    except Exception as e:
        print(f"Error processing {gene_name}: {e}")
        return False

def load_outlier_accessions():
    """Load outlier accessions from the text file."""
    outliers_file = os.environ.get('OUTLIERS_FILE', 'outliers_set.txt')
    if not os.path.exists(outliers_file):
        print(f"Warning: Outliers file not found at {outliers_file}")
        return set()
    
    try:
        with open(outliers_file, 'r') as f:
            return {line.strip().split('.')[0] for line in f}
    except Exception as e:
        print(f"Error reading outliers file: {e}")
        return set()

def calculate_embedding_diversity(embeddings, metric='cosine'):
    """
    Calculate diversity metrics for protein embeddings.
    
    Args:
        embeddings: numpy array of shape (n_proteins, embedding_dim)
        metric: distance metric ('cosine', 'euclidean', 'manhattan')
    
    Returns:
        dict with diversity metrics
    """
    if metric == 'cosine':
        # Calculate cosine distances (1 - cosine similarity)
        similarity_matrix = cosine_similarity(embeddings)
        distance_matrix = 1 - similarity_matrix
    else:
        distance_matrix = pairwise_distances(embeddings, metric=metric)
    
    # Remove diagonal (self-distances)
    n = distance_matrix.shape[0]
    upper_triangle = distance_matrix[np.triu_indices(n, k=1)]
    
    diversity_metrics = {
        'mean_distance': np.mean(upper_triangle),
        'std_distance': np.std(upper_triangle),
        'median_distance': np.median(upper_triangle),
        'max_distance': np.max(upper_triangle),
        'min_distance': np.min(upper_triangle),
        'cv_distance': np.std(upper_triangle) / np.mean(upper_triangle) if np.mean(upper_triangle) > 0 else 0,
        'iqr_distance': np.percentile(upper_triangle, 75) - np.percentile(upper_triangle, 25),
        'dispersion_index': np.var(upper_triangle) / np.mean(upper_triangle) if np.mean(upper_triangle) > 0 else 0
    }
    
    return diversity_metrics

def calculate_phylum_diversity(embeddings, phyla, metric='cosine'):
    """
    Calculate diversity metrics within and between phylums.
    
    Returns:
        dict with within-phylum and between-phylum diversity
    """
    unique_phyla = list(set(phyla))
    phylum_indices = {phylum: [i for i, p in enumerate(phyla) if p == phylum] 
                      for phylum in unique_phyla}
    
    if metric == 'cosine':
        similarity_matrix = cosine_similarity(embeddings)
        distance_matrix = 1 - similarity_matrix
    else:
        distance_matrix = pairwise_distances(embeddings, metric=metric)
    
    within_phylum_diversity = {}
    for phylum, indices in phylum_indices.items():
        if len(indices) > 1:
            phylum_distances = distance_matrix[np.ix_(indices, indices)]
            upper_triangle = phylum_distances[np.triu_indices(len(indices), k=1)]
            within_phylum_diversity[phylum] = {
                'mean_distance': np.mean(upper_triangle),
                'std_distance': np.std(upper_triangle),
                'n_sequences': len(indices)
            }
    
    # Calculate between-phylum diversity
    between_phylum_diversity = {}
    for i, phylum1 in enumerate(unique_phyla):
        for phylum2 in unique_phyla[i+1:]:
            indices1 = phylum_indices[phylum1]
            indices2 = phylum_indices[phylum2]
            between_distances = distance_matrix[np.ix_(indices1, indices2)].flatten()
            between_phylum_diversity[f"{phylum1}_vs_{phylum2}"] = {
                'mean_distance': np.mean(between_distances),
                'std_distance': np.std(between_distances)
            }
    
    return within_phylum_diversity, between_phylum_diversity

def process_gene_data(gene_name, layer=12):
    """Process single gene and return UMAP coordinates, phyla, and diversity metrics"""
    embedding_file = os.environ.get('EMBEDDING_FILE', f'{gene_name}_layer_{layer}_embeddings.npy')
    taxa_file = os.environ.get('TAXA_FILE', 'deduplicated_genomes.csv')
    id_file = os.environ.get('ID_FILE', f'{gene_name}_layer_{layer}_ids.txt')
    umap_cache_file = os.environ.get('UMAP_CACHE_FILE', f'umap_cache/{gene_name}_layer_{layer}_umap.npz')

    # Check if required files exist
    if not os.path.exists(embedding_file):
        print(f"Warning: Embedding file not found for {gene_name}: {embedding_file}")
        return None
    if not os.path.exists(id_file):
        print(f"Warning: ID file not found for {gene_name}: {id_file}")
        return None
    if not os.path.exists(taxa_file):
        print(f"Warning: Taxa file not found: {taxa_file}")
        return None

    print(f"Processing {gene_name}...")

    # Load outliers
    try:
        outliers = load_outlier_accessions()
    except FileNotFoundError:
        print(f"Warning: Outliers file not found, proceeding without outlier filtering")
        outliers = set()

    try:
        taxa_dict = load_taxa_info(taxa_file)
        accessions = get_accessions_from_id_file(id_file)

        # Filter out outliers
        valid_indices = [i for i, acc in enumerate(accessions) if acc not in outliers]
        filtered_accessions = [accessions[i] for i in valid_indices]
        phyla = [taxa_dict[acc] for acc in filtered_accessions if acc in taxa_dict]

        # Check if we have any valid sequences after filtering
        if len(filtered_accessions) == 0:
            print(f"Warning: No valid sequences found for {gene_name} after filtering")
            return None

    except Exception as e:
        print(f"Error processing taxa/id data for {gene_name}: {e}")
        return None

    # Try to load cached UMAP coordinates
    if os.path.exists(umap_cache_file):
        print(f"Loading cached UMAP coordinates for {gene_name}")
        cached_data = np.load(umap_cache_file, allow_pickle=True)

        # Check if diversity metrics are in cache
        if 'diversity_metrics' in cached_data.files:
            diversity_metrics = cached_data['diversity_metrics'].item()
            within_phylum_div = cached_data['within_phylum_diversity'].item()
            between_phylum_div = cached_data['between_phylum_diversity'].item()
            return gene_name, cached_data['umap_coords'], phyla, filtered_accessions, diversity_metrics, within_phylum_div, between_phylum_div
        else:
            # Cache exists but without diversity metrics, need to recalculate
            print(f"Cache exists but missing diversity metrics for {gene_name}, recalculating...")

    # If no cache exists or cache is incomplete, proceed with full processing
    try:
        embeddings, _ = load_embeddings(embedding_file)
        # Filter embeddings
        embeddings = embeddings[valid_indices]

        # Check if we have enough embeddings for meaningful analysis
        if len(embeddings) < 2:
            print(f"Warning: Not enough valid embeddings for {gene_name} (found {len(embeddings)})")
            return None

        # Calculate diversity metrics
        diversity_metrics = calculate_embedding_diversity(embeddings, metric='cosine')
        within_phylum_div, between_phylum_div = calculate_phylum_diversity(embeddings, phyla, metric='cosine')

        similarity_matrix = cosine_similarity(embeddings)
        distance_matrix = 1 - similarity_matrix
        # Ensure no negative distances (clip to [0, 2] since cosine distance should be in this range)
        distance_matrix = np.clip(distance_matrix, 0, 2)

        umap = UMAP(
            n_components=2,
            metric='precomputed',
            random_state=42,
            min_dist=0.1,
            n_neighbors=min(100, len(embeddings) - 1)  # Ensure n_neighbors is not larger than n_samples - 1
        )

        umap_coords = umap.fit_transform(distance_matrix)

        # Cache the results including diversity metrics
        os.makedirs(os.path.dirname(umap_cache_file), exist_ok=True)
        np.savez(umap_cache_file,
                 umap_coords=umap_coords,
                 diversity_metrics=diversity_metrics,
                 within_phylum_diversity=within_phylum_div,
                 between_phylum_diversity=between_phylum_div)

        return gene_name, umap_coords, phyla, filtered_accessions, diversity_metrics, within_phylum_div, between_phylum_div

    except Exception as e:
        print(f"Error processing embeddings for {gene_name}: {e}")
        return None

def save_diversity_results(results, output_file):
    """Save diversity metrics to CSV file."""
    diversity_data = []
    
    for result in results:
        if len(result) == 7:  # Full result with diversity metrics
            gene_name, _, _, _, diversity_metrics, within_phylum_div, between_phylum_div = result
        else:  # Old cached result without diversity metrics
            print(f"Warning: No diversity metrics for {result[0]}, skipping...")
            continue
            
        row = {'gene': gene_name}
        row.update(diversity_metrics)
        
        # Add within-phylum diversity averages
        if within_phylum_div:
            within_means = [metrics['mean_distance'] for metrics in within_phylum_div.values()]
            within_stds = [metrics['std_distance'] for metrics in within_phylum_div.values()]
            row['avg_within_phylum_mean'] = np.mean(within_means)
            row['avg_within_phylum_std'] = np.mean(within_stds)
        
        diversity_data.append(row)
    
    df = pd.DataFrame(diversity_data)
    df.to_csv(output_file, index=False)
    print(f"Diversity metrics saved to {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='UMAP plot for ESM embeddings')
    parser.add_argument('--gene_names', nargs='+', default=["LYS20", "ACO2", "LYS4", "LYS12", "ARO8", "LYS2", "LYS9", "LYS1"], help='List of gene names')
    parser.add_argument('--use_subplots', action='store_true', help='Use subplots for all genes')
    parser.add_argument('--save_format', type=str, default='png', help='Save format (png or fig)')
    parser.add_argument('--embedding_file', type=str, help='Embedding file (for single gene)')
    parser.add_argument('--taxa_file', type=str, help='Taxa file')
    parser.add_argument('--id_file', type=str, help='ID file')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--layer', type=int, default=12, help='Layer number')
    parser.add_argument('--outliers_file', type=str, help='Outliers file')
    parser.add_argument('--umap_cache_file', type=str, help='UMAP cache file')
    parser.add_argument('--diversity_csv', type=str, default='protein_diversity_metrics.csv', help='Diversity metrics output CSV')
    args = parser.parse_args()

    # Set environment variables for file overrides
    if args.embedding_file:
        os.environ['EMBEDDING_FILE'] = args.embedding_file
    if args.taxa_file:
        os.environ['TAXA_FILE'] = args.taxa_file
    if args.id_file:
        os.environ['ID_FILE'] = args.id_file
    if args.output_dir:
        os.environ['OUTPUT_DIR'] = args.output_dir
    if args.outliers_file:
        os.environ['OUTLIERS_FILE'] = args.outliers_file
    if args.umap_cache_file:
        os.environ['UMAP_CACHE_FILE'] = args.umap_cache_file

    gene_names = args.gene_names
    use_subplots = args.use_subplots
    save_format = args.save_format
    layer = args.layer

    if use_subplots:
        num_cores = multiprocessing.cpu_count()
        print(f"Running on {num_cores} cores")
        with multiprocessing.Pool(processes=num_cores) as pool:
            all_results = pool.map(process_gene_data, gene_names)
        results = [r for r in all_results if r is not None]
        if len(results) == 0:
            print("Error: No genes were successfully processed!")
            exit(1)
        print(f"Successfully processed {len(results)} out of {len(gene_names)} genes")
        if len(results) < len(gene_names):
            failed_genes = [gene for gene, result in zip(gene_names, all_results) if result is None]
            print(f"Failed to process: {', '.join(failed_genes)}")
        save_diversity_results(results, args.diversity_csv)
        print("\nDiversity Summary (mean pairwise cosine distance):")
        print("-" * 60)
        valid_results = [r for r in results if len(r) == 7]
        for gene_name, _, _, _, diversity_metrics, _, _ in valid_results:
            print(f"{gene_name:8}: {diversity_metrics['mean_distance']:.4f} ± {diversity_metrics['std_distance']:.4f}")
        if valid_results:
            sorted_results = sorted(valid_results, key=lambda x: x[4]['mean_distance'], reverse=True)
            print(f"\nMost diverse protein: {sorted_results[0][0]} (diversity: {sorted_results[0][4]['mean_distance']:.4f})")
            print(f"Least diverse protein: {sorted_results[-1][0]} (diversity: {sorted_results[-1][4]['mean_distance']:.4f})")
        # ...existing code for plotting and legend...
        # (No hardcoded output paths; user must specify output_dir or use default)
    else:
        num_cores = multiprocessing.cpu_count()
        print(f"Running on {num_cores} cores")
        available_genes = []
        for gene_name in gene_names:
            embedding_file = os.environ.get('EMBEDDING_FILE', f'{gene_name}_layer_{layer}_embeddings.npy')
            # No hardcoded fasta_file check; user must provide files as needed
            if os.path.exists(embedding_file):
                available_genes.append(gene_name)
            else:
                print(f"Skipping {gene_name}: missing required files")
        if len(available_genes) == 0:
            print("Error: No genes have the required files available!")
            exit(1)
        print(f"Processing {len(available_genes)} available genes: {', '.join(available_genes)}")
        with multiprocessing.Pool(processes=num_cores) as pool:
            from functools import partial
            main_with_format = partial(main, save_format=save_format)
            pool.map(main_with_format, available_genes)