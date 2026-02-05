import pandas as pd
import os
import sys
from Bio import SeqIO
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import numpy as np

def extract_accessions_from_fasta(fasta_file):
    """Extract accession IDs from a FASTA file."""
    accessions = []
    if os.path.exists(fasta_file):
        with open(fasta_file, 'r') as f:
            for record in SeqIO.parse(f, 'fasta'):
                accessions.append(record.id)
    return set(accessions)

def analyze_taxonomic_distribution(accessions, taxonomy_df, label="Dataset"):
    """Analyze taxonomic distribution for a set of accessions."""
    # Filter taxonomy data for accessions present in the dataset
    filtered_taxonomy = taxonomy_df[taxonomy_df['Accession'].isin(accessions)]
    
    distributions = {}
    taxonomic_levels = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus']
    
    for level in taxonomic_levels:
        if level in filtered_taxonomy.columns:
            counts = filtered_taxonomy[level].value_counts()
            distributions[level] = {
                'counts': counts,
                'total': len(filtered_taxonomy),
                'unique_taxa': len(counts)
            }
    
    return distributions

def create_comparison_table(original_dist, filtered_dist, output_file):
    """Create a detailed comparison table showing losses at each taxonomic level."""
    
    taxonomic_levels = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus']
    
    # Create summary table
    summary_data = []
    
    for level in taxonomic_levels:
        if level in original_dist and level in filtered_dist:
            orig_total = original_dist[level]['total']
            filt_total = filtered_dist[level]['total']
            orig_taxa = original_dist[level]['unique_taxa']
            filt_taxa = filtered_dist[level]['unique_taxa']
            
            # Calculate losses
            genome_loss = orig_total - filt_total
            genome_retention = (filt_total / orig_total * 100) if orig_total > 0 else 0
            taxa_loss = orig_taxa - filt_taxa
            taxa_retention = (filt_taxa / orig_taxa * 100) if orig_taxa > 0 else 0
            
            summary_data.append({
                'Taxonomic_Level': level,
                'Original_Genomes': orig_total,
                'Filtered_Genomes': filt_total,
                'Genomes_Lost': genome_loss,
                'Genome_Retention_%': round(genome_retention, 1),
                'Original_Taxa': orig_taxa,
                'Filtered_Taxa': filt_taxa,
                'Taxa_Lost': taxa_loss,
                'Taxa_Retention_%': round(taxa_retention, 1)
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary table
    summary_file = output_file.replace('.csv', '_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary table saved to: {summary_file}")
    
    # Create detailed comparison for each taxonomic level
    detailed_data = []
    
    for level in taxonomic_levels:
        if level in original_dist and level in filtered_dist:
            orig_counts = original_dist[level]['counts']
            filt_counts = filtered_dist[level]['counts']
            
            # Get all taxa from both datasets
            all_taxa = set(orig_counts.index) | set(filt_counts.index)
            
            for taxon in all_taxa:
                orig_count = orig_counts.get(taxon, 0)
                filt_count = filt_counts.get(taxon, 0)
                loss = orig_count - filt_count
                retention = (filt_count / orig_count * 100) if orig_count > 0 else 0
                
                detailed_data.append({
                    'Taxonomic_Level': level,
                    'Taxon': taxon,
                    'Original_Count': orig_count,
                    'Filtered_Count': filt_count,
                    'Loss': loss,
                    'Retention_%': round(retention, 1),
                    'Status': 'Retained' if filt_count > 0 else 'Lost'
                })
    
    detailed_df = pd.DataFrame(detailed_data)
    detailed_df.to_csv(output_file, index=False)
    print(f"Detailed comparison saved to: {output_file}")
    
    return summary_df, detailed_df

def analyze_taxonomic_composition(input_dir, original_dataset, filtered_dataset, output_dir):
    """Analyze taxonomic composition between two datasets."""
    
    # Look for taxonomy file (now passed as argument)
    taxonomy_file = analyze_taxonomic_composition.taxonomy_file
    if not os.path.exists(taxonomy_file):
        raise FileNotFoundError(f"Taxonomy file not found: {taxonomy_file}")
    print(f"Loading taxonomy data from: {taxonomy_file}")
    taxonomy_df = pd.read_csv(taxonomy_file)
    print(f"Loaded taxonomy for {len(taxonomy_df)} genomes")
    
    # Load accessions from both datasets
    original_path = os.path.join(input_dir, original_dataset)
    filtered_path = os.path.join(input_dir, filtered_dataset)
    
    # Get accessions from sample enzyme files
    original_file = os.path.join(original_path, 'ACO2.fasta')
    filtered_file = os.path.join(filtered_path, 'ACO2.fasta')
    
    if not os.path.exists(original_file):
        raise FileNotFoundError(f"Original dataset file not found: {original_file}")
    if not os.path.exists(filtered_file):
        raise FileNotFoundError(f"Filtered dataset file not found: {filtered_file}")
    
    print(f"Extracting accessions from {original_dataset}...")
    original_accessions = extract_accessions_from_fasta(original_file)
    print(f"Found {len(original_accessions)} accessions in original dataset")
    
    print(f"Extracting accessions from {filtered_dataset}...")
    filtered_accessions = extract_accessions_from_fasta(filtered_file)
    print(f"Found {len(filtered_accessions)} accessions in filtered dataset")
    
    # Analyze taxonomic distributions
    original_dist = analyze_taxonomic_distribution(original_accessions, taxonomy_df, original_dataset)
    filtered_dist = analyze_taxonomic_distribution(filtered_accessions, taxonomy_df, filtered_dataset)
    
    # Create comparison table
    comparison_file = os.path.join(output_dir, 'detailed_taxonomic_analysis.csv')
    summary_df, detailed_df = create_comparison_table(original_dist, filtered_dist, comparison_file)
    
    return summary_df, detailed_df

def create_multi_dataset_comparison(input_dir, dataset_names, output_dir):
    """Create comprehensive comparison across multiple datasets."""
    
    # Look for taxonomy file (now passed as argument)
    taxonomy_file = create_multi_dataset_comparison.taxonomy_file
    if not os.path.exists(taxonomy_file):
        raise FileNotFoundError(f"Taxonomy file not found: {taxonomy_file}")
    print(f"Loading taxonomy data from: {taxonomy_file}")
    taxonomy_df = pd.read_csv(taxonomy_file)
    print(f"Loaded taxonomy for {len(taxonomy_df)} genomes")
    
    # Analyze each dataset
    distributions = {}
    
    for dataset_name in dataset_names:
        print(f"\nAnalyzing {dataset_name}...")
        dataset_path = os.path.join(input_dir, dataset_name)
        sample_file = os.path.join(dataset_path, 'ACO2.fasta')
        
        if os.path.exists(sample_file):
            accessions = extract_accessions_from_fasta(sample_file)
            print(f"  Found {len(accessions)} accessions")
            
            # Analyze taxonomic distribution
            dist = analyze_taxonomic_distribution(accessions, taxonomy_df, dataset_name)
            distributions[dataset_name] = dist
            
            # Print summary
            for level, data in dist.items():
                print(f"    {level}: {data['total']} genomes, {data['unique_taxa']} unique taxa")
        else:
            print(f"  WARNING: Sample file not found: {sample_file}")
    
    # Create comprehensive comparison tables
    genome_comparison, taxa_comparison, genome_retention, taxa_retention = create_multi_dataset_comparison_tables(distributions, output_dir)
    
    # Create comprehensive visualization
    create_comprehensive_visualization(genome_comparison, taxa_comparison, genome_retention, taxa_retention, output_dir)
    
    return distributions

def create_multi_dataset_comparison_tables(distributions, output_dir):
    """Create comparison tables for multiple datasets."""
    
    taxonomic_levels = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus']
    dataset_names = list(distributions.keys())
    
    # Create comprehensive summary table
    comprehensive_data = []
    
    for level in taxonomic_levels:
        for dataset_name in dataset_names:
            if level in distributions[dataset_name]:
                data = distributions[dataset_name][level]
                comprehensive_data.append({
                    'Dataset': dataset_name,
                    'Taxonomic_Level': level,
                    'Total_Genomes': data['total'],
                    'Unique_Taxa': data['unique_taxa']
                })
    
    comprehensive_df = pd.DataFrame(comprehensive_data)
    
    # Pivot for easier comparison
    genome_comparison = comprehensive_df.pivot(index='Taxonomic_Level', columns='Dataset', values='Total_Genomes')
    taxa_comparison = comprehensive_df.pivot(index='Taxonomic_Level', columns='Dataset', values='Unique_Taxa')
    
    # Calculate retention rates relative to first dataset
    first_dataset = dataset_names[0]
    genome_retention = genome_comparison.div(genome_comparison[first_dataset], axis=0) * 100
    taxa_retention = taxa_comparison.div(taxa_comparison[first_dataset], axis=0) * 100
    
    # Save comparison tables
    genome_comparison.to_csv(os.path.join(output_dir, 'genome_counts_all_datasets.csv'))
    taxa_comparison.to_csv(os.path.join(output_dir, 'taxa_counts_all_datasets.csv'))
    genome_retention.to_csv(os.path.join(output_dir, 'genome_retention_all_datasets.csv'))
    taxa_retention.to_csv(os.path.join(output_dir, 'taxa_retention_all_datasets.csv'))
    
    print(f"Multi-dataset comparison tables saved to: {output_dir}")
    
    return genome_comparison, taxa_comparison, genome_retention, taxa_retention

def create_pairwise_comparisons(input_dir, dataset_names, output_dir):
    """Create pairwise comparisons between consecutive datasets."""
    
    # Load distributions first
    distributions = {}
    
    # Look for taxonomy file (now passed as argument)
    taxonomy_file = create_pairwise_comparisons.taxonomy_file
    if not os.path.exists(taxonomy_file):
        raise FileNotFoundError(f"Taxonomy file not found: {taxonomy_file}")
    print(f"Loading taxonomy data from: {taxonomy_file}")
    taxonomy_df = pd.read_csv(taxonomy_file)
    
    # Load distributions for each dataset
    for dataset_name in dataset_names:
        dataset_path = os.path.join(input_dir, dataset_name)
        sample_file = os.path.join(dataset_path, 'ACO2.fasta')
        
        if os.path.exists(sample_file):
            accessions = extract_accessions_from_fasta(sample_file)
            distributions[dataset_name] = analyze_taxonomic_distribution(accessions, taxonomy_df, dataset_name)
    
    # Create pairwise comparisons
    pairwise_dir = os.path.join(output_dir, 'pairwise_comparisons')
    os.makedirs(pairwise_dir, exist_ok=True)
    
    for i in range(len(dataset_names) - 1):
        dataset1 = dataset_names[i]
        dataset2 = dataset_names[i + 1]
        
        print(f"Creating pairwise comparison: {dataset1} vs {dataset2}")
        
        # Create comparison table
        comparison_file = os.path.join(pairwise_dir, f'{dataset1}_vs_{dataset2}_comparison.csv')
        summary_df, detailed_df = create_comparison_table(
            distributions[dataset1], 
            distributions[dataset2], 
            comparison_file
        )
        
        # Create individual visualization
        viz_dir = os.path.join(pairwise_dir, f'{dataset1}_vs_{dataset2}')
        os.makedirs(viz_dir, exist_ok=True)
        create_visualization(summary_df, detailed_df, viz_dir)


def create_comprehensive_visualization(genome_comparison, taxa_comparison, genome_retention, taxa_retention, output_dir):
    """Create comprehensive visualizations comparing all datasets."""
    
    dataset_names = genome_comparison.columns.tolist()
    taxonomic_levels = genome_comparison.index.tolist()
    
    # Color palette for datasets
    colors = plt.cm.Set3(np.linspace(0, 1, len(dataset_names)))
    
    # Figure 1: Genome counts across all datasets
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # Genome counts
    ax1 = axes[0, 0]
    x = np.arange(len(taxonomic_levels))
    width = 0.8 / len(dataset_names)
    
    for i, dataset in enumerate(dataset_names):
        ax1.bar(x + i * width - width * (len(dataset_names) - 1) / 2, 
                genome_comparison[dataset], width, label=dataset, color=colors[i], alpha=0.8)
    
    ax1.set_title('Genome Counts Across All Datasets', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Number of Genomes', fontsize=12)
    ax1.set_xlabel('Taxonomic Level', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(taxonomic_levels, rotation=45)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Taxa diversity
    ax2 = axes[0, 1]
    for i, dataset in enumerate(dataset_names):
        ax2.bar(x + i * width - width * (len(dataset_names) - 1) / 2, 
                taxa_comparison[dataset], width, label=dataset, color=colors[i], alpha=0.8)
    
    ax2.set_title('Taxonomic Diversity Across All Datasets', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Number of Unique Taxa', fontsize=12)
    ax2.set_xlabel('Taxonomic Level', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(taxonomic_levels, rotation=45)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Genome retention rates
    ax3 = axes[1, 0]
    for i, dataset in enumerate(dataset_names):
        ax3.plot(taxonomic_levels, genome_retention[dataset], 
                marker='o', linewidth=2, label=dataset, color=colors[i])
    
    ax3.set_title('Genome Retention Rates (% of Original)', fontsize=16, fontweight='bold')
    ax3.set_ylabel('Retention Percentage (%)', fontsize=12)
    ax3.set_xlabel('Taxonomic Level', fontsize=12)
    ax3.set_ylim(0, 105)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    # Taxa retention rates
    ax4 = axes[1, 1]
    for i, dataset in enumerate(dataset_names):
        ax4.plot(taxonomic_levels, taxa_retention[dataset], 
                marker='s', linewidth=2, label=dataset, color=colors[i])
    
    ax4.set_title('Taxa Diversity Retention (% of Original)', fontsize=16, fontweight='bold')
    ax4.set_ylabel('Retention Percentage (%)', fontsize=12)
    ax4.set_xlabel('Taxonomic Level', fontsize=12)
    ax4.set_ylim(0, 105)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comprehensive_taxonomic_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comprehensive comparison plot saved to: {os.path.join(output_dir, 'comprehensive_taxonomic_comparison.png')}")
    
    # Figure 2: Phylum-specific detailed comparison
    create_phylum_comparison_all_datasets(genome_comparison, taxa_comparison, output_dir)

def create_phylum_comparison_all_datasets(genome_comparison, taxa_comparison, output_dir):
    """Create detailed phylum comparison across all datasets."""
    
    # This would need phylum-level data - let's create a placeholder for now
    # and focus on the main comparison above
    print("Phylum-specific comparison across datasets would require detailed phylum data...")

def create_visualization(summary_df, detailed_df, output_dir):
    """Create visualizations showing taxonomic losses between two datasets."""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Figure 1: Summary retention rates
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Genome retention rates
    ax1 = axes[0, 0]
    bars1 = ax1.bar(summary_df['Taxonomic_Level'], summary_df['Genome_Retention_%'])
    ax1.set_title('Genome Retention Rates by Taxonomic Level', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Retention Percentage (%)', fontsize=12)
    ax1.set_xlabel('Taxonomic Level', fontsize=12)
    ax1.set_ylim(0, 100)
    
    # Add value labels on bars
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Taxa retention rates
    ax2 = axes[0, 1]
    bars2 = ax2.bar(summary_df['Taxonomic_Level'], summary_df['Taxa_Retention_%'])
    ax2.set_title('Taxonomic Diversity Retention', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Retention Percentage (%)', fontsize=12)
    ax2.set_xlabel('Taxonomic Level', fontsize=12)
    ax2.set_ylim(0, 100)
    
    # Add value labels on bars
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Absolute numbers comparison
    ax3 = axes[1, 0]
    x = np.arange(len(summary_df))
    width = 0.35
    
    bars3a = ax3.bar(x - width/2, summary_df['Original_Genomes'], width, label='Original', alpha=0.8)
    bars3b = ax3.bar(x + width/2, summary_df['Filtered_Genomes'], width, label='Filtered', alpha=0.8)
    
    ax3.set_title('Genome Counts: Original vs Filtered', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Number of Genomes', fontsize=12)
    ax3.set_xlabel('Taxonomic Level', fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(summary_df['Taxonomic_Level'])
    ax3.legend()
    
    # Taxa diversity comparison
    ax4 = axes[1, 1]
    bars4a = ax4.bar(x - width/2, summary_df['Original_Taxa'], width, label='Original', alpha=0.8)
    bars4b = ax4.bar(x + width/2, summary_df['Filtered_Taxa'], width, label='Filtered', alpha=0.8)
    
    ax4.set_title('Taxonomic Diversity: Original vs Filtered', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Number of Unique Taxa', fontsize=12)
    ax4.set_xlabel('Taxonomic Level', fontsize=12)
    ax4.set_xticks(x)
    ax4.set_xticklabels(summary_df['Taxonomic_Level'])
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'taxonomic_retention_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print(f"Summary plot saved to: {os.path.join(output_dir, 'taxonomic_retention_summary.png')}")
    
    # Figure 2: Detailed phylum-level analysis (most important for fungi)
    phylum_data = detailed_df[detailed_df['Taxonomic_Level'] == 'Phylum'].copy()
    if not phylum_data.empty:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Sort by original count for better visualization
        phylum_data = phylum_data.sort_values('Original_Count', ascending=False)
        
        # Retention rates by phylum
        colors = ['red' if status == 'Lost' else 'green' for status in phylum_data['Status']]
        bars = ax1.barh(phylum_data['Taxon'], phylum_data['Retention_%'], color=colors, alpha=0.7)
        ax1.set_title('Phylum Retention Rates', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Retention Percentage (%)', fontsize=12)
        ax1.set_ylabel('Phylum', fontsize=12)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width + 1, bar.get_y() + bar.get_height()/2,
                    f'{width:.1f}%', ha='left', va='center', fontsize=9)
        
        # Absolute counts comparison
        x_pos = np.arange(len(phylum_data))
        ax2.barh(x_pos - 0.2, phylum_data['Original_Count'], 0.4, label='Original', alpha=0.8)
        ax2.barh(x_pos + 0.2, phylum_data['Filtered_Count'], 0.4, label='Filtered', alpha=0.8)
        ax2.set_title('Phylum Genome Counts', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Number of Genomes', fontsize=12)
        ax2.set_ylabel('Phylum', fontsize=12)
        ax2.set_yticks(x_pos)
        ax2.set_yticklabels(phylum_data['Taxon'])
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'phylum_detailed_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        print(f"Phylum analysis plot saved to: {os.path.join(output_dir, 'phylum_detailed_analysis.png')}")
    else:
        print("No phylum data found for detailed analysis")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze taxonomic filtering effects between datasets.')
    parser.add_argument('--input_dir', required=True, help='Input directory containing datasets and taxonomy file')
    parser.add_argument('--output_dir', required=True, help='Directory to write output results and plots')
    parser.add_argument('--taxonomy_file', default=None, help='Path to taxonomy file (default: <input_dir>/assembly_summary.txt)')
    parser.add_argument('--datasets', nargs='*', help='Names of datasets to compare (omit to auto-detect)')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    taxonomy_file = args.taxonomy_file if args.taxonomy_file else os.path.join(input_dir, 'assembly_summary.txt')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.datasets and len(args.datasets) > 0:
        dataset_names = args.datasets
    else:
        fasta_files = [f for f in os.listdir(input_dir) if f.endswith('.fasta')]
        dataset_names = [f.replace('.fasta', '') for f in fasta_files]
        dataset_names.sort()

    print(f"Found datasets: {dataset_names}")
    if len(dataset_names) < 2:
        print("Error: Need at least 2 datasets for comparison")
        return

    # Pass taxonomy_file to all relevant functions via function attribute
    analyze_taxonomic_composition.taxonomy_file = taxonomy_file
    create_multi_dataset_comparison.taxonomy_file = taxonomy_file
    create_pairwise_comparisons.taxonomy_file = taxonomy_file

    if len(dataset_names) == 2:
        print(f"Performing pairwise comparison: {dataset_names[0]} vs {dataset_names[1]}")
        summary_df, detailed_df = analyze_taxonomic_composition(
            input_dir, dataset_names[0], dataset_names[1], output_dir
        )
        create_visualization(summary_df, detailed_df, output_dir)
        print(f"\nAnalysis complete! Results saved to: {output_dir}")
        print(f"Summary CSV: {os.path.join(output_dir, 'taxonomic_retention_summary.csv')}")
        print(f"Detailed CSV: {os.path.join(output_dir, 'detailed_taxonomic_analysis.csv')}")
    else:
        print(f"Performing multi-dataset comparison across {len(dataset_names)} datasets")
        create_multi_dataset_comparison(input_dir, dataset_names, output_dir)
        create_pairwise_comparisons(input_dir, dataset_names, output_dir)
        print(f"\nComprehensive analysis complete! Results saved to: {output_dir}")
        print(f"Multi-dataset visualization: {os.path.join(output_dir, 'multi_dataset_comparison.png')}")
        print(f"Pairwise comparisons in: {os.path.join(output_dir, 'pairwise_comparisons/')}")

if __name__ == "__main__":
    main()
