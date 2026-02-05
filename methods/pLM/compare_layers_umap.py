#!/usr/bin/env python3
"""
Compare UMAP visualizations across different ESM layers for a single gene.
"""

import matplotlib.pyplot as plt
import sys
import os
import glob
import re
import json
import csv
from collections import Counter
import numpy as np
import argparse
import hdbscan
from scipy.spatial import ConvexHull
from matplotlib.patches import Ellipse
import importlib.util
import pathlib

# Dynamically import plot_umap from the same directory or user-specified path
def import_plot_umap_module(custom_path=None):
    if custom_path:
        plot_umap_path = pathlib.Path(custom_path)
    else:
        plot_umap_path = pathlib.Path(__file__).parent / 'plot_umap.py'
    spec = importlib.util.spec_from_file_location('plot_umap', str(plot_umap_path))
    plot_umap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plot_umap)
    return plot_umap
from plot_umap import process_gene_data, PHYLUM_COLORS, add_organism_labels



# Define lifestyle colors for visualization
LIFESTYLE_COLORS = {
    'plant_pathogen': '#d62728',           # Red
    'soil_saprotroph': '#8c564b',          # Brown 
    'wood_saprotroph': '#ff7f0e',          # Orange
    'nectar/tap_saprotroph': '#ffbb78',    # Light orange
    'animal_parasite': '#9467bd',          # Purple
    'animal_endosymbiont': '#c5b0d5',      # Light purple
    'lichenized': '#2ca02c',               # Green
    'arbuscular_mycorrhizal': '#98df8a',   # Light green
    'ectomycorrhizal': '#1f77b4',          # Blue
    'animal_pathogen': '#17becf',          # Cyan
    'unspecified_saprotroph': '#7f7f7f',   # Gray
    'endophyte': '#bcbd22',                # Olive
    'Unknown': '#c7c7c7',                  # Light gray
    '': '#e377c2'                          # Pink for empty values
}

# Define keyword-based lifestyle categories
LIFESTYLE_KEYWORD_COLORS = {
    'pathogen': '#d62728',        # Red - plant_pathogen, animal_pathogen
    'saprotroph': '#8c564b',      # Brown - soil_saprotroph, wood_saprotroph, etc.
    'parasite': '#9467bd',        # Purple - animal_parasite
    'symbiont': '#c5b0d5',        # Light purple - endosymbiont
    'mycorrhizal': '#1f77b4',     # Blue - ectomycorrhizal, arbuscular_mycorrhizal
    'lichen': '#2ca02c',          # Green - lichenized
    'endophyte': '#bcbd22',       # Olive - endophyte
    'epiphyte': '#17becf',        # Cyan - epiphytic
    'mold': '#ff7f0e',            # Orange - mold-related
    'pathotroph': '#d62728',      # Red - pathotroph (same as pathogen)
    'associated': '#98df8a',      # Light green - various associations
    'Unknown': '#c7c7c7',         # Light gray
    '': '#e377c2'                 # Pink for empty values
}

# Define growth form colors
GROWTH_FORM_COLORS = {
    'yeast': '#ff7f0e',                           # Orange
    'filamentous_mycelium': '#1f77b4',            # Blue
    'dimorphic_yeast': '#d62728',                 # Red
    'thallus_photosynthetic': '#2ca02c',          # Green
    'zoosporic-rhizomycelial_(chytrid-like)': '#9467bd',  # Purple
    'crustose': '#8c564b',                        # Brown
    'foliose': '#17becf',                         # Cyan
    'fruticose': '#bcbd22',                       # Olive
    'squamulose': '#e377c2',                      # Pink
    'Unknown': '#c7c7c7',                         # Light gray
    '': '#7f7f7f'                                # Gray for empty values
}

# Define enzyme colors for pathway analysis
ENZYME_COLORS = {
    'ACO2': '#0a16aa',      # Blue
    'ARO8': '#ff3f68',      # Orange
    'LYS1': '#fff123',      # Green
    'LYS2': '#ff8845',      # Red
    'LYS4': '#910491',      # Purple
    'LYS9': '#ffc21f',      # Brown
    'LYS12': '#d50080',     # Pink
    'LYS20': '#000c7d',     # Cyan
    'Unknown': '#c7c7c7',   # Light gray
    '': '#7f7f7f'           # Gray for empty values
}

def categorize_by_keywords(lifestyle_text):
    """Categorize lifestyle by keywords."""
    import pandas as pd
    
    if not lifestyle_text or pd.isna(lifestyle_text):
        return 'Unknown'
    
    lifestyle_lower = str(lifestyle_text).lower()
    
    # Check for keywords in order of priority
    if 'pathogen' in lifestyle_lower or 'pathotroph' in lifestyle_lower:
        return 'pathogen'
    elif 'parasite' in lifestyle_lower:
        return 'parasite'
    elif 'saprotroph' in lifestyle_lower:
        return 'saprotroph'
    elif 'symbiont' in lifestyle_lower:
        return 'symbiont'
    elif 'mycorrhizal' in lifestyle_lower:
        return 'mycorrhizal'
    elif 'lichen' in lifestyle_lower:
        return 'lichen'
    elif 'endophyte' in lifestyle_lower:
        return 'endophyte'
    elif 'epiphyte' in lifestyle_lower:
        return 'epiphyte'
    elif 'mold' in lifestyle_lower:
        return 'mold'
    elif 'associated' in lifestyle_lower:
        return 'associated'
    else:
        return 'Unknown'


def extract_enzyme_from_id(sequence_id):
    """
    Extract enzyme name from sequence ID.
    
    Expected format: >ENZYME_GCA_XXXXXX or ENZYME_GCA_XXXXXX
    Examples: >ACO2_GCA_001445595.3, ARO8_GCA_001445595.3
    
    Args:
        sequence_id: The sequence identifier
        
    Returns:
        str: The enzyme name (e.g., 'ACO2', 'ARO8') or 'Unknown'
    """
    try:
        # Remove '>' if present
        clean_id = sequence_id.lstrip('>')
        
        # Split by underscore and take the first part
        parts = clean_id.split('_')
        
        if len(parts) >= 1:
            enzyme_name = parts[0].upper()
            
            # Check if it's a known enzyme pattern or looks like an enzyme name
            if enzyme_name and len(enzyme_name) >= 3:
                # Check for common patterns: letters + numbers (like LYS1, ACO2, ARO8)
                # or just letters (like ACO, ARO)
                if any(c.isalpha() for c in enzyme_name) and len(enzyme_name) <= 10:
                    return enzyme_name
        
        return 'Unknown'
        
    except Exception as e:
        print(f"Error extracting enzyme from ID '{sequence_id}': {e}")
        return 'Unknown'


def generate_enzyme_colors(protein_ids):
    """
    Generate a color mapping for enzymes based on the actual enzyme names found in the data.
    
    Args:
        protein_ids: List of protein IDs
        
    Returns:
        dict: Mapping of enzyme names to colors
    """
    import matplotlib.pyplot as plt
    
    # Extract all unique enzyme names
    enzyme_names = set()
    for pid in protein_ids:
        enzyme = extract_enzyme_from_id(pid)
        enzyme_names.add(enzyme)
    
    # Remove 'Unknown' for color assignment, will add it back later
    enzyme_names.discard('Unknown')
    enzyme_names = sorted(list(enzyme_names))
    
    # Use a colormap to generate distinct colors
    if len(enzyme_names) <= 10:
        # Use the tab10 colormap for up to 10 enzymes
        colors = plt.cm.tab10(np.linspace(0, 1, len(enzyme_names)))
    elif len(enzyme_names) <= 20:
        # Use tab20 for up to 20 enzymes
        colors = plt.cm.tab20(np.linspace(0, 1, len(enzyme_names)))
    else:
        # Use hsv colormap for many enzymes
        colors = plt.cm.hsv(np.linspace(0, 1, len(enzyme_names)))
    
    # Create the color dictionary
    enzyme_color_dict = {}
    for i, enzyme in enumerate(enzyme_names):
        enzyme_color_dict[enzyme] = colors[i]
    
    # Add colors for Unknown and empty values
    enzyme_color_dict['Unknown'] = '#c7c7c7'  # Light gray
    enzyme_color_dict[''] = '#7f7f7f'         # Gray
    
    return enzyme_color_dict


def load_lifestyle_info(traits_file):
    """Load lifestyle information from the traits CSV file."""
    import pandas as pd
    
    try:
        df = pd.read_csv(traits_file)
        
        # Create a dictionary mapping accession to lifestyle
        lifestyle_dict = {}
        for _, row in df.iterrows():
            accession = row['Accession']
            lifestyle = row.get('primary_lifestyle', '')
            if pd.isna(lifestyle) or lifestyle is None:
                lifestyle = ''
            lifestyle_dict[accession] = str(lifestyle)
        
        print(f"Loaded lifestyle info for {len(lifestyle_dict)} organisms")
        
        # Print unique lifestyles found
        unique_lifestyles = set(lifestyle_dict.values())
        print(f"Unique lifestyles found: {sorted(unique_lifestyles)}")
        
        return lifestyle_dict
        
    except Exception as e:
        print(f"Error loading lifestyle info: {e}")
        return {}


def load_growth_form_info(traits_file):
    """Load growth form information from the traits CSV file."""
    import pandas as pd
    
    try:
        df = pd.read_csv(traits_file)
        
        # Create a dictionary mapping accession to growth form
        growth_form_dict = {}
        for _, row in df.iterrows():
            accession = row['Accession']
            growth_form = row.get('Growth_form_template', '')
            if pd.isna(growth_form) or growth_form is None:
                growth_form = ''
            growth_form_dict[accession] = str(growth_form)
        
        print(f"Loaded growth form info for {len(growth_form_dict)} organisms")
        
        # Print unique growth forms found
        unique_growth_forms = set(growth_form_dict.values())
        print(f"Unique growth forms found: {sorted(unique_growth_forms)}")
        
        return growth_form_dict
        
    except Exception as e:
        print(f"Error loading growth form info: {e}")
        return {}


def load_taxa_info(traits_file):
    """Load taxa information from the traits CSV file."""
    import pandas as pd
    
    try:
        df = pd.read_csv(traits_file)
        
        # Create a dictionary mapping accession to phylum
        taxa_dict = {}
        for _, row in df.iterrows():
            accession = row['Accession']
            phylum = row.get('Phylum', 'Unknown')
            if pd.isna(phylum) or phylum is None:
                phylum = 'Unknown'
            taxa_dict[accession] = str(phylum)
        
        print(f"Loaded taxa info for {len(taxa_dict)} organisms")
        return taxa_dict
        
    except Exception as e:
        print(f"Error loading taxa info: {e}")
        return {}


_CLASS_CACHE = None
_ORDER_CACHE = None
_GENUS_CACHE = None


def load_class_info(traits_file):
    """Load class information from the traits CSV file."""
    import pandas as pd
    global _CLASS_CACHE

    if _CLASS_CACHE is not None and _CLASS_CACHE.get('file') == traits_file:
        return _CLASS_CACHE.get('data', {})

    try:
        df = pd.read_csv(traits_file)

        class_dict = {}
        for _, row in df.iterrows():
            accession = row['Accession']
            class_val = row.get('Class', 'Unknown')
            if pd.isna(class_val) or class_val is None:
                class_val = 'Unknown'
            class_dict[accession] = str(class_val)

        _CLASS_CACHE = {'file': traits_file, 'data': class_dict}
        print(f"Loaded class info for {len(class_dict)} organisms")
        return class_dict

    except Exception as e:
        print(f"Error loading class info: {e}")
        return {}


def load_order_info(traits_file):
    """Load order information from the traits CSV file."""
    import pandas as pd
    global _ORDER_CACHE

    if _ORDER_CACHE is not None and _ORDER_CACHE.get('file') == traits_file:
        return _ORDER_CACHE.get('data', {})

    try:
        df = pd.read_csv(traits_file)

        order_dict = {}
        for _, row in df.iterrows():
            accession = row['Accession']
            order_val = row.get('Order', 'Unknown')
            if pd.isna(order_val) or order_val is None:
                order_val = 'Unknown'
            order_dict[accession] = str(order_val)

        _ORDER_CACHE = {'file': traits_file, 'data': order_dict}
        print(f"Loaded order info for {len(order_dict)} organisms")
        return order_dict

    except Exception as e:
        print(f"Error loading order info: {e}")
        return {}


def load_genus_info(traits_file):
    """Load genus information from the traits CSV file."""
    import pandas as pd
    global _GENUS_CACHE

    if _GENUS_CACHE is not None and _GENUS_CACHE.get('file') == traits_file:
        return _GENUS_CACHE.get('data', {})

    try:
        df = pd.read_csv(traits_file)

        genus_dict = {}
        for _, row in df.iterrows():
            accession = row['Accession']
            genus_val = row.get('Genus', 'Unknown')
            if pd.isna(genus_val) or genus_val is None:
                genus_val = 'Unknown'
            genus_dict[accession] = str(genus_val)

        _GENUS_CACHE = {'file': traits_file, 'data': genus_dict}
        print(f"Loaded genus info for {len(genus_dict)} organisms")
        return genus_dict

    except Exception as e:
        print(f"Error loading genus info: {e}")
        return {}


def map_accessions_to_labels(protein_ids, data_dict):
    """Map protein IDs to labels (phylum/class) handling versions flexibly."""
    labels = []

    def find_matching_accession(target_acc, data_dict_inner):
        if target_acc in data_dict_inner:
            return target_acc, data_dict_inner[target_acc]
        for key in data_dict_inner.keys():
            if key.startswith(target_acc + '.'):
                return key, data_dict_inner[key]
        base_acc = target_acc.split('.')[0]
        if base_acc in data_dict_inner:
            return base_acc, data_dict_inner[base_acc]
        return None, None

    for acc in protein_ids:
        parts = acc.split('_')
        assembly_acc = acc
        if len(parts) >= 2:
            assembly_acc = parts[0] + '_' + parts[1]

        matched_key, label = find_matching_accession(assembly_acc, data_dict)
        if matched_key:
            labels.append(label if label else 'Unknown')
            continue

        matched_key, label = find_matching_accession(acc, data_dict)
        if matched_key:
            labels.append(label if label else 'Unknown')
        else:
            labels.append('Unknown')

    return labels


def get_fasta_accessions(fasta_file):
    """Get accession IDs from FASTA file."""
    from Bio import SeqIO
    
    accessions = []
    try:
        for record in SeqIO.parse(fasta_file, "fasta"):
            accessions.append(record.id)
        return accessions
    except Exception as e:
        print(f"Error reading FASTA file: {e}")
        return []


def calculate_phylum_diversity(embeddings, categories):
    """Calculate diversity metrics for the embeddings."""
    from sklearn.metrics.pairwise import euclidean_distances
    
    try:
        # Calculate pairwise distances
        distances = euclidean_distances(embeddings)
        
        # Get upper triangle (excluding diagonal)
        upper_tri_indices = np.triu_indices_from(distances, k=1)
        distances_flat = distances[upper_tri_indices]
        
        # Calculate basic statistics
        diversity_metrics = {
            'mean_distance': np.mean(distances_flat),
            'std_distance': np.std(distances_flat),
            'median_distance': np.median(distances_flat),
            'max_distance': np.max(distances_flat),
            'min_distance': np.min(distances_flat)
        }
        
        return diversity_metrics
        
    except Exception as e:
        print(f"Error calculating diversity metrics: {e}")
        return {'mean_distance': 0, 'std_distance': 0}


def calculate_separation_metrics(embeddings, categories):
    """
    Calculate various metrics to quantify how well categories are separated.
    
    Returns:
        dict: Dictionary containing separation quality metrics
    """
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics.pairwise import euclidean_distances
    import numpy as np
    from collections import Counter
    
    try:
        # Convert categories to numeric labels
        le = LabelEncoder()
        numeric_categories = le.fit_transform(categories)
        unique_categories = le.classes_
        n_categories = len(unique_categories)
        
        # Skip if only one category or too few samples
        if n_categories < 2 or len(embeddings) < 4:
            return {'error': 'Insufficient categories or samples for separation metrics'}
        
        metrics = {}
        
        # 1. Silhouette Score (higher is better, range: -1 to 1)
        # Measures how well-separated the clusters are
        try:
            silhouette = silhouette_score(embeddings, numeric_categories)
            metrics['silhouette_score'] = silhouette
        except:
            metrics['silhouette_score'] = None
        
        # 2. Calinski-Harabasz Index (higher is better)
        # Ratio of between-cluster dispersion to within-cluster dispersion
        try:
            ch_score = calinski_harabasz_score(embeddings, numeric_categories)
            metrics['calinski_harabasz_score'] = ch_score
        except:
            metrics['calinski_harabasz_score'] = None
        
        # 3. Davies-Bouldin Index (lower is better)
        # Average similarity ratio of each cluster with its most similar cluster
        try:
            db_score = davies_bouldin_score(embeddings, numeric_categories)
            metrics['davies_bouldin_score'] = db_score
        except:
            metrics['davies_bouldin_score'] = None
        
        # 4. Inter vs Intra-cluster distances
        distances = euclidean_distances(embeddings)
        
        # Calculate within-cluster (intra) distances
        intra_distances = []
        for cat in unique_categories:
            cat_mask = np.array(categories) == cat
            cat_indices = np.where(cat_mask)[0]
            if len(cat_indices) > 1:
                cat_distances = distances[np.ix_(cat_indices, cat_indices)]
                # Get upper triangle (excluding diagonal)
                upper_tri = np.triu_indices_from(cat_distances, k=1)
                intra_distances.extend(cat_distances[upper_tri])
        
        # Calculate between-cluster (inter) distances
        inter_distances = []
        for i, cat1 in enumerate(unique_categories):
            for cat2 in unique_categories[i+1:]:
                cat1_mask = np.array(categories) == cat1
                cat2_mask = np.array(categories) == cat2
                cat1_indices = np.where(cat1_mask)[0]
                cat2_indices = np.where(cat2_mask)[0]
                if len(cat1_indices) > 0 and len(cat2_indices) > 0:
                    between_distances = distances[np.ix_(cat1_indices, cat2_indices)]
                    inter_distances.extend(between_distances.flatten())
        
        if intra_distances and inter_distances:
            metrics['mean_intra_distance'] = np.mean(intra_distances)
            metrics['mean_inter_distance'] = np.mean(inter_distances)
            metrics['separation_ratio'] = np.mean(inter_distances) / np.mean(intra_distances)
        else:
            metrics['mean_intra_distance'] = None
            metrics['mean_inter_distance'] = None
            metrics['separation_ratio'] = None
        
        # 5. Category distribution balance (entropy-based)
        category_counts = Counter(categories)
        total_samples = len(categories)
        probabilities = [count/total_samples for count in category_counts.values()]
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        max_entropy = np.log2(n_categories)
        metrics['category_balance'] = entropy / max_entropy if max_entropy > 0 else 0
        
        return metrics
        
    except Exception as e:
        print(f"Error calculating separation metrics: {e}")
        return {'error': str(e)}


def compute_cluster_category_percentages(labels, categories):
    """Summarize how categories distribute across HDBSCAN clusters."""
    from collections import Counter

    labels_arr = np.array(labels)
    categories_arr = np.array(categories)
    total = len(labels_arr)
    stats = {}

    if total == 0:
        return stats

    for cluster_id in sorted(set(labels_arr)):
        mask = labels_arr == cluster_id
        cluster_size = int(mask.sum())
        if cluster_size == 0:
            continue

        cat_counts = Counter(categories_arr[mask])
        cat_percentages = {cat: (count / cluster_size) * 100 for cat, count in cat_counts.items()}
        stats[int(cluster_id)] = {
            'size': cluster_size,
            'percent_of_total': (cluster_size / total) * 100,
            'category_percentages': cat_percentages,
        }

    return stats


def compute_cluster_nmi(cluster_labels, categories):
    """Compute normalized mutual information between clusters and categories."""
    from sklearn.metrics import normalized_mutual_info_score

    if len(set(cluster_labels)) < 2 or len(set(categories)) < 2:
        return None
    try:
        return normalized_mutual_info_score(categories, cluster_labels)
    except Exception:
        return None


def summarize_phylum_clusters(cluster_labels, categories, cluster_stats, dominance_threshold=80.0):
    """Summaries focused on phylum composition for each cluster."""
    total_points = len(cluster_labels)
    dominated = []  # (cid, top_phylum, top_pct, size)

    # Identify dominated/pure clusters
    for cid, stat in cluster_stats.items():
        if cid == -1:
            continue
        cat_pct = stat.get('category_percentages', {})
        if not cat_pct:
            continue
        top_phylum, top_pct = max(cat_pct.items(), key=lambda kv: kv[1])
        if top_pct >= dominance_threshold:
            dominated.append((cid, top_phylum, top_pct, stat.get('size', 0)))

    # Fractions of interest
    def frac_for_condition(condition_fn):
        size = 0
        for cid, stat in cluster_stats.items():
            if cid == -1:
                continue
            cat_pct = stat.get('category_percentages', {})
            if condition_fn(cat_pct):
                size += stat.get('size', 0)
        return (size / total_points * 100) if total_points else 0

    # Ascomycota-heavy = cluster with max Asco %
    asco_best_size = 0
    asco_best_pct = 0
    for cid, stat in cluster_stats.items():
        if cid == -1:
            continue
        cat_pct = stat.get('category_percentages', {})
        asco_pct = cat_pct.get('Ascomycota', 0)
        if asco_pct > asco_best_pct:
            asco_best_pct = asco_pct
            asco_best_size = stat.get('size', 0)
    asco_heavy_fraction = (asco_best_size / total_points * 100) if total_points else 0

    def only_phylum(name):
        def _cond(cat_pct):
            total_other = sum(v for k, v in cat_pct.items() if k != name)
            return cat_pct.get(name, 0) > 0 and total_other < 1e-6
        return _cond

    basidio_only_fraction = frac_for_condition(only_phylum('Basidiomycota'))
    zoopa_only_fraction = frac_for_condition(only_phylum('Zoopagomycota'))
    muco_only_fraction = frac_for_condition(only_phylum('Mucoromycota'))

    noise_size = cluster_stats.get(-1, {}).get('size', 0)
    noise_fraction = (noise_size / total_points * 100) if total_points else 0

    return {
        'dominated_clusters': dominated,
        'asco_heavy_fraction': asco_heavy_fraction,
        'basidio_only_fraction': basidio_only_fraction,
        'zoopa_only_fraction': zoopa_only_fraction,
        'muco_only_fraction': muco_only_fraction,
        'noise_fraction': noise_fraction,
    }


def compute_enzyme_silhouette(embeddings, enzyme_labels):
    """Compute overall and per-enzyme silhouette scores in embedding space."""
    from sklearn.metrics import silhouette_score, silhouette_samples
    if embeddings is None or len(embeddings) < 3:
        return None, None
    unique = set(enzyme_labels)
    if len(unique) < 2:
        return None, None
    try:
        overall = silhouette_score(embeddings, enzyme_labels, metric='euclidean')
        per_point = silhouette_samples(embeddings, enzyme_labels, metric='euclidean')
        per_enzyme = {}
        labels_arr = np.array(enzyme_labels)
        for enz in unique:
            mask = labels_arr == enz
            if mask.sum() > 0:
                per_enzyme[enz] = float(per_point[mask].mean())
        return float(overall), per_enzyme
    except Exception:
        return None, None


def run_hdbscan_on_umap(umap_coords, categories, min_cluster_size=50, min_samples=5, metric='cosine'):
    """Run HDBSCAN on UMAP coordinates and summarize category composition per cluster."""
    try:
        if metric == 'cosine':
            # Precompute cosine distances for environments where hdbscan lacks native cosine support
            from sklearn.metrics import pairwise_distances

            distance_matrix = pairwise_distances(umap_coords, metric='cosine')
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='precomputed',
            )
            labels = clusterer.fit_predict(distance_matrix)
        else:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric=metric,
            )
            labels = clusterer.fit_predict(umap_coords)
    except Exception as e:
        print(f"HDBSCAN on UMAP failed with metric '{metric}': {e}")
        raise
    stats = compute_cluster_category_percentages(labels, categories)
    return labels, stats


def run_hdbscan_on_embeddings(embeddings, categories, min_cluster_size=50, min_samples=5, metric='cosine'):
    """Run HDBSCAN on high-dimensional embeddings and summarize category composition."""
    try:
        if metric == 'cosine':
            from sklearn.metrics import pairwise_distances

            distance_matrix = pairwise_distances(embeddings, metric='cosine')
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='precomputed',
            )
            labels = clusterer.fit_predict(distance_matrix)
        else:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric=metric,
            )
            labels = clusterer.fit_predict(embeddings)
    except Exception as e:
        print(f"HDBSCAN on embeddings failed with metric '{metric}': {e}")
        raise
    stats = compute_cluster_category_percentages(labels, categories)
    return labels, stats


def smooth_polygon(points, iterations=2):
    """Chaikin-style corner cutting to smooth polygon edges."""
    pts = np.asarray(points)
    if pts.shape[0] < 3:
        return pts
    for _ in range(iterations):
        new_pts = []
        for i in range(len(pts)):
            p1 = pts[i]
            p2 = pts[(i + 1) % len(pts)]
            q = 0.75 * p1 + 0.25 * p2
            r = 0.25 * p1 + 0.75 * p2
            new_pts.extend([q, r])
        pts = np.array(new_pts)
    return pts


def cluster_to_ellipse(cluster_points, scale=2.5):
    """Compute ellipse parameters (width, height, angle, center) from points."""
    if cluster_points.shape[0] < 2:
        return None
    center = cluster_points.mean(axis=0)
    cov = np.cov(cluster_points.T)
    vals, vecs = np.linalg.eigh(cov)
    # Sort descending
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    # Width/height scaled by chosen std multiplier
    width, height = 2 * scale * np.sqrt(np.maximum(vals, 1e-9))
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    return width, height, angle, center


def process_gene_data_custom(gene_name, layer, base_dir, color_by='phylum'):
    """Custom processing function that can handle different base directories."""
    # Import necessary functions from plot_umap
    from plot_umap import load_embeddings, load_taxa_info, get_accessions_from_id_file, calculate_phylum_diversity
    from umap import UMAP
    from sklearn.metrics.pairwise import cosine_similarity
    import pandas as pd
    
    # Construct file paths
    embedding_file = os.path.join(base_dir, f"{gene_name}_layer_{layer}_embeddings.npy")
    ids_file = os.path.join(base_dir, f"{gene_name}_layer_{layer}_ids.txt")
    traits_file = process_gene_data_custom.traits_file
    id_file = os.path.join(base_dir, f"{gene_name}_layer_{layer}_ids.txt")
    
    # Check if required files exist
    if not os.path.exists(embedding_file):
        print(f"Warning: Embedding file not found: {embedding_file}")
        return None
    if not os.path.exists(ids_file):
        print(f"Warning: IDs file not found: {ids_file}")
        return None
    if not os.path.exists(id_file):
        print(f"Warning: ID file not found: {id_file}")
        return None
    if not os.path.exists(traits_file):
        print(f"Warning: Traits file not found: {traits_file}")
        return None
    
    try:
        # Load data
        embeddings = np.load(embedding_file)
        
        # Load protein IDs
        with open(ids_file, 'r') as f:
            protein_ids = [line.strip() for line in f]
        
        # Load data based on color_by parameter
        if color_by == 'lifestyle':
            # Load lifestyle data
            lifestyle_dict = load_lifestyle_info(traits_file)
            print(f"Lifestyle dictionary loaded with {len(lifestyle_dict)} entries")
        elif color_by == 'lifestyle_keywords':
            # Load lifestyle data for keyword processing
            lifestyle_dict = load_lifestyle_info(traits_file)
            print(f"Lifestyle dictionary loaded with {len(lifestyle_dict)} entries")
        elif color_by == 'growth_form':
            # Load growth form data
            growth_form_dict = load_growth_form_info(traits_file)
            print(f"Growth form dictionary loaded with {len(growth_form_dict)} entries")
        elif color_by == 'enzyme':
            # For enzyme coloring, we don't need to load external data
            print(f"Using enzyme-based coloring from sequence IDs")
            # Generate dynamic color mapping based on actual enzyme names in the data
            enzyme_color_dict = generate_enzyme_colors(protein_ids)
            print(f"Generated colors for {len(enzyme_color_dict)-2} unique enzymes: {list(enzyme_color_dict.keys())}")
        else:
            # Load phylogeny data using the correct function name
            taxa_dict = load_taxa_info(traits_file)
            print(f"Taxa dictionary loaded with {len(taxa_dict)} entries")
        
        print(f"Number of protein IDs in embeddings: {len(protein_ids)}")
        print(f"Number of embeddings: {embeddings.shape[0]}")

        # Verify that embeddings and protein_ids match
        if len(protein_ids) != embeddings.shape[0]:
            print(f"ERROR: Mismatch between protein IDs ({len(protein_ids)}) and embeddings ({embeddings.shape[0]})")
            return None

        print(f"First 5 protein IDs from embeddings: {protein_ids[:5]}")

        # Match data to embeddings - iterate through protein_ids
        categories = []
        matched_count = 0
        for i, accession in enumerate(protein_ids):
            # Extract the assembly accession (before the underscore)
            assembly_acc = accession.split('_')[0] + '_' + accession.split('_')[1]  # Get GCA_XXXXXX or GCF_XXXXXX format

            # Debug: print first few matches
            if i < 5:
                print(f"Protein accession {i}: {accession} -> assembly_acc: {assembly_acc}")

            # Function to find matching accession in dictionary (with or without version)
            def find_matching_accession(target_acc, data_dict):
                # First try exact match
                if target_acc in data_dict:
                    return target_acc, data_dict[target_acc]

                # Try with different version numbers
                for key in data_dict.keys():
                    if key.startswith(target_acc + '.'):
                        return key, data_dict[key]

                # Try without version (in case accession has version but CSV doesn't)
                base_acc = target_acc.split('.')[0]
                if base_acc in data_dict:
                    return base_acc, data_dict[base_acc]

                return None, None

            if color_by == 'lifestyle':
                # Look for matching entry in lifestyle_dict
                matched_key, category = find_matching_accession(assembly_acc, lifestyle_dict)
                if matched_key:
                    categories.append(category if category else 'Unknown')
                    matched_count += 1
                    if i < 5:
                        print(f"  -> Found lifestyle: '{category}' (key: {matched_key})")
                else:
                    # Try full accession
                    matched_key, category = find_matching_accession(accession, lifestyle_dict)
                    if matched_key:
                        categories.append(category if category else 'Unknown')
                        matched_count += 1
                        if i < 5:
                            print(f"  -> Found lifestyle (full accession): '{category}' (key: {matched_key})")
                    else:
                        categories.append('Unknown')
                        if i < 5:
                            print(f"  -> Not found, using 'Unknown'")
            elif color_by == 'lifestyle_keywords':
                # Look for matching entry and categorize by keywords
                matched_key, lifestyle_text = find_matching_accession(assembly_acc, lifestyle_dict)
                if matched_key:
                    category = categorize_by_keywords(lifestyle_text)
                    categories.append(category)
                    matched_count += 1
                    if i < 5:
                        print(f"  -> Found lifestyle: '{lifestyle_text}' -> keyword: '{category}' (key: {matched_key})")
                else:
                    # Try full accession
                    matched_key, lifestyle_text = find_matching_accession(accession, lifestyle_dict)
                    if matched_key:
                        category = categorize_by_keywords(lifestyle_text)
                        categories.append(category)
                        matched_count += 1
                        if i < 5:
                            print(f"  -> Found lifestyle (full accession): '{lifestyle_text}' -> keyword: '{category}' (key: {matched_key})")
                    else:
                        categories.append('Unknown')
                        if i < 5:
                            print(f"  -> Not found, using 'Unknown'")
            elif color_by == 'growth_form':
                # Look for matching entry in growth_form_dict
                matched_key, category = find_matching_accession(assembly_acc, growth_form_dict)
                if matched_key:
                    categories.append(category if category else 'Unknown')
                    matched_count += 1
                else:
                    # Try full accession
                    matched_key, category = find_matching_accession(accession, growth_form_dict)
                    if matched_key:
                        categories.append(category if category else 'Unknown')
                        matched_count += 1
                    else:
                        categories.append('Unknown')
            elif color_by == 'enzyme':
                # Extract enzyme name from sequence ID
                enzyme_name = extract_enzyme_from_id(accession)
                categories.append(enzyme_name)
                matched_count += 1
                if i < 5:
                    print(f"  -> Extracted enzyme: '{enzyme_name}' from ID: '{accession}'")
            else:
                    # Look for matching entry in taxa_dict (phylum)
                    matched_key, phylum = find_matching_accession(assembly_acc, taxa_dict)
                    if matched_key:
                        # Handle NaN values in phylum data
                        if phylum is None or (isinstance(phylum, float) and np.isnan(phylum)):
                            categories.append('Unknown')
                        else:
                            categories.append(str(phylum))
                        matched_count += 1
                    else:
                        # Try full accession
                        matched_key, phylum = find_matching_accession(accession, taxa_dict)
                        if matched_key:
                            # Handle NaN values in phylum data  
                            if phylum is None or (isinstance(phylum, float) and np.isnan(phylum)):
                                categories.append('Unknown')
                            else:
                                categories.append(str(phylum))
                            matched_count += 1
                        else:
                            categories.append('Unknown')
        
        print(f"Successfully matched {matched_count}/{len(protein_ids)} protein IDs")
        
        # Print category distribution
        from collections import Counter
        category_counts = Counter(categories)
        print(f"Category distribution: {dict(category_counts)}")
        
        # Debug: show some CSV accessions
        if color_by == 'lifestyle_keywords':
            sample_csv_keys = list(lifestyle_dict.keys())[:5]
            print(f"Sample CSV accessions: {sample_csv_keys}")
            sample_values = [lifestyle_dict[k] for k in sample_csv_keys]
            print(f"Sample lifestyle values: {sample_values}")
        elif color_by == 'growth_form':
            sample_csv_keys = list(growth_form_dict.keys())[:5]
            print(f"Sample CSV accessions: {sample_csv_keys}")
            sample_values = [growth_form_dict[k] for k in sample_csv_keys]
            print(f"Sample growth form values: {sample_values}")
        elif color_by == 'lifestyle':
            sample_csv_keys = list(lifestyle_dict.keys())[:5]
            print(f"Sample CSV accessions: {sample_csv_keys}")
            sample_values = [lifestyle_dict[k] for k in sample_csv_keys]
            print(f"Sample lifestyle values: {sample_values}")
        elif color_by == 'enzyme':
            # Show sample enzyme extractions
            sample_ids = protein_ids[:5]
            sample_enzymes = [extract_enzyme_from_id(pid) for pid in sample_ids]
            print(f"Sample protein IDs: {sample_ids}")
            print(f"Sample enzyme names: {sample_enzymes}")
        else:
            sample_csv_keys = list(taxa_dict.keys())[:5]
            print(f"Sample CSV accessions: {sample_csv_keys}")
            sample_values = [taxa_dict[k] for k in sample_csv_keys]
            print(f"Sample phylum values: {sample_values}")
        
        # Compute UMAP
        umap_model = UMAP(
            n_neighbors=100,
            min_dist=0.1,
            n_components=2,
            metric='cosine',
            random_state=42
        )
        umap_coords = umap_model.fit_transform(embeddings)
        
        # Compute diversity metrics
        diversity_metrics = calculate_phylum_diversity(embeddings, categories)
        
        # Compute separation metrics
        separation_metrics = calculate_separation_metrics(embeddings, categories)
        
        # Return enzyme color dict if using enzyme coloring (embeddings appended for downstream clustering)
        if color_by == 'enzyme':
            return (
                gene_name,
                umap_coords,
                categories,
                protein_ids,
                diversity_metrics,
                separation_metrics,
                embeddings,
                enzyme_color_dict,
            )
        else:
            return (
                gene_name,
                umap_coords,
                categories,
                protein_ids,
                diversity_metrics,
                separation_metrics,
                embeddings,
            )
        
    except Exception as e:
        print(f"Error processing {gene_name} layer {layer}: {e}")
        return None

def find_available_layers(gene_name, base_dir=None):
    """
    Find all available layers for a gene by scanning the embedding files.
    
    Args:
        gene_name: name of the gene
        base_dir: base directory to search, if None uses default paths
    
    Returns:
        list of available layer numbers
    """
    if base_dir is None:
        # Use current working directory if not provided
        possible_dirs = [os.getcwd()]
    else:
        possible_dirs = [base_dir]
    
    layers = set()
    
    for search_dir in possible_dirs:
        if os.path.exists(search_dir):
            print(f"Searching in directory: {search_dir}")
            # Look for files matching pattern: {gene_name}_layer_{number}_embeddings.npy
            pattern = os.path.join(search_dir, f"{gene_name}_layer_*_embeddings.npy")
            files = glob.glob(pattern)
            print(f"Found {len(files)} embedding files matching pattern")
            
            for file in files:
                # Extract layer number from filename
                match = re.search(rf"{gene_name}_layer_(\d+)_embeddings\.npy", os.path.basename(file))
                if match:
                    layer_num = int(match.group(1))
                    layers.add(layer_num)
    
    return sorted(list(layers))

def compare_layers_umap(
    gene_name,
    layers=None,
    save_path=None,
    base_dir=None,
    output_dir=None,
    max_layers=None,
    sample_layers=None,
    color_by='phylum',
    hdbscan_min_cluster_size=50,
    hdbscan_min_samples=5,
    hdbscan_metric='cosine',
    overlay_clusters=True,
    cluster_overlay_style='ellipse',
    save_svg=False,
    export_hdbscan=False,
    export_hdbscan_membership=False,
    hdbscan_export_dir=None,
    hdbscan_summary_csv=None,
):
    """
    Create side-by-side UMAP plots for different layers of the same gene.
    
    Args:
        gene_name: name of the gene
        layers: specific layers to compare, if None will auto-detect all available
        save_path: custom save path (overrides output_dir)
        base_dir: base directory for embeddings
        output_dir: output directory for plots
        max_layers: maximum number of layers to plot (for readability)
        sample_layers: if more layers than max_layers, how to sample them ('uniform', 'start_end', 'specific')
        color_by: what to color points by ('phylum', 'lifestyle', 'lifestyle_keywords', 'growth_form', or 'enzyme')
        hdbscan_min_cluster_size: HDBSCAN minimum cluster size (default: 50)
        hdbscan_min_samples: HDBSCAN min_samples; higher -> denser clusters (default: 5)
        hdbscan_metric: distance metric for HDBSCAN (default: cosine)
        overlay_clusters: whether to draw cluster outlines/labels on UMAP plots
        cluster_overlay_style: 'ellipse' for covariance ellipses, 'hull' for convex hull (default: ellipse)
        save_svg: also save an SVG version alongside PNG (default: False)
        export_hdbscan: write HDBSCAN labels/stats to disk (default: False)
        export_hdbscan_membership: also write per-cluster membership (protein IDs) TSV when exporting
        hdbscan_export_dir: directory for HDBSCAN exports (default: output_dir)
        hdbscan_summary_csv: optional path to write aggregated HDBSCAN summaries across layers (default: None)
    """
    if layers is None:
        layers = find_available_layers(gene_name, base_dir)
        if not layers:
            print(f"Error: No embedding files found for gene {gene_name}")
            if base_dir:
                print(f"Searched in: {base_dir}")
            return
        print(f"Found {len(layers)} layers: {layers}")
    
    # Handle too many layers for visualization
    original_layers = layers.copy()
    
    # Check if user explicitly requested all layers
    plot_all_layers = False
    if hasattr(compare_layers_umap, '_plot_all_requested'):
        plot_all_layers = compare_layers_umap._plot_all_requested
    
    if not plot_all_layers and max_layers and len(layers) > max_layers:
        print(f"Found {len(layers)} layers, but limiting to {max_layers} for visualization")
        
        if sample_layers == 'uniform':
            # Sample uniformly across the range
            step = len(layers) // max_layers
            layers = layers[::step][:max_layers]
        elif sample_layers == 'start_end':
            # Take first few, last few, and some middle
            n_each = max_layers // 3
            layers = layers[:n_each] + layers[len(layers)//2-n_each//2:len(layers)//2+n_each//2] + layers[-n_each:]
            layers = sorted(list(set(layers)))[:max_layers]
        elif sample_layers == 'specific':
            # Use specific meaningful layers
            meaningful_layers = [0, 2, 5, 8, 12, 16, 20, 24, 28, 32, 35]
            layers = [l for l in meaningful_layers if l in original_layers][:max_layers]
        else:
            # Default: take first max_layers
            layers = layers[:max_layers]
        
        print(f"Selected layers for plotting: {layers}")
    elif plot_all_layers:
        print(f"Plotting all {len(layers)} layers as requested")
    
    print(f"Comparing layers {layers} for gene {gene_name} (colored by {color_by})")
    
    # Choose color scheme based on color_by parameter
    if color_by == 'lifestyle':
        color_dict = LIFESTYLE_COLORS
        print(f"Using lifestyle-based coloring with {len(color_dict)} categories")
    elif color_by == 'lifestyle_keywords':
        color_dict = LIFESTYLE_KEYWORD_COLORS
        print(f"Using keyword-based lifestyle coloring with {len(color_dict)} categories")
    elif color_by == 'growth_form':
        color_dict = GROWTH_FORM_COLORS
        print(f"Using growth form-based coloring with {len(color_dict)} categories")
    elif color_by == 'enzyme':
        color_dict = ENZYME_COLORS
        print(f"Using enzyme-based coloring with {len(color_dict)} categories")
    else:
        color_dict = PHYLUM_COLORS
        print(f"Using phylum-based coloring with {len(color_dict)} categories")
    print(f"HDBSCAN metric: {hdbscan_metric}")
    
    # Process each layer
    results = {}
    summary_rows = []
    dynamic_enzyme_colors = None
    for layer in layers:
        print(f"Processing layer {layer}...")
        
        # If base_dir is provided, we need to modify the embedding file path
        if base_dir:
            # Create a custom embedding file path
            embedding_file = os.path.join(base_dir, f"{gene_name}_layer_{layer}_embeddings.npy")
            if not os.path.exists(embedding_file):
                print(f"Warning: Embedding file not found: {embedding_file}")
                continue
            
            # Call process_gene_data with custom path by temporarily modifying the function
            # For now, let's create a custom processing function
            result = process_gene_data_custom(gene_name, layer, base_dir, color_by)
        else:
            result = process_gene_data(gene_name, layer=layer)
            
        if result is not None:
            # Prefer embedding-space clustering; fall back to UMAP if embeddings are unavailable
            cluster_labels = None
            cluster_stats = None
            try:
                # Extract embeddings when present
                embeddings_idx = 6 if color_by != 'enzyme' else 6
                embeddings = result[embeddings_idx] if len(result) > embeddings_idx else None

                if embeddings is not None:
                    cluster_labels, cluster_stats = run_hdbscan_on_embeddings(
                        embeddings,
                        result[2],
                        min_cluster_size=hdbscan_min_cluster_size,
                        min_samples=hdbscan_min_samples,
                        metric=hdbscan_metric,
                    )
                else:
                    cluster_labels, cluster_stats = run_hdbscan_on_umap(
                        result[1],
                        result[2],
                        min_cluster_size=hdbscan_min_cluster_size,
                        min_samples=hdbscan_min_samples,
                        metric=hdbscan_metric,
                    )

                result = result + (cluster_labels, cluster_stats)

                # Optional secondary compositions (class/order) when coloring by phylum
                class_cluster_stats = None
                order_cluster_stats = None
                class_labels = None
                order_labels = None
                genus_labels = None
                if color_by == 'phylum':
                    try:
                        class_dict = load_class_info(TRAITS_FILE)
                        if class_dict:
                            protein_ids_for_layer = result[3] if len(result) > 3 else []
                            class_labels = map_accessions_to_labels(protein_ids_for_layer, class_dict)
                            if len(class_labels) == len(cluster_labels):
                                class_cluster_stats = compute_cluster_category_percentages(cluster_labels, class_labels)
                        order_dict = load_order_info(TRAITS_FILE)
                        if order_dict:
                            protein_ids_for_layer = result[3] if len(result) > 3 else []
                            order_labels = map_accessions_to_labels(protein_ids_for_layer, order_dict)
                            if len(order_labels) == len(cluster_labels):
                                order_cluster_stats = compute_cluster_category_percentages(cluster_labels, order_labels)
                        genus_dict = load_genus_info(TRAITS_FILE)
                        if genus_dict:
                            protein_ids_for_layer = result[3] if len(result) > 3 else []
                            genus_labels = map_accessions_to_labels(protein_ids_for_layer, genus_dict)
                    except Exception as e:
                        print(f"Warning: Failed to compute class/order composition for layer {layer}: {e}")

                # Optional export of HDBSCAN outputs
                if export_hdbscan:
                    export_dir = hdbscan_export_dir or output_dir
                    os.makedirs(export_dir, exist_ok=True)

                    # Cluster-by-category breakdown table (no JSON) with per-cluster and per-category coverage
                    comp_path = os.path.join(
                        export_dir,
                        f'hdbscan_{gene_name}_layer_{layer}_{color_by}_composition.tsv'
                    )

                    # Precompute totals for pct_of_category calculations
                    phylum_totals = Counter(result[2]) if len(result) > 2 else Counter()
                    class_totals = Counter(class_labels) if class_labels else Counter()
                    order_totals = Counter(order_labels) if order_labels else Counter()

                    with open(comp_path, 'w') as cf:
                        header_fields = [
                            'category_type',
                            'cluster_id',
                            'cluster_size',
                            'percent_of_total',
                            'category',
                            'count_in_cluster',
                            'pct_of_cluster',
                            'pct_of_category',
                        ]
                        cf.write('\t'.join(header_fields) + '\n')

                        def write_rows(cat_type, stats_obj, total_counts):
                            for cid, stat in sorted(stats_obj.items()):
                                cluster_size = stat.get('size', 0)
                                pct_total = stat.get('percent_of_total', 0)
                                for cat, pct in stat.get('category_percentages', {}).items():
                                    count_in_cluster = pct * cluster_size / 100.0
                                    total_cat = total_counts.get(cat, 0)
                                    pct_of_category = (count_in_cluster / total_cat * 100.0) if total_cat else 0.0
                                    cf.write(
                                        f"{cat_type}\t{cid}\t{cluster_size}\t{pct_total:.4f}\t{cat}\t{count_in_cluster:.2f}\t{pct:.4f}\t{pct_of_category:.4f}\n"
                                    )

                        # Always write the primary color_by breakdown
                        write_rows(color_by, cluster_stats, phylum_totals if color_by == 'phylum' else Counter())

                        # If available, also write class/order breakdowns
                        if color_by == 'phylum' and class_cluster_stats:
                            write_rows('class', class_cluster_stats, class_totals)
                        if color_by == 'phylum' and order_cluster_stats:
                            write_rows('order', order_cluster_stats, order_totals)

                    print(f"Saved HDBSCAN composition table to: {comp_path}")

                    # Optional per-point membership table
                    if export_hdbscan_membership:
                        membership_path = os.path.join(
                            export_dir,
                            f'hdbscan_{gene_name}_layer_{layer}_{color_by}_membership.tsv'
                        )
                        with open(membership_path, 'w') as mf:
                            mf.write('cluster_id\tprotein_id\tprimary_category\tclass\torder\tgenus\n')
                            for idx, pid in enumerate(result[3]):
                                cid = cluster_labels[idx] if cluster_labels is not None else ''
                                primary_cat = result[2][idx] if len(result) > 2 else ''
                                cls_val = class_labels[idx] if class_labels and len(class_labels) > idx else ''
                                ord_val = order_labels[idx] if order_labels and len(order_labels) > idx else ''
                                genus_val = genus_labels[idx] if genus_labels and len(genus_labels) > idx else ''
                                mf.write(f"{cid}\t{pid}\t{primary_cat}\t{cls_val}\t{ord_val}\t{genus_val}\n")
                        print(f"Saved HDBSCAN membership table to: {membership_path}")
            except Exception as e:
                print(f"Warning: HDBSCAN failed for layer {layer}: {e}")
            results[layer] = result
            # Collect summary row for CSV export
            try:
                separation_metrics = result[5] if len(result) > 5 else {}

                # Pull cluster labels/stats from the result tuple to avoid scope issues
                labels_idx = 8 if color_by == 'enzyme' else 7
                stats_idx = 9 if color_by == 'enzyme' else 8
                cluster_labels_for_layer = result[labels_idx] if len(result) > labels_idx else None
                cluster_stats_for_layer = result[stats_idx] if len(result) > stats_idx else None

                if cluster_labels_for_layer is None or cluster_stats_for_layer is None:
                    raise ValueError("Missing cluster labels/stats for summary export")

                total_points = len(cluster_labels_for_layer)
                noise_n = cluster_stats_for_layer.get(-1, {}).get('size', 0) if cluster_stats_for_layer else 0
                n_clusters = len([cid for cid in cluster_stats_for_layer if cid != -1]) if cluster_stats_for_layer else 0
                row = {
                    'layer': layer,
                    'color_by': color_by,
                    'n_points': total_points,
                    'n_clusters': n_clusters,
                    'noise_n': noise_n,
                    'noise_pct': (noise_n / total_points * 100) if total_points else 0,
                    'hdbscan_metric': hdbscan_metric,
                    'hdbscan_min_cluster_size': hdbscan_min_cluster_size,
                    'hdbscan_min_samples': hdbscan_min_samples,
                    'silhouette_score': separation_metrics.get('silhouette_score') if isinstance(separation_metrics, dict) else None,
                    'calinski_harabasz_score': separation_metrics.get('calinski_harabasz_score') if isinstance(separation_metrics, dict) else None,
                    'davies_bouldin_score': separation_metrics.get('davies_bouldin_score') if isinstance(separation_metrics, dict) else None,
                    'separation_ratio': separation_metrics.get('separation_ratio') if isinstance(separation_metrics, dict) else None,
                }
                if color_by == 'phylum':
                    nmi = compute_cluster_nmi(cluster_labels_for_layer, result[2]) if total_points else None
                    summary = summarize_phylum_clusters(cluster_labels_for_layer, result[2], cluster_stats_for_layer) if cluster_stats_for_layer else None
                    row.update({
                        'cluster_phylum_nmi': nmi,
                        'asco_heavy_pct': summary['asco_heavy_fraction'] if summary else None,
                        'basidio_only_pct': summary['basidio_only_fraction'] if summary else None,
                        'zoopa_only_pct': summary['zoopa_only_fraction'] if summary else None,
                        'muco_only_pct': summary['muco_only_fraction'] if summary else None,
                        'noise_pct_redundant': (noise_n / total_points * 100) if total_points else 0,
                        'dominated_clusters_count': len(summary['dominated_clusters']) if summary else None,
                        'dominated_clusters': json.dumps(summary['dominated_clusters']) if summary else None,
                    })
                if color_by == 'enzyme':
                    nmi = compute_cluster_nmi(cluster_labels_for_layer, result[2]) if total_points else None
                    overall_sil, per_enzyme_sil = compute_enzyme_silhouette(embeddings, result[2]) if embeddings is not None else (None, None)
                    row.update({
                        'cluster_enzyme_nmi': nmi,
                        'enzyme_silhouette_overall': overall_sil,
                        'enzyme_silhouette_per_enzyme': json.dumps(per_enzyme_sil) if per_enzyme_sil else None,
                    })

                # Store cluster composition as JSON for inspection, including class/order when available
                row['cluster_stats_json'] = json.dumps(cluster_stats_for_layer) if cluster_stats_for_layer else None
                row['class_cluster_stats_json'] = json.dumps(class_cluster_stats) if color_by == 'phylum' and class_cluster_stats else None
                row['order_cluster_stats_json'] = json.dumps(order_cluster_stats) if color_by == 'phylum' and order_cluster_stats else None
                summary_rows.append(row)
            except Exception as e:
                print(f"Warning: Failed to record summary for layer {layer}: {e}")
            # If using enzyme coloring, extract the dynamic color dictionary from the first result
            if color_by == 'enzyme' and len(result) >= 8 and dynamic_enzyme_colors is None:
                dynamic_enzyme_colors = result[7]  # enzyme_color_dict
                print(f"Using dynamic enzyme colors: {list(dynamic_enzyme_colors.keys())}")
        else:
            print(f"Warning: Failed to process layer {layer}")
    
    if len(results) == 0:
        print("Error: No layers were successfully processed!")
        return
    
    # Update color_dict if using dynamic enzyme colors
#    if color_by == 'enzyme' and dynamic_enzyme_colors is not None:
#        color_dict = dynamic_enzyme_colors
    
    # Create subplot figure - adjust layout based on number of layers
    n_layers = len(results)
    
    # For all layers mode, use 6x6 grid
    if plot_all_layers or n_layers > 16:
        cols = 6
        rows = 6
        figsize = (18, 18)  # 3 inches per subplot
        print(f"Using 6x6 grid for {n_layers} layers")
    elif n_layers <= 4:
        cols = n_layers
        rows = 1
        figsize = (6*cols, 5)
    elif n_layers <= 8:
        cols = 4
        rows = 2
        figsize = (6*cols, 5*rows)
    else:
        cols = 6
        rows = (n_layers + cols - 1) // cols  # Ceiling division
        figsize = (4*cols, 4*rows)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Handle single subplot case
    if n_layers == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Flatten axes for easy iteration
    axes_flat = axes.flatten() if n_layers > 1 else axes
    
    # Plot each layer
    for idx, (layer, result) in enumerate(results.items()):
        if len(result) >= 4:  # Has required data
            gene_name_result, umap_coords, categories, protein_ids_for_layer = result[:4]
            
            ax = axes_flat[idx]
            
            # Set styling variables first
            if plot_all_layers or n_layers > 16:
                title_fontsize = 8
                point_size = 5
                label_fontsize = 4
            else:
                title_fontsize = max(10, 14 - n_layers // 6)
                point_size = 5
                label_fontsize = max(6, 10 - n_layers // 4)
            
            # Plot points colored by category (phylum or lifestyle)
            for category in color_dict.keys():
                if category in set(categories):
                    mask = [c == category for c in categories]
                    ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                             alpha=0.7,
                             color=color_dict[category],
                             s=point_size)

            # Optionally overlay HDBSCAN cluster outlines/labels
            if overlay_clusters:
                labels_idx = 8 if color_by == 'enzyme' else 7
                if len(result) > labels_idx:
                    cluster_labels = np.array(result[labels_idx])
                    unique_clusters = sorted(set(cluster_labels))
                    cmap = plt.cm.tab20
                    for cid in unique_clusters:
                        if cid == -1:
                            continue  # skip noise
                        cluster_mask = cluster_labels == cid
                        if not np.any(cluster_mask):
                            continue
                        cluster_points = umap_coords[cluster_mask]
                        try:
                            # Prefer explicit color list when available
                            colors_list = getattr(cmap, 'colors', None)
                            if colors_list:
                                color = colors_list[cid % len(colors_list)]
                            else:
                                color = cmap(float((cid % 20) / 20))
                        except Exception:
                            color = 'black'
                        if cluster_overlay_style == 'ellipse' and cluster_points.shape[0] >= 3:
                            ellipse_params = cluster_to_ellipse(cluster_points)
                            if ellipse_params:
                                width, height, angle, center = ellipse_params
                                patch = Ellipse(
                                    xy=center,
                                    width=width,
                                    height=height,
                                    angle=angle,
                                    facecolor=color,
                                    alpha=0.12,
                                    edgecolor='black',
                                    linewidth=0.8,
                                )
                                ax.add_patch(patch)
                            else:
                                ax.scatter(
                                    cluster_points[:, 0],
                                    cluster_points[:, 1],
                                    facecolors='none',
                                    edgecolors='black',
                                    linewidths=0.8,
                                    s=point_size + 20,
                                    alpha=0.9,
                                )
                        elif cluster_points.shape[0] >= 3:
                            try:
                                hull = ConvexHull(cluster_points)
                                hull_pts = cluster_points[hull.vertices]
                                hull_pts = smooth_polygon(hull_pts, iterations=2)
                                ax.fill(
                                    hull_pts[:, 0],
                                    hull_pts[:, 1],
                                    facecolor=color,
                                    alpha=0.15,
                                    edgecolor='black',
                                    linewidth=0.8,
                                )
                            except Exception:
                                ax.scatter(
                                    cluster_points[:, 0],
                                    cluster_points[:, 1],
                                    facecolors='none',
                                    edgecolors='black',
                                    linewidths=0.8,
                                    s=point_size + 20,
                                    alpha=0.9,
                                )
                        else:
                            ax.scatter(
                                cluster_points[:, 0],
                                cluster_points[:, 1],
                                facecolors='none',
                                edgecolors='black',
                                linewidths=0.8,
                                s=point_size + 20,
                                alpha=0.9,
                            )

                        # Label cluster centroid
                        centroid = cluster_points.mean(axis=0)
                        ax.text(
                            centroid[0],
                            centroid[1],
                            str(cid),
                            fontsize=max(6, label_fontsize),
                            ha='center',
                            va='center',
                            fontweight='bold',
                            color='black',
                            alpha=0.9,
                        )
            
            # Add organism labels (smaller for many subplots)
            if plot_all_layers or n_layers > 16:
                # For 6x6 grid, make labels very small or skip them
                # Only add labels for a subset to avoid overcrowding
                if idx < 6:  # Only first row gets labels
                    add_organism_labels(ax, umap_coords, protein_ids_for_layer, fontsize=label_fontsize)
            else:
                add_organism_labels(ax, umap_coords, protein_ids_for_layer, fontsize=label_fontsize)
            
            # Apply styling
            ax.set_title(f'Layer {layer}', fontsize=title_fontsize, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(True, linestyle='--', alpha=0.3)
    
    # Hide extra subplots
    for idx in range(len(results), len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    # Add legend outside the figure
    if len(results) > 0:
        # Create legend handles manually from color_dict
        import matplotlib.patches as mpatches
        
        # Get all unique categories that actually appear in the data
        all_categories = set()
        for layer, result in results.items():
            if len(result) >= 3:
                categories = result[2]
                all_categories.update(categories)
        
        # Filter out NaN values and convert to strings for sorting
        all_categories = {str(cat) for cat in all_categories if cat is not None and str(cat) != 'nan'}
        
        # Create legend handles only for categories that appear in the data
        legend_handles = []
        for category in sorted(all_categories):
            if category in color_dict:
                handle = mpatches.Patch(color=color_dict[category], label=category)
                legend_handles.append(handle)
        
        if plot_all_layers or n_layers > 16:
            # For 6x6 grid, place legend outside the figure
            fig.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=6)
        else:
            fig.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    
    # Add overall title
    color_type_str = color_by.capitalize()
    fig.suptitle(f'{gene_name} - ESM Layer Comparison (Colored by {color_type_str})', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    if save_path is None:
        # Use output_dir if provided, otherwise use current working directory
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), 'plots')
        layer_str = f"{min(original_layers)}-{max(original_layers)}" if len(original_layers) > 5 else "_".join(map(str, layers))
        save_path = os.path.join(output_dir, f'layer_comparison_{gene_name}_layers_{layer_str}_{color_by}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if save_svg:
        svg_path = os.path.splitext(save_path)[0] + '.svg'
        plt.savefig(svg_path, format='svg', bbox_inches='tight')
    plt.show()

    # Write aggregated HDBSCAN summary if requested
    if summary_rows and hdbscan_summary_csv:
        try:
            os.makedirs(os.path.dirname(hdbscan_summary_csv), exist_ok=True)
            fieldnames = sorted({k for row in summary_rows for k in row.keys()})
            with open(hdbscan_summary_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in summary_rows:
                    writer.writerow(row)
            print(f"HDBSCAN summary written to: {hdbscan_summary_csv}")
        except Exception as e:
            print(f"Warning: Failed to write HDBSCAN summary CSV: {e}")
    elif hdbscan_summary_csv:
        print(f"Warning: No HDBSCAN summary rows to write for {gene_name}; skipping CSV export.")
    
    print(f"Comparison plot saved to: {save_path}")
    
    # Analyze separation quality across layers
    print(f"\nSeparation Quality Analysis for {gene_name} (colored by {color_by}):")
    print("=" * 80)
    
    # Collect metrics for ranking
    layer_metrics = {}
    metric_names = ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score', 'separation_ratio']
    
    for layer, result in results.items():
        if len(result) >= 6:  # Has separation metrics
            separation_metrics = result[5]
            layer_metrics[layer] = separation_metrics
    
    if layer_metrics:
        # Print detailed metrics for each layer
        print(f"{'Layer':<6} {'Silhouette':<12} {'Calinski-H':<12} {'Davies-B':<10} {'Sep.Ratio':<10} {'Balance':<8}")
        print("-" * 80)
        
        for layer in sorted(layer_metrics.keys()):
            metrics = layer_metrics[layer]
            if 'error' not in metrics:
                sil = f"{metrics.get('silhouette_score', 0):.4f}" if metrics.get('silhouette_score') is not None else "N/A"
                ch = f"{metrics.get('calinski_harabasz_score', 0):.2f}" if metrics.get('calinski_harabasz_score') is not None else "N/A"
                db = f"{metrics.get('davies_bouldin_score', 0):.4f}" if metrics.get('davies_bouldin_score') is not None else "N/A"
                sep = f"{metrics.get('separation_ratio', 0):.4f}" if metrics.get('separation_ratio') is not None else "N/A"
                bal = f"{metrics.get('category_balance', 0):.4f}" if metrics.get('category_balance') is not None else "N/A"
                print(f"{layer:<6} {sil:<12} {ch:<12} {db:<10} {sep:<10} {bal:<8}")
            else:
                print(f"{layer:<6} Error: {metrics['error']}")
        
        print("\nMetric Explanations:")
        print("- Silhouette Score: Higher is better (range: -1 to 1). Measures cluster cohesion and separation.")
        print("- Calinski-Harabasz: Higher is better. Ratio of between/within cluster dispersion.")
        print("- Davies-Bouldin: Lower is better. Average similarity of clusters with most similar cluster.")
        print("- Separation Ratio: Higher is better. Ratio of inter-cluster to intra-cluster distances.")
        print("- Balance: Higher is better (0-1). Entropy-based measure of category distribution balance.")
        
        # Find best layers for each metric
        print(f"\nBest Layers by Metric:")
        print("-" * 40)
        
        # Silhouette Score (higher is better)
        valid_sil = {layer: metrics.get('silhouette_score') for layer, metrics in layer_metrics.items() 
                    if 'error' not in metrics and metrics.get('silhouette_score') is not None}
        if valid_sil:
            best_sil_layer = max(valid_sil.keys(), key=lambda x: valid_sil[x])
            print(f"Best Silhouette Score: Layer {best_sil_layer} ({valid_sil[best_sil_layer]:.4f})")
        
        # Calinski-Harabasz (higher is better)
        valid_ch = {layer: metrics.get('calinski_harabasz_score') for layer, metrics in layer_metrics.items() 
                   if 'error' not in metrics and metrics.get('calinski_harabasz_score') is not None}
        if valid_ch:
            best_ch_layer = max(valid_ch.keys(), key=lambda x: valid_ch[x])
            print(f"Best Calinski-Harabasz: Layer {best_ch_layer} ({valid_ch[best_ch_layer]:.2f})")
        
        # Davies-Bouldin (lower is better)
        valid_db = {layer: metrics.get('davies_bouldin_score') for layer, metrics in layer_metrics.items() 
                   if 'error' not in metrics and metrics.get('davies_bouldin_score') is not None}
        if valid_db:
            best_db_layer = min(valid_db.keys(), key=lambda x: valid_db[x])
            print(f"Best Davies-Bouldin: Layer {best_db_layer} ({valid_db[best_db_layer]:.4f})")
        
        # Separation Ratio (higher is better)
        valid_sep = {layer: metrics.get('separation_ratio') for layer, metrics in layer_metrics.items() 
                    if 'error' not in metrics and metrics.get('separation_ratio') is not None}
        if valid_sep:
            best_sep_layer = max(valid_sep.keys(), key=lambda x: valid_sep[x])
            print(f"Best Separation Ratio: Layer {best_sep_layer} ({valid_sep[best_sep_layer]:.4f})")
        
        # Overall ranking (composite score)
        print(f"\nOverall Ranking (Composite Score):")
        print("-" * 40)
        
        composite_scores = {}
        for layer, metrics in layer_metrics.items():
            if 'error' not in metrics:
                score = 0
                count = 0
                
                # Normalize and combine metrics
                if metrics.get('silhouette_score') is not None:
                    # Silhouette already normalized (-1 to 1), shift to (0 to 2)
                    score += (metrics['silhouette_score'] + 1) / 2
                    count += 1
                
                if metrics.get('separation_ratio') is not None and valid_sep:
                    # Normalize separation ratio
                    max_sep = max(valid_sep.values())
                    min_sep = min(valid_sep.values())
                    if max_sep > min_sep:
                        normalized_sep = (metrics['separation_ratio'] - min_sep) / (max_sep - min_sep)
                        score += normalized_sep
                        count += 1
                
                if metrics.get('davies_bouldin_score') is not None and valid_db:
                    # Davies-Bouldin: lower is better, so invert
                    max_db = max(valid_db.values())
                    min_db = min(valid_db.values())
                    if max_db > min_db:
                        normalized_db = 1 - (metrics['davies_bouldin_score'] - min_db) / (max_db - min_db)
                        score += normalized_db
                        count += 1
                
                if count > 0:
                    composite_scores[layer] = score / count
        
        if composite_scores:
            sorted_layers = sorted(composite_scores.keys(), key=lambda x: composite_scores[x], reverse=True)
            for i, layer in enumerate(sorted_layers[:5]):  # Top 5
                print(f"{i+1:2d}. Layer {layer:2d}: {composite_scores[layer]:.4f}")
        
    else:
        print("No separation metrics available for analysis.")
    
    # HDBSCAN cluster composition per layer
    print(f"\nHDBSCAN cluster composition (category percentages per cluster):")
    print("-" * 60)
    for layer, result in results.items():
        labels_idx = 8 if color_by == 'enzyme' else 7
        stats_idx = 9 if color_by == 'enzyme' else 8

        if len(result) <= stats_idx:
            print(f"Layer {layer:2d}: No clustering information available")
            continue

        cluster_labels = result[labels_idx]
        cluster_stats = result[stats_idx]
        total_points = len(cluster_labels)
        noise_size = cluster_stats.get(-1, {}).get('size', 0)
        noise_pct = (noise_size / total_points * 100) if total_points else 0
        n_clusters = len([cid for cid in cluster_stats if cid != -1])

        # Totals for percent-of-category reporting
        primary_totals = Counter(result[2]) if len(result) > 2 else Counter()
        class_totals = Counter()
        order_totals = Counter()

        print(f"Layer {layer:2d}: clusters={n_clusters}, noise={noise_pct:.1f}%")

        class_cluster_stats = None
        order_cluster_stats = None
        if color_by == 'phylum':
            try:
                class_dict = load_class_info(TRAITS_FILE)
                if class_dict:
                    protein_ids_for_layer = result[3] if len(result) > 3 else []
                    class_labels = map_accessions_to_labels(protein_ids_for_layer, class_dict)
                    if len(class_labels) == len(cluster_labels):
                        class_cluster_stats = compute_cluster_category_percentages(cluster_labels, class_labels)
                        class_totals = Counter(class_labels)
                order_dict = load_order_info(TRAITS_FILE)
                if order_dict:
                    protein_ids_for_layer = result[3] if len(result) > 3 else []
                    order_labels = map_accessions_to_labels(protein_ids_for_layer, order_dict)
                    if len(order_labels) == len(cluster_labels):
                        order_cluster_stats = compute_cluster_category_percentages(cluster_labels, order_labels)
                        order_totals = Counter(order_labels)
            except Exception as e:
                print(f"  Warning: Failed to compute class composition: {e}")

        for cid, stat in sorted(cluster_stats.items(), key=lambda x: (-x[1]['size'], x[0])):
            label_name = 'Noise' if cid == -1 else f'Cluster {cid}'
            pct_total = stat.get('percent_of_total', 0)
            print(f"  {label_name}: n={stat['size']} ({pct_total:.1f}% of points)")
            top_cats = sorted(stat['category_percentages'].items(), key=lambda x: (-x[1], x[0]))[:5]
            if top_cats:
                cat_parts = []
                for cat, pct in top_cats:
                    cat_total = primary_totals.get(cat, 0)
                    pct_of_cat = (pct * stat['size'] / 100.0 / cat_total * 100.0) if cat_total else 0.0
                    cat_parts.append(f"{cat}: {pct:.1f}% (of cat: {pct_of_cat:.1f}%)")
                print(f"    {', '.join(cat_parts)}")
            if class_cluster_stats and cid in class_cluster_stats:
                class_stat = class_cluster_stats[cid]
                top_classes = sorted(class_stat['category_percentages'].items(), key=lambda x: (-x[1], x[0]))[:5]
                if top_classes:
                    class_parts = []
                    for cls, pct in top_classes:
                        cls_total = class_totals.get(cls, 0)
                        pct_of_cls = (pct * class_stat['size'] / 100.0 / cls_total * 100.0) if cls_total else 0.0
                        class_parts.append(f"{cls}: {pct:.1f}% (of cls: {pct_of_cls:.1f}%)")
                    print(f"    Classes: {', '.join(class_parts)}")
            if order_cluster_stats and cid in order_cluster_stats:
                order_stat = order_cluster_stats[cid]
                top_orders = sorted(order_stat['category_percentages'].items(), key=lambda x: (-x[1], x[0]))[:5]
                if top_orders:
                    order_parts = []
                    for ordr, pct in top_orders:
                        ord_total = order_totals.get(ordr, 0)
                        pct_of_ord = (pct * order_stat['size'] / 100.0 / ord_total * 100.0) if ord_total else 0.0
                        order_parts.append(f"{ordr}: {pct:.1f}% (of ord: {pct_of_ord:.1f}%)")
                    print(f"    Orders: {', '.join(order_parts)}")

        # Additional summaries by mode
        categories = result[2] if len(result) >= 3 else []
        embeddings = result[6] if len(result) > 6 else None

        if color_by == 'phylum':
            nmi = compute_cluster_nmi(cluster_labels, categories)
            summary = summarize_phylum_clusters(cluster_labels, categories, cluster_stats)
            if nmi is not None:
                print(f"  Cluster–phylum NMI: {nmi:.4f}")
            dominated = summary['dominated_clusters']
            if dominated:
                dominated_str = "; ".join([f"C{cid} {phylum} {pct:.1f}% (n={size})" for cid, phylum, pct, size in dominated])
                print(f"  Phylum-dominated clusters (>{80}%): {dominated_str}")
            print(f"  Ascomycota-heavy cluster coverage: {summary['asco_heavy_fraction']:.1f}% of points")
            print(f"  Basidiomycota-only coverage: {summary['basidio_only_fraction']:.1f}% | Zoopagomycota-only: {summary['zoopa_only_fraction']:.1f}% | Mucoromycota-only: {summary['muco_only_fraction']:.1f}%")
            print(f"  Noise coverage: {summary['noise_fraction']:.1f}%")

                if color_by == 'phylum':
                    try:
                        traits_file = process_gene_data_custom.traits_file
                        class_dict = load_class_info(traits_file)
                        if class_dict:
                            protein_ids_for_layer = result[3] if len(result) > 3 else []
                            class_labels = map_accessions_to_labels(protein_ids_for_layer, class_dict)
                            if len(class_labels) == len(cluster_labels):
                                class_cluster_stats = compute_cluster_category_percentages(cluster_labels, class_labels)
                        order_dict = load_order_info(traits_file)
                        if order_dict:
                            protein_ids_for_layer = result[3] if len(result) > 3 else []
                            order_labels = map_accessions_to_labels(protein_ids_for_layer, order_dict)
                            if len(order_labels) == len(cluster_labels):
                                order_cluster_stats = compute_cluster_category_percentages(cluster_labels, order_labels)
                        genus_dict = load_genus_info(traits_file)
                        if genus_dict:
                            protein_ids_for_layer = result[3] if len(result) > 3 else []
                            genus_labels = map_accessions_to_labels(protein_ids_for_layer, genus_dict)
                    except Exception as e:
                        print(f"Warning: Failed to compute class/order composition for layer {layer}: {e}")
            if per_enzyme_sil:
                per_str = "; ".join([f"{enz}: {score:.4f}" for enz, score in sorted(per_enzyme_sil.items())])
                print(f"  Enzyme silhouettes: {per_str}")

    # Print diversity comparison if available
    print(f"\nDiversity comparison for {gene_name}:")
    print("-" * 50)
    for layer, result in results.items():
        if len(result) >= 5:  # Has diversity metrics
            diversity_metrics = result[4]
            if isinstance(diversity_metrics, dict):
                print(f"Layer {layer:2d}: Mean distance = {diversity_metrics['mean_distance']:.4f} ± {diversity_metrics['std_distance']:.4f}")
            else:
                print(f"Layer {layer:2d}: Diversity metrics = {diversity_metrics}")
        else:
            print(f"Layer {layer:2d}: No diversity metrics available")
    
    return results


def find_best_separation_layer(gene_name, base_dir=None, color_by='phylum', layers=None, top_n=5):
    """
    Find the best layer(s) for separating categories without creating plots.
    
    Args:
        gene_name: name of the gene
        base_dir: base directory for embeddings  
        color_by: what to color points by ('phylum', 'lifestyle', 'lifestyle_keywords', 'growth_form', or 'enzyme')
        layers: specific layers to analyze, if None will auto-detect all available
        top_n: number of top layers to return
    
    Returns:
        dict: Results with separation metrics for each layer
    """
    if layers is None:
        layers = find_available_layers(gene_name, base_dir)
        if not layers:
            print(f"Error: No embedding files found for gene {gene_name}")
            return {}
    
    print(f"Analyzing separation quality for {gene_name} across {len(layers)} layers...")
    print(f"Coloring by: {color_by}")
    
    # Process each layer
    layer_results = {}
    for layer in layers:
        print(f"Processing layer {layer}...")
        
        if base_dir:
            result = process_gene_data_custom(gene_name, layer, base_dir, color_by)
        else:
            result = process_gene_data(gene_name, layer=layer)
            
        if result is not None and len(result) >= 6:
            layer_results[layer] = result[5]  # separation_metrics
        else:
            print(f"Warning: Failed to process layer {layer}")
    
    if not layer_results:
        print("Error: No layers were successfully processed!")
        return {}
    
    # Analyze results
    print(f"\nSeparation Quality Analysis:")
    print("=" * 60)
    
    # Print detailed metrics
    print(f"{'Layer':<6} {'Silhouette':<12} {'Calinski-H':<12} {'Davies-B':<10} {'Sep.Ratio':<10}")
    print("-" * 60)
    
    valid_layers = {}
    for layer in sorted(layer_results.keys()):
        metrics = layer_results[layer]
        if 'error' not in metrics:
            valid_layers[layer] = metrics
            sil = f"{metrics.get('silhouette_score', 0):.4f}" if metrics.get('silhouette_score') is not None else "N/A"
            ch = f"{metrics.get('calinski_harabasz_score', 0):.2f}" if metrics.get('calinski_harabasz_score') is not None else "N/A"
            db = f"{metrics.get('davies_bouldin_score', 0):.4f}" if metrics.get('davies_bouldin_score') is not None else "N/A"
            sep = f"{metrics.get('separation_ratio', 0):.4f}" if metrics.get('separation_ratio') is not None else "N/A"
            print(f"{layer:<6} {sil:<12} {ch:<12} {db:<10} {sep:<10}")
        else:
            print(f"{layer:<6} Error: {metrics['error']}")
    
    # Calculate composite rankings
    if valid_layers:
        composite_scores = {}
        
        # Get all valid values for normalization
        all_sil = {layer: m.get('silhouette_score') for layer, m in valid_layers.items() 
                  if m.get('silhouette_score') is not None}
        all_sep = {layer: m.get('separation_ratio') for layer, m in valid_layers.items() 
                  if m.get('separation_ratio') is not None}
        all_db = {layer: m.get('davies_bouldin_score') for layer, m in valid_layers.items() 
                 if m.get('davies_bouldin_score') is not None}
        
        for layer, metrics in valid_layers.items():
            score = 0
            count = 0
            
            # Silhouette score (already normalized -1 to 1)
            if metrics.get('silhouette_score') is not None:
                score += (metrics['silhouette_score'] + 1) / 2  # Convert to 0-1 range
                count += 1
            
            # Separation ratio (higher is better)
            if metrics.get('separation_ratio') is not None and all_sep:
                max_sep = max(all_sep.values())
                min_sep = min(all_sep.values())
                if max_sep > min_sep:
                    normalized_sep = (metrics['separation_ratio'] - min_sep) / (max_sep - min_sep)
                    score += normalized_sep
                    count += 1
            
            # Davies-Bouldin (lower is better, so invert)
            if metrics.get('davies_bouldin_score') is not None and all_db:
                max_db = max(all_db.values())
                min_db = min(all_db.values())
                if max_db > min_db:
                    normalized_db = 1 - (metrics['davies_bouldin_score'] - min_db) / (max_db - min_db)
                    score += normalized_db
                    count += 1
            
            if count > 0:
                composite_scores[layer] = score / count
        
        # Print rankings
        print(f"\nTop {top_n} Layers (Composite Ranking):")
        print("-" * 40)
        
        if composite_scores:
            sorted_layers = sorted(composite_scores.keys(), key=lambda x: composite_scores[x], reverse=True)
            top_layers = sorted_layers[:top_n]
            
            for i, layer in enumerate(top_layers):
                print(f"{i+1:2d}. Layer {layer:2d}: Score = {composite_scores[layer]:.4f}")
                
                # Show individual metrics for this layer
                metrics = valid_layers[layer]
                details = []
                if metrics.get('silhouette_score') is not None:
                    details.append(f"Silhouette: {metrics['silhouette_score']:.4f}")
                if metrics.get('separation_ratio') is not None:
                    details.append(f"Sep.Ratio: {metrics['separation_ratio']:.4f}")
                if metrics.get('davies_bouldin_score') is not None:
                    details.append(f"Davies-B: {metrics['davies_bouldin_score']:.4f}")
                
                if details:
                    print(f"      ({', '.join(details)})")
            
            print(f"\nRecommendation: Use Layer {top_layers[0]} for best separation of {color_by} categories.")
            
            return {
                'best_layer': top_layers[0],
                'top_layers': top_layers,
                'composite_scores': composite_scores,
                'detailed_metrics': valid_layers
            }
    
    return layer_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare UMAP visualizations across different ESM layers')
    
    parser.add_argument('gene_name', help='Name of the gene to analyze')
    parser.add_argument('--layers', type=str, default=None,
                       help='Comma-separated layer numbers (e.g., "2,8,12") or "all" for plotting all available layers in 6x6 grid')
    parser.add_argument('--input_dir', type=str, default=None,
                       help='Directory containing embedding files')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save plots')
    parser.add_argument('--max_layers', type=int, default=12,
                       help='Maximum number of layers to plot for readability when auto-detecting (ignored if --layers all is used)')
    parser.add_argument('--sample_method', choices=['uniform', 'start_end', 'specific'], default='specific',
                       help='How to sample layers when more than max_layers (default: specific)')
    parser.add_argument('--color_by', choices=['phylum', 'lifestyle', 'lifestyle_keywords', 'growth_form', 'enzyme'], default='phylum',
                       help='What to color points by: phylum, lifestyle, lifestyle_keywords, growth_form, or enzyme (default: phylum)')
    parser.add_argument('--hdbscan_min_cluster_size', type=int, default=50,
                       help='HDBSCAN minimum cluster size (default: 50)')
    parser.add_argument('--hdbscan_min_samples', type=int, default=5,
                       help='HDBSCAN min_samples (default: 5)')
    parser.add_argument('--hdbscan_metric', type=str, default='cosine',
                       help='HDBSCAN distance metric (e.g., cosine, euclidean; default: cosine)')
    parser.add_argument('--no_cluster_overlay', action='store_true',
                       help='Disable drawing HDBSCAN clusters on the UMAP plots')
    parser.add_argument('--cluster_overlay_style', choices=['ellipse', 'hull'], default='ellipse',
                       help='How to draw clusters: ellipse (covariance-based) or hull')
    parser.add_argument('--save_svg', action='store_true',
                       help='Also save the plot as SVG alongside PNG')
    parser.add_argument('--export_hdbscan', action='store_true',
                       help='Export HDBSCAN labels (TSV) and stats (JSON) to disk')
    parser.add_argument('--export_hdbscan_membership', action='store_true',
                       help='When exporting HDBSCAN, also write per-cluster membership TSV (protein IDs)')
    parser.add_argument('--hdbscan_export_dir', type=str, default=None,
                       help='Directory for HDBSCAN export files (default: output_dir)')
    parser.add_argument('--hdbscan_summary_csv', type=str, default=None,
                       help='Path to write aggregated HDBSCAN summaries across layers (optional)')
    parser.add_argument('--analyze_only', action='store_true',
                       help='Only analyze separation metrics without creating plots')
    parser.add_argument('--top_n', type=int, default=5,
                       help='Number of top layers to show in analysis (default: 5)')
    
    args = parser.parse_args()

    overlay_clusters = not args.no_cluster_overlay
    
    # Parse layers argument
    if args.layers is None or args.layers.lower() == 'all':
        layers_to_compare = None
        # Set flag to indicate we want to plot all layers
        compare_layers_umap._plot_all_requested = (args.layers is not None and args.layers.lower() == 'all')
    else:
        layers_to_compare = [int(x.strip()) for x in args.layers.split(',')]
        compare_layers_umap._plot_all_requested = False
    
    print(f"Comparing layers for gene: {args.gene_name}")
    print(f"Coloring by: {args.color_by}")
    if args.input_dir:
        print(f"Input directory: {args.input_dir}")
    if args.output_dir:
        print(f"Output directory: {args.output_dir}")
    
    if layers_to_compare:
        print(f"Specific layers: {layers_to_compare}")
    else:
        print("Auto-detecting all available layers...")
    
    # Run the analysis
    if args.analyze_only:
        # Only analyze separation metrics without plotting
        find_best_separation_layer(
            gene_name=args.gene_name,
            base_dir=args.input_dir,
            color_by=args.color_by,
            layers=layers_to_compare,
            top_n=args.top_n
        )
    else:
        # Run the full comparison with plots
        compare_layers_umap(
            gene_name=args.gene_name,
            layers=layers_to_compare,
            base_dir=args.input_dir,
            output_dir=args.output_dir,
            max_layers=args.max_layers,
            sample_layers=args.sample_method,
            color_by=args.color_by,
            hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
            hdbscan_min_samples=args.hdbscan_min_samples,
            hdbscan_metric=args.hdbscan_metric,
            overlay_clusters=overlay_clusters,
            cluster_overlay_style=args.cluster_overlay_style,
            save_svg=args.save_svg,
            export_hdbscan=args.export_hdbscan,
            export_hdbscan_membership=args.export_hdbscan_membership,
            hdbscan_export_dir=args.hdbscan_export_dir,
            hdbscan_summary_csv=args.hdbscan_summary_csv,
        )
