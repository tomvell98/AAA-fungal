import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import itertools
from matplotlib.colors import rgb2hex
import numba as nb
from numba import types
from numba.typed import Dict, List
import matplotlib.colors as mcolors
import os
import datashader as ds
from datashader.colors import colormap_select, Greys9
from datashader.utils import export_image
import colorcet as cc
import holoviews as hv
from holoviews.operation.datashader import datashade, shade
hv.extension('bokeh')
import math
import argparse
from sklearn.linear_model import LinearRegression
from scipy import stats


def load_distance_matrix(path):
    """Robustly load a distance matrix, handling comma or whitespace separators."""
    dm = pd.read_csv(path, sep=None, engine='python', header=None, skiprows=1)
    if dm.shape[1] == 1:
        dm = pd.read_csv(path, sep=',', header=None, skiprows=1)
    return dm

matplotlib.use('Agg')

def calculate_residuals_and_deviations(df, taxa_info, group_column='Phylum'):
    """
    Calculate residuals from the correlation line and identify systematic deviations by taxonomic groups.
    
    Parameters:
        df (DataFrame): DataFrame with 'x', 'y' columns containing distance measurements
        taxa_info (dict): Dictionary mapping accessions to taxonomic information
        group_column (str): Which taxonomic level to analyze ('Phylum' or 'Class')
    
    Returns:
        dict: Analysis results including residuals, group statistics, and deviation rankings
    """
    # Remove any infinite or NaN values
    mask = np.isfinite(df['x']) & np.isfinite(df['y'])
    clean_df = df[mask].copy()
    
    if len(clean_df) == 0:
        return None
    
    # Fit linear regression
    X = clean_df['x'].values.reshape(-1, 1)
    y = clean_df['y'].values
    
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)
    residuals = y - y_pred
    
    # Calculate correlation statistics
    from scipy import stats as scipy_stats
    correlation, p_value = scipy_stats.pearsonr(clean_df['x'], clean_df['y'])
    
    # Add residuals and predictions to dataframe
    clean_df = clean_df.copy()
    clean_df['y_pred'] = y_pred
    clean_df['residuals'] = residuals
    clean_df['abs_residuals'] = np.abs(residuals)
    
    # Map taxonomic groups to pairs (this requires the pair information)
    if 'phylum_pair' in clean_df.columns:
        # Extract individual taxa from pairs for residual analysis
        group_residuals = {}
        
        for idx, row in clean_df.iterrows():
            pair_name = row['phylum_pair']
            if '-' in str(pair_name):
                taxa_a, taxa_b = str(pair_name).split('-', 1)
                residual = row['residuals']
                
                # Add residual to both taxa involved in the pair
                for taxon in [taxa_a, taxa_b]:
                    if taxon not in group_residuals:
                        group_residuals[taxon] = []
                    group_residuals[taxon].append(residual)
    else:
        # If we have individual accession information, we need to handle it differently
        group_residuals = {}
    
    # Calculate statistics for each group
    group_stats = {}
    for group, residual_list in group_residuals.items():
        if len(residual_list) > 0:
            group_stats[group] = {
                'mean_residual': np.mean(residual_list),
                'median_residual': np.median(residual_list),
                'std_residual': np.std(residual_list),
                'count': len(residual_list),
                'mean_abs_residual': np.mean(np.abs(residual_list))
            }
    
    # Rank groups by systematic deviation
    if group_stats:
        sorted_by_mean = sorted(group_stats.items(), key=lambda x: x[1]['mean_residual'])
        sorted_by_abs_mean = sorted(group_stats.items(), key=lambda x: x[1]['mean_abs_residual'], reverse=True)
    else:
        sorted_by_mean = []
        sorted_by_abs_mean = []
    
    return {
        'dataframe': clean_df,
        'regression': reg,
        'correlation': correlation,
        'p_value': p_value,
        'group_stats': group_stats,
        'ranked_by_mean_residual': sorted_by_mean,
        'ranked_by_abs_residual': sorted_by_abs_mean,
        'slope': reg.coef_[0],
        'intercept': reg.intercept_
    }

def create_taxa_mapping(accessions, group_level='Phylum', taxa_csv=None):
    """
    Create mapping from accessions to taxonomic groups.
    
    Parameters:
        accessions (list): List of accession IDs
        group_level (str): 'Phylum' or 'Class'
        taxa_csv (str): Path to taxa CSV file (required)
    
    Returns:
        dict: Mapping from accession to taxonomic group
    """
    if taxa_csv is None:
        raise ValueError("taxa_csv path must be provided to create_taxa_mapping.")
    taxa_df = pd.read_csv(taxa_csv)
    taxa_df[group_level] = taxa_df[group_level].fillna("Unknown")
    # Create mapping
    taxa_dict = dict(zip(taxa_df['Accession'].str[:15], taxa_df[group_level]))
    # Map accessions to groups
    group_mapping = {}
    for acc in accessions:
        group = taxa_dict.get(acc, "Unknown")
        if pd.isna(group):
            group = "Unknown"
        group_mapping[acc] = str(group)
    return group_mapping

@nb.njit
def get_colors_numba(rows, cols, phyla_indices, pair_to_color_idx):
    """Numba-accelerated function to determine colors based on phylum pairs"""
    n = len(rows)
    color_indices = np.empty(n, dtype=np.int64)
    
    for k in range(n):
        i, j = rows[k], cols[k]
        phylum_i = phyla_indices[i]
        phylum_j = phyla_indices[j]
        
        # Ensure consistent ordering for pair lookup
        if phylum_i > phylum_j:
            pair_key = (phylum_j, phylum_i)
        else:
            pair_key = (phylum_i, phylum_j)
            
        color_indices[k] = pair_to_color_idx[pair_key]
        
    return color_indices


def filter_quantiles(dm1, dm2, quantile=0.9999):
    """Filter distance matrices to only include values below a specified quantile."""
    # Compute the upper triangle indices
    rows, cols = np.triu_indices(dm1.shape[0], k=1)

    # Flatten the upper triangle values
    dm1_flat = dm1[rows, cols]
    dm2_flat = dm2[rows, cols]

    # Compute quantile thresholds
    q1 = np.quantile(dm1_flat, quantile)
    q2 = np.quantile(dm2_flat, quantile)

    # Filter values below the quantile thresholds
    mask = (dm1_flat <= q1) & (dm2_flat <= q2)

    # Create filtered matrices
    filtered_dm1 = np.zeros_like(dm1)
    filtered_dm2 = np.zeros_like(dm2)

    filtered_dm1[rows[mask], cols[mask]] = dm1[rows[mask], cols[mask]]
    filtered_dm2[rows[mask], cols[mask]] = dm2[rows[mask], cols[mask]]

    return filtered_dm1, filtered_dm2


def square_ranges(x_min, x_max, y_min, y_max):
    """Pad axis limits so the plotted span is square without forcing shared mins/maxes."""
    span_x = x_max - x_min
    span_y = y_max - y_min
    span = max(span_x, span_y)
    if span == 0:
        span = 1e-9
    pad_x = (span - span_x) / 2
    pad_y = (span - span_y) / 2
    return x_min - pad_x, x_max + pad_x, y_min - pad_y, y_max + pad_y


def plot_dms_multiple(dm_list, ref_dm, ref_name, dm_names, to_remove=None, quantile=0.9999):
    """
    Plot multiple distance matrices against a reference matrix.

    Parameters:
        dm_list (list): List of dataframes for distance matrices to compare.
        ref_dm (DataFrame): Reference distance matrix.
        ref_name (str): Name of the reference matrix.
        dm_names (list): Names of the matrices in dm_list.
        to_remove (list): List of accessions to remove.
        quantile (float): Quantile threshold for filtering.
    """
    for dm, dm_name in zip(dm_list, dm_names):
        print(f"Processing {dm_name} against {ref_name}...")
        # Put the reference on the y-axis by passing it as dm2
        plot_dms(dm, ref_dm, dm_name, ref_name, to_remove=to_remove, quantile=quantile)


def plot_dms_with_deviations(dm1, dm2, dm1_name='dm1', dm2_name='dm2', to_remove=None, quantile=0.9999, group_level='Phylum'):
    """
    Enhanced plotting function that includes deviation analysis and class-level coloring.
    """
    # [Initial data processing - same as original plot_dms]
    dm1.iloc[:, 0] = dm1.iloc[:, 0].str[:15]
    dm2.iloc[:, 0] = dm2.iloc[:, 0].str[:15]
    
    dm1_order = dm1.iloc[:, 0].tolist()
    dm2_order = dm2.iloc[:, 0].tolist()
    
    common_entries = list(set(dm1_order) & set(dm2_order))
    missing_entries = set(dm1_order) - set(dm2_order)
    if missing_entries:
        print(f"Warning: Following entries are missing from {dm2_name}:")
        print(missing_entries)
        print("Removing these entries from both matrices...")
    
    keep_indices_dm1 = [i for i, acc in enumerate(dm1_order) if acc in common_entries]
    keep_indices_dm2 = [i for i, acc in enumerate(dm2_order) if acc in common_entries]
    
    dm1 = dm1.iloc[keep_indices_dm1].reset_index(drop=True)
    dm2 = dm2.iloc[keep_indices_dm2].reset_index(drop=True)
    
    col_indices_dm1 = [0] + [i + 1 for i in keep_indices_dm1]
    col_indices_dm2 = [0] + [i + 1 for i in keep_indices_dm2]
    
    dm1 = dm1.iloc[:, col_indices_dm1]
    dm2 = dm2.iloc[:, col_indices_dm2]
    
    dm1_order = dm1.iloc[:, 0].tolist()
    dm2_order = dm2.iloc[:, 0].tolist()
    
    order_mapping = [dm2_order.index(x) for x in dm1_order]
    
    dm2 = dm2.iloc[order_mapping]
    dm2_cols = dm2.columns.tolist()
    dm2 = dm2[[dm2_cols[0]] + [dm2_cols[i+1] for i in order_mapping]]

    if to_remove is not None:
        to_remove = [acc.strip() for acc in to_remove]
        keep_mask = ~dm1.iloc[:, 0].isin(to_remove)
        keep_indices = np.where(keep_mask)[0]
        dm1 = dm1.iloc[keep_indices].reset_index(drop=True)
        dm2 = dm2.iloc[keep_indices].reset_index(drop=True)
        col_indices = [0] + [i + 1 for i in keep_indices]
        dm1 = dm1.iloc[:, col_indices]
        dm2 = dm2.iloc[:, col_indices]

    accessions = dm1.iloc[:, 0].tolist()
    dm1 = dm1.iloc[:, 1:]
    dm2 = dm2.iloc[:, 1:]

    dm1_array = dm1.to_numpy()
    dm2_array = dm2.to_numpy()
    
    dm1_array, dm2_array = filter_quantiles(dm1_array, dm2_array, quantile=quantile)

    # Create taxonomic mapping
    # User must provide taxa_csv as a keyword argument
    import inspect
    taxa_csv = None
    frame = inspect.currentframe()
    try:
        outer_frames = inspect.getouterframes(frame)
        for f in outer_frames:
            if 'taxa_csv' in f.frame.f_locals:
                taxa_csv = f.frame.f_locals['taxa_csv']
                break
    finally:
        del frame
    if taxa_csv is None:
        raise ValueError("taxa_csv must be provided to plot_dms_with_deviations via the calling context.")
    taxa_mapping = create_taxa_mapping(accessions, group_level, taxa_csv=taxa_csv)
    
    # Get upper triangle and create dataframe
    rows, cols = np.triu_indices(dm1_array.shape[0], k=1)
    dm1_flat = dm1_array[rows, cols]
    dm2_flat = dm2_array[rows, cols]
    
    # Create enhanced dataframe with taxonomic info
    df = pd.DataFrame({
        'x': dm1_flat,
        'y': dm2_flat,
        'acc1': [accessions[i] for i in rows],
        'acc2': [accessions[i] for i in cols],
        'taxa1': [taxa_mapping[accessions[i]] for i in rows],
        'taxa2': [taxa_mapping[accessions[i]] for i in cols],
    })
    
    # Create pair labels for both phylum and class levels
    df['phylum_pair'] = pd.Categorical([
        f"{min(df.iloc[i]['taxa1'], df.iloc[i]['taxa2'])}-{max(df.iloc[i]['taxa1'], df.iloc[i]['taxa2'])}"
        for i in range(len(df))
    ])
    
    # Perform deviation analysis
    print(f"Performing deviation analysis for {group_level} level...")
    deviation_results = calculate_residuals_and_deviations(df, taxa_mapping, group_level)
    
    if deviation_results:
        # Print summary statistics
        print(f"\nCorrelation: {deviation_results['correlation']:.4f} (p-value: {deviation_results['p_value']:.2e})")
        print(f"Regression equation: y = {deviation_results['slope']:.4f}x + {deviation_results['intercept']:.4f}")
        
        print(f"\n{group_level} groups with largest systematic deviations (mean residual):")
        for group, stats in deviation_results['ranked_by_mean_residual'][:5]:
            direction = "above" if stats['mean_residual'] > 0 else "below"
            print(f"  {group}: {stats['mean_residual']:.4f} ({direction} regression line, n={stats['count']})")
        
        print(f"\n{group_level} groups with largest absolute deviations:")
        for group, stats in deviation_results['ranked_by_abs_residual'][:5]:
            print(f"  {group}: {stats['mean_abs_residual']:.4f} (|mean residual|, n={stats['count']})")
        
        # Update dataframe with residual information
        df = deviation_results['dataframe']
    
    # Generate enhanced color scheme for the selected taxonomic level
    unique_groups = list(set(taxa_mapping.values()))
    
    # Use a larger color palette for class-level analysis
    if group_level == 'Class':
        # Use tab20 + Set3 for more colors
        tab20_colors = plt.cm.tab20.colors
        set3_colors = plt.cm.Set3.colors
        all_colors = list(tab20_colors) + list(set3_colors)
        group_colors = {group: mcolors.rgb2hex(all_colors[i % len(all_colors)]) 
                       for i, group in enumerate(sorted(unique_groups))}
    else:
        # Use existing phylum colors when available
        preselected_colors = {
            "Ascomycota": "#377eb8",
            "Basidiomycota": "#e41a1c",
            "Mucoromycota": "#4daf4a",
            "Zoopagomycota": "#984ea3",
            "Chytridiomycota": "#ff7f00",
            "Blastocladiomycota": "#ffff33",
            "Cryptomycota": "#a65628",
        }
        group_colors = preselected_colors.copy()
        # Add colors for any missing groups
        remaining_groups = [g for g in unique_groups if g not in group_colors]
        tab10_colors = plt.cm.tab10.colors
        for i, group in enumerate(remaining_groups):
            group_colors[group] = mcolors.rgb2hex(tab10_colors[i % len(tab10_colors)])
    
    default_color = '#BBBBBB'
    
    # Create plots
    os.makedirs('figures1', exist_ok=True)
    safe_dm1 = ''.join(c if (c.isalnum() or c in ('_', '-')) else '_' for c in dm1_name.replace(' ', '_'))
    safe_dm2 = ''.join(c if (c.isalnum() or c in ('_', '-')) else '_' for c in dm2_name.replace(' ', '_'))
    
    x_min, x_max = df['x'].min(), df['x'].max()
    y_min, y_max = df['y'].min(), df['y'].max()
    if x_min == x_max:
        x_min -= 1e-9
        x_max += 1e-9
    if y_min == y_max:
        y_min -= 1e-9
        y_max += 1e-9
    x_min, x_max, y_min, y_max = square_ranges(x_min, x_max, y_min, y_max)
    
    # 1. Enhanced taxonomic plot
    fig, ax = plt.subplots(figsize=(14, 12))
    canvas = ds.Canvas(plot_width=1600, plot_height=1600,
                      x_range=(x_min, x_max),
                      y_range=(y_min, y_max))
    
    # Create color mapping for pairs
    unique_pairs = df['phylum_pair'].unique()
    def parse_pair(p):
        a, b = str(p).split('-', 1)
        return a, b
    
    color_key = {}
    for pair in unique_pairs:
        a, b = parse_pair(pair)
        if a == b:  # Same group pairs
            color_key[str(pair)] = group_colors.get(a, default_color)
        else:  # Mixed pairs
            color_key[str(pair)] = default_color
    
    agg = canvas.points(df, 'x', 'y', agg=ds.count_cat('phylum_pair'))
    img = ds.tf.shade(agg, color_key=color_key, how='eq_hist')
    img = ds.tf.set_background(img, 'white')
    
    # Add diagonal and regression lines
    diag_x0 = max(x_min, y_min)
    diag_x1 = min(x_max, y_max)
    diagonal_df = pd.DataFrame({'x': [diag_x0, diag_x1], 'y': [diag_x0, diag_x1]})
    diagonal_agg = canvas.line(diagonal_df, 'x', 'y')
    diagonal_img = ds.tf.shade(diagonal_agg, cmap=['black'])
    
    if deviation_results:
        reg_y_min = deviation_results['slope'] * x_min + deviation_results['intercept']
        reg_y_max = deviation_results['slope'] * x_max + deviation_results['intercept']
        reg_df = pd.DataFrame({'x': [x_min, x_max], 'y': [reg_y_min, reg_y_max]})
        reg_agg = canvas.line(reg_df, 'x', 'y')
        reg_img = ds.tf.shade(reg_agg, cmap=['red'])
        final_img = ds.transfer_functions.stack(img, diagonal_img, reg_img)
    else:
        final_img = ds.transfer_functions.stack(img, diagonal_img)
    
    ax.imshow(final_img.to_pil(), extent=[x_min, x_max, y_min, y_max])
    if hasattr(ax, 'set_box_aspect'):
        ax.set_box_aspect(1)  # Keep square axes while allowing different x/y spans
    ax.set_xlabel(f'{dm1_name} Distance (substitutions/site)', fontsize=18)
    ax.set_ylabel(f'{dm2_name} Distance (substitutions/site)', fontsize=18)
    ax.set_title(f'{group_level} Groups (Red=Regression, Black=Diagonal)', fontsize=14)
    
    # Add legend for same-group pairs only
    self_pairs_in_data = sorted({p for p in unique_pairs if parse_pair(p)[0] == parse_pair(p)[1]})
    if self_pairs_in_data:
        legend_elements = [matplotlib.patches.Patch(facecolor=color_key[str(pair)], label=str(pair))
                          for pair in self_pairs_in_data]
        legend = ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5),
                          fontsize=10, ncol=1)
        plt.subplots_adjust(right=0.85)
    
    plt.savefig(f'figures1/{safe_dm1}_vs_{safe_dm2}_{group_level.lower()}_enhanced.png', dpi=600, bbox_inches='tight')
    plt.savefig(f'figures1/{safe_dm1}_vs_{safe_dm2}_{group_level.lower()}_enhanced.svg', format='svg', bbox_inches='tight')
    plt.close()
    
    # 2. Residual analysis plot
    if deviation_results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Residuals vs fitted values
        fitted_vals = deviation_results['dataframe']['y_pred']
        residuals = deviation_results['dataframe']['residuals']
        
        ax1.scatter(fitted_vals, residuals, alpha=0.6, s=1)
        ax1.axhline(y=0, color='red', linestyle='--')
        ax1.set_xlabel(f'Fitted {dm2_name} Distance')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Fitted Values')
        
        # Histogram of residuals by top deviating groups
        top_groups = [item[0] for item in deviation_results['ranked_by_abs_residual'][:5]]
        
        for group in top_groups:
            if group in deviation_results['group_stats']:
                # This is simplified - in practice you'd need to extract residuals by group
                pass
        
        ax2.hist(residuals, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--')
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Residuals')
        
        plt.tight_layout()
        plt.savefig(f'figures1/{safe_dm1}_vs_{safe_dm2}_residual_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'figures1/{safe_dm1}_vs_{safe_dm2}_residual_analysis.svg', format='svg', bbox_inches='tight')
        plt.close()
    
    print(f'Enhanced plots with {group_level} analysis created successfully!')

def create_deviation_focused_plots(dm_list, ref_dm, ref_name, dm_names, to_remove=None, quantile=0.9999, group_level='Class', taxa_csv=None):
    """
    Create plots specifically focused on identifying and visualizing taxonomic deviations.
    """
    # Load taxa information once
    if taxa_csv is None:
        raise ValueError("taxa_csv path must be provided to create_deviation_focused_plots.")
    taxa_df = pd.read_csv(taxa_csv)
    taxa_df[group_level] = taxa_df[group_level].fillna("Unknown")
    taxa_dict = dict(zip(taxa_df['Accession'].str[:15], taxa_df[group_level]))
    
    all_deviations = {}
    summary_stats = []
    
    for dm, dm_name in zip(dm_list, dm_names):
        print(f"\nAnalyzing deviations for {dm_name} vs {ref_name}...")
        
        # Process data (simplified version of the processing in plot_dms)
        dm1, dm2 = dm.copy(), ref_dm.copy()
        
        # Data cleaning and alignment (same as before)
        dm1.iloc[:, 0] = dm1.iloc[:, 0].str[:15]
        dm2.iloc[:, 0] = dm2.iloc[:, 0].str[:15]
        dm1_order = dm1.iloc[:, 0].tolist()
        dm2_order = dm2.iloc[:, 0].tolist()
        common_entries = list(set(dm1_order) & set(dm2_order))
        
        keep_indices_dm1 = [i for i, acc in enumerate(dm1_order) if acc in common_entries]
        keep_indices_dm2 = [i for i, acc in enumerate(dm2_order) if acc in common_entries]
        
        dm1 = dm1.iloc[keep_indices_dm1].reset_index(drop=True)
        dm2 = dm2.iloc[keep_indices_dm2].reset_index(drop=True)
        
        col_indices_dm1 = [0] + [i + 1 for i in keep_indices_dm1]
        col_indices_dm2 = [0] + [i + 1 for i in keep_indices_dm2]
        dm1 = dm1.iloc[:, col_indices_dm1]
        dm2 = dm2.iloc[:, col_indices_dm2]
        
        dm1_order = dm1.iloc[:, 0].tolist()
        dm2_order = dm2.iloc[:, 0].tolist()
        order_mapping = [dm2_order.index(x) for x in dm1_order]
        dm2 = dm2.iloc[order_mapping]
        dm2_cols = dm2.columns.tolist()
        dm2 = dm2[[dm2_cols[0]] + [dm2_cols[i+1] for i in order_mapping]]
        
        if to_remove is not None:
            to_remove_clean = [acc.strip() for acc in to_remove]
            keep_mask = ~dm1.iloc[:, 0].isin(to_remove_clean)
            keep_indices = np.where(keep_mask)[0]
            dm1 = dm1.iloc[keep_indices].reset_index(drop=True)
            dm2 = dm2.iloc[keep_indices].reset_index(drop=True)
            col_indices = [0] + [i + 1 for i in keep_indices]
            dm1 = dm1.iloc[:, col_indices]
            dm2 = dm2.iloc[:, col_indices]
        
        accessions = dm1.iloc[:, 0].tolist()
        dm1 = dm1.iloc[:, 1:]
        dm2 = dm2.iloc[:, 1:]
        dm1_array = dm1.to_numpy()
        dm2_array = dm2.to_numpy()
        dm1_array, dm2_array = filter_quantiles(dm1_array, dm2_array, quantile=quantile)
        
        # Create taxonomic mapping
        taxa_mapping = {acc: str(taxa_dict.get(acc, "Unknown")) for acc in accessions}
        
        # Get distances and create dataframe
        rows, cols = np.triu_indices(dm1_array.shape[0], k=1)
        dm1_flat = dm1_array[rows, cols]
        dm2_flat = dm2_array[rows, cols]
        
        df = pd.DataFrame({
            'x': dm1_flat,
            'y': dm2_flat,
            'taxa1': [taxa_mapping[accessions[i]] for i in rows],
            'taxa2': [taxa_mapping[accessions[i]] for i in cols],
        })
        
        # Calculate residuals and statistics
        mask = np.isfinite(df['x']) & np.isfinite(df['y'])
        clean_df = df[mask].copy()
        
        if len(clean_df) > 10:  # Need minimum points for regression
            X = clean_df['x'].values.reshape(-1, 1)
            y = clean_df['y'].values
            reg = LinearRegression().fit(X, y)
            y_pred = reg.predict(X)
            residuals = y - y_pred
            
            # Analyze residuals by taxonomic groups - separate within vs between class
            group_residuals_within = {}  # Within-class comparisons only
            group_residuals_between = {}  # Between-class comparisons only
            group_residuals_all = {}  # All comparisons (original approach)
            
            for i, (idx, row) in enumerate(clean_df.iterrows()):
                taxa_a, taxa_b = row['taxa1'], row['taxa2']
                residual = residuals[i]  # Use i instead of idx since residuals is 0-indexed
                
                is_within_class = (taxa_a == taxa_b)
                
                # Add to both taxa (each pairwise comparison involves two organisms)
                for taxon in [taxa_a, taxa_b]:
                    # All comparisons (original approach)
                    if taxon not in group_residuals_all:
                        group_residuals_all[taxon] = []
                    group_residuals_all[taxon].append(residual)
                    
                    # Within-class comparisons only
                    if is_within_class:
                        if taxon not in group_residuals_within:
                            group_residuals_within[taxon] = []
                        group_residuals_within[taxon].append(residual)
                    
                    # Between-class comparisons only  
                    else:
                        if taxon not in group_residuals_between:
                            group_residuals_between[taxon] = []
                        group_residuals_between[taxon].append(residual)
            
            # Calculate group statistics for all three types
            group_stats_all = {}
            group_stats_within = {}
            group_stats_between = {}
            
            # All comparisons (original approach)
            for group, residual_list in group_residuals_all.items():
                if len(residual_list) >= 5:  # Minimum sample size
                    group_stats_all[group] = {
                        'mean_residual': np.mean(residual_list),
                        'median_residual': np.median(residual_list),
                        'std_residual': np.std(residual_list),
                        'count': len(residual_list),
                        'comparison': dm_name,
                        'type': 'all'
                    }
            
            # Within-class comparisons only
            for group, residual_list in group_residuals_within.items():
                if len(residual_list) >= 3:  # Lower threshold since within-class pairs are rarer
                    group_stats_within[group] = {
                        'mean_residual': np.mean(residual_list),
                        'median_residual': np.median(residual_list),
                        'std_residual': np.std(residual_list),
                        'count': len(residual_list),
                        'comparison': dm_name,
                        'type': 'within'
                    }
            
            # Between-class comparisons only
            for group, residual_list in group_residuals_between.items():
                if len(residual_list) >= 5:  # Standard threshold
                    group_stats_between[group] = {
                        'mean_residual': np.mean(residual_list),
                        'median_residual': np.median(residual_list),
                        'std_residual': np.std(residual_list),
                        'count': len(residual_list),
                        'comparison': dm_name,
                        'type': 'between'
                    }
            
            # Store all three types
            all_deviations[dm_name] = {
                'all': group_stats_all,
                'within': group_stats_within,
                'between': group_stats_between
            }
            
            # Add to summary
            from scipy import stats as scipy_stats
            correlation, p_value = scipy_stats.pearsonr(clean_df['x'], clean_df['y'])
            summary_stats.append({
                'comparison': dm_name,
                'correlation': correlation,
                'p_value': p_value,
                'slope': reg.coef_[0],
                'intercept': reg.intercept_,
                'n_points': len(clean_df)
            })
    
    # Create comprehensive analysis plots
    os.makedirs('figures1', exist_ok=True)
    
    # 1. Summary plots of mean residuals across comparisons
    if all_deviations:
        # Create plots for each comparison type: all, within, between
        for comparison_type in ['all', 'within', 'between']:
            print(f"\nCreating heatmap for {comparison_type}-class comparisons...")
            
            # Get all unique groups for this comparison type
            all_groups = set()
            for comparison_stats in all_deviations.values():
                if comparison_type in comparison_stats:
                    all_groups.update(comparison_stats[comparison_type].keys())
            
            if not all_groups:
                print(f"No data for {comparison_type}-class comparisons, skipping...")
                continue
                
            all_groups = sorted(list(all_groups))
            
            # Create matrix of mean residuals
            residual_matrix = []
            comparison_names = []
            for comparison, stats_dict in all_deviations.items():
                if comparison_type in stats_dict:
                    stats = stats_dict[comparison_type]
                    residual_row = [stats.get(group, {}).get('mean_residual', np.nan) for group in all_groups]
                    residual_matrix.append(residual_row)
                    comparison_names.append(comparison.replace(' Enzyme Evolutionary', ''))
            
            if not residual_matrix:
                print(f"No data to plot for {comparison_type}-class comparisons")
                continue
                
            residual_matrix = np.array(residual_matrix)
        
            # Plot heatmap
            fig, ax = plt.subplots(figsize=(max(12, len(all_groups)*0.8), len(comparison_names)*0.8))
            
            # Create masked array to handle NaN values
            masked_matrix = np.ma.masked_invalid(residual_matrix)
            
            im = ax.imshow(masked_matrix, cmap='RdBu_r', aspect='auto', vmin=-0.1, vmax=0.1)
            
            ax.set_xticks(range(len(all_groups)))
            ax.set_xticklabels(all_groups, rotation=45, ha='right')
            ax.set_yticks(range(len(comparison_names)))
            ax.set_yticklabels(comparison_names)
            
            # Different titles for different comparison types
            if comparison_type == 'all':
                title = f'Mean Residuals by {group_level} - All Comparisons\n(Red = Above regression line, Blue = Below)'
            elif comparison_type == 'within':
                title = f'Mean Residuals by {group_level} - Within-Class Comparisons Only\n(Red = Above regression line, Blue = Below)'
            else:  # between
                title = f'Mean Residuals by {group_level} - Between-Class Comparisons Only\n(Red = Above regression line, Blue = Below)'
            
            ax.set_title(title, fontsize=14)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Mean Residual', rotation=270, labelpad=20)
            
            # Add text annotations for significant values
            for i in range(len(comparison_names)):
                for j in range(len(all_groups)):
                    if not np.isnan(residual_matrix[i, j]) and abs(residual_matrix[i, j]) > 0.02:
                        text_color = 'white' if abs(residual_matrix[i, j]) > 0.05 else 'black'
                        ax.text(j, i, f'{residual_matrix[i, j]:.3f}', 
                               ha='center', va='center', color=text_color, fontsize=8)
            
            plt.tight_layout()
            plt.savefig(f'figures1/deviation_heatmap_{group_level.lower()}_{comparison_type}.png', dpi=300, bbox_inches='tight')
            plt.savefig(f'figures1/deviation_heatmap_{group_level.lower()}_{comparison_type}.svg', format='svg', bbox_inches='tight')
            plt.close()
        
        # 2. Summary statistics tables for each comparison type
        print(f"\n{'='*80}")
        print(f"DEVIATION ANALYSIS SUMMARY ({group_level} level)")
        print(f"{'='*80}")
        
        # Analyze each comparison type separately
        for comparison_type in ['all', 'within', 'between']:
            print(f"\n{comparison_type.upper()}-CLASS COMPARISONS:")
            print("-" * 50)
            
            # Get all groups for this comparison type
            type_groups = set()
            for comparison_stats in all_deviations.values():
                if comparison_type in comparison_stats:
                    type_groups.update(comparison_stats[comparison_type].keys())
            
            if not type_groups:
                print(f"No data available for {comparison_type}-class comparisons")
                continue
            
            # Find consistently deviating groups
            group_deviation_summary = {}
            for group in type_groups:
                residuals_across_comparisons = []
                for comparison_stats in all_deviations.values():
                    if comparison_type in comparison_stats and group in comparison_stats[comparison_type]:
                        residuals_across_comparisons.append(comparison_stats[comparison_type][group]['mean_residual'])
                
                min_comparisons = 2 if comparison_type == 'within' else 3  # Lower threshold for within-class
                if len(residuals_across_comparisons) >= min_comparisons:
                    group_deviation_summary[group] = {
                        'mean_across_comparisons': np.mean(residuals_across_comparisons),
                        'std_across_comparisons': np.std(residuals_across_comparisons),
                        'n_comparisons': len(residuals_across_comparisons),
                        'consistent_direction': len([r for r in residuals_across_comparisons if r > 0]) / len(residuals_across_comparisons)
                    }
            
            if not group_deviation_summary:
                print(f"No groups with sufficient data for {comparison_type}-class analysis")
                continue
            
            # Sort by absolute mean deviation
            sorted_groups = sorted(group_deviation_summary.items(), 
                                 key=lambda x: abs(x[1]['mean_across_comparisons']), reverse=True)
            
            min_comps = 2 if comparison_type == 'within' else 3
            print(f"\nGroups with most consistent deviations (present in ≥{min_comps} comparisons):")
            print(f"{'Group':<25} {'Mean Residual':<15} {'Std':<10} {'N':<5} {'Direction':<20}")
            print("-" * 80)
            
            for group, stats in sorted_groups[:10]:
                direction = "Consistently above" if stats['consistent_direction'] > 0.7 else \
                           "Consistently below" if stats['consistent_direction'] < 0.3 else \
                           "Mixed"
                print(f"{group:<25} {stats['mean_across_comparisons']:<15.4f} {stats['std_across_comparisons']:<10.4f} "
                      f"{stats['n_comparisons']:<5} {direction:<20}")
        
        # Summary interpretation
        print(f"\n{'='*80}")
        print("INTERPRETATION GUIDE:")
        print("- WITHIN-class: How classes behave when comparing species within that class")
        print("- BETWEEN-class: How classes behave when compared to other classes") 
        print("- Negative residuals: Faster enzyme evolution than expected from BUSCO phylogeny")
        print("- Positive residuals: Slower enzyme evolution than expected from BUSCO phylogeny")
        print("- Note: For formal hypothesis testing, consider multiple testing correction")
        print(f"{'='*80}")
    
    return all_deviations, summary_stats

def plot_dms(dm1, dm2, dm1_name='dm1', dm2_name='dm2', to_remove=None, quantile=0.9999):
    # Clean up and standardize accession IDs
    dm1.iloc[:, 0] = dm1.iloc[:, 0].str[:15]
    dm2.iloc[:, 0] = dm2.iloc[:, 0].str[:15]
    
    dm1_order = dm1.iloc[:, 0].tolist()
    dm2_order = dm2.iloc[:, 0].tolist()
    
    # Find common entries between dm1 and dm2
    common_entries = list(set(dm1_order) & set(dm2_order))
    missing_entries = set(dm1_order) - set(dm2_order)
    if missing_entries:
        print(f"Warning: Following entries are missing from {dm2_name}:")
        print(missing_entries)
        print("Removing these entries from both matrices...")
    
    # Filter rows and corresponding columns for both matrices
    keep_indices_dm1 = [i for i, acc in enumerate(dm1_order) if acc in common_entries]
    keep_indices_dm2 = [i for i, acc in enumerate(dm2_order) if acc in common_entries]
    
    # Filter rows
    dm1 = dm1.iloc[keep_indices_dm1].reset_index(drop=True)
    dm2 = dm2.iloc[keep_indices_dm2].reset_index(drop=True)
    
    # Filter columns (excluding first column which contains accessions)
    col_indices_dm1 = [0] + [i + 1 for i in keep_indices_dm1]
    col_indices_dm2 = [0] + [i + 1 for i in keep_indices_dm2]
    
    dm1 = dm1.iloc[:, col_indices_dm1]
    dm2 = dm2.iloc[:, col_indices_dm2]
    
    # Update orders after filtering
    dm1_order = dm1.iloc[:, 0].tolist()
    dm2_order = dm2.iloc[:, 0].tolist()
    
    # Create mapping for reordering dm2
    order_mapping = [dm2_order.index(x) for x in dm1_order]
    
    # Reorder dm2 to match dm1's order
    dm2 = dm2.iloc[order_mapping]
    dm2_cols = dm2.columns.tolist()
    dm2 = dm2[[dm2_cols[0]] + [dm2_cols[i+1] for i in order_mapping]]

    # Remove specified accessions if to_remove is provided
    if to_remove is not None:
        # Clean up the accessions to remove (remove newlines and whitespace)
        to_remove = [acc.strip() for acc in to_remove]
        
        # Create mask for rows to keep
        keep_mask = ~dm1.iloc[:, 0].isin(to_remove)
        
        # Get integer indices from boolean mask
        keep_indices = np.where(keep_mask)[0]
        
        # Filter both matrices
        dm1 = dm1.iloc[keep_indices].reset_index(drop=True)
        dm2 = dm2.iloc[keep_indices].reset_index(drop=True)
        
        # Filter columns (excluding first column which contains accessions)
        col_indices = [0] + [i + 1 for i in keep_indices]  # +1 because first column is accessions
        dm1 = dm1.iloc[:, col_indices]
        dm2 = dm2.iloc[:, col_indices]

    # Continue with numerical part extraction
    accessions = dm1.iloc[:, 0].tolist()  # Save accessions before removing the column
    dm1 = dm1.iloc[:, 1:]
    dm2 = dm2.iloc[:, 1:]

    # Convert to numpy arrays
    dm1_array = dm1.to_numpy()
    dm2_array = dm2.to_numpy()
    
    # Filter matrices based on quantiles
    dm1_array, dm2_array = filter_quantiles(dm1_array, dm2_array, quantile=quantile)

    # Load taxa information to get phyla
    # Use taxa_csv argument instead of hardcoded path
    taxa_df = pd.read_csv(taxa_csv)
    
    # Clean up the Phylum column - replace NaN with "Unknown"
    taxa_df['Phylum'] = taxa_df['Phylum'].fillna("Unknown")
    
    # Create the dictionary
    taxa_dict = dict(zip(taxa_df['Accession'].str[:15], taxa_df['Phylum']))
    
    # Map accessions to phyla
    phyla = []
    missing_count = 0
    for acc in accessions:
        phylum = taxa_dict.get(acc, "Unknown")
        # Double-check for any remaining NaN values
        if pd.isna(phylum):
            phylum = "Unknown"
            missing_count += 1
        phyla.append(str(phylum))  # Ensure it's a string
    
    if missing_count > 0:
        print(f"Warning: {missing_count} accessions had NaN phyla values, assigned to 'Unknown'")
    
    print(f"Sample of phyla found: {list(set(phyla))[:10]}")  # Debug: show sample phyla
    
    # Hardcoded colors for self-comparisons
    preselected_colors = {
        "Ascomycota": "#377eb8",  # Blue
        "Basidiomycota": "#e41a1c",  # Red
        "Mucoromycota": "#4daf4a",  # Green
        "Zoopagomycota": "#984ea3",  # Purple
        "Chytridiomycota": "#ff7f00",  # Orange
        "Blastocladiomycota": "#ffff33",  # Yellow
        "Cryptomycota": "#a65628",  # Brown
    }

    default_color = '#BBBBBB'  # Gray for other comparisons

    # Create color mapping for self-pairs
    color_mapping_self = {
        (phylum, phylum): preselected_colors.get(phylum, default_color)
        for phylum in preselected_colors
    }

    # Get upper triangle indices with the original i, j positions
    rows, cols = np.triu_indices(dm1_array.shape[0], k=1)

    # Extract distances using these indices
    dm1_flat = dm1_array[rows, cols]
    dm2_flat = dm2_array[rows, cols]

    # Map color indices back to actual colors (not used directly by datashader, but kept for clarity)
    colors = [
        color_mapping_self.get((phyla[rows[i]], phyla[cols[i]]), default_color)
        for i in range(len(rows))
    ]

    print(dm1_array.shape)
    print(dm2_array.shape)
    print(dm1_flat.shape)
    print(dm2_flat.shape)

    # After getting dm1_flat and dm2_flat, create a DataFrame for datashader
    df = pd.DataFrame({
        'x': dm1_flat,
        'y': dm2_flat,
        'phylum_pair': pd.Categorical([
            f"{min(phyla[rows[i]], phyla[cols[i]])}-{max(phyla[rows[i]], phyla[cols[i]])}"
            for i in range(len(rows))
        ])
    })

    # Calculate min and max values for plot ranges (per-axis so X is not forced to Y span)
    x_min, x_max = df['x'].min(), df['x'].max()
    y_min, y_max = df['y'].min(), df['y'].max()
    if x_min == x_max:
        x_min -= 1e-9
        x_max += 1e-9
    if y_min == y_max:
        y_min -= 1e-9
        y_max += 1e-9
    x_min, x_max, y_min, y_max = square_ranges(x_min, x_max, y_min, y_max)

    # Build color key for datashader: self-pairs use preselected colors, others gray
    unique_pairs = df['phylum_pair'].unique()
    def parse_pair(p):
        a, b = str(p).split('-', 1)
        return a, b
    color_key = {}
    for pair in unique_pairs:
        a, b = parse_pair(pair)
        if a == b and a in preselected_colors:
            color_key[str(pair)] = preselected_colors[a]
        else:
            color_key[str(pair)] = default_color

    # Create both density and phylum-colored plots
    os.makedirs('figures1', exist_ok=True)  # Ensure figures1 directory exists

    # Build safe filenames to avoid overwrite across multiple matrices
    safe_dm1 = ''.join(c if (c.isalnum() or c in ('_', '-')) else '_' for c in dm1_name.replace(' ', '_'))
    safe_dm2 = ''.join(c if (c.isalnum() or c in ('_', '-')) else '_' for c in dm2_name.replace(' ', '_'))

    for plot_type in ['density', 'phylum']:
        # Create matplotlib figure and axis
        fig, ax = plt.subplots(figsize=(12, 12))

        canvas = ds.Canvas(plot_width=1600, plot_height=1600,
              x_range=(x_min, x_max),
              y_range=(y_min, y_max))

        if plot_type == 'density':
            agg = canvas.points(df, 'x', 'y')
            img = ds.tf.shade(agg, cmap=cc.bmy)

            # Create proper matplotlib colormap for the colorbar
            bmy_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('bmy', cc.bmy)
        else:
            # Use the same canvas size as density plot
            agg = canvas.points(df, 'x', 'y', agg=ds.count_cat('phylum_pair'))
            img = ds.tf.shade(agg, color_key=color_key, how='eq_hist')

        img = ds.tf.set_background(img, 'white')

        # Add diagonal line using the same canvas
        diag_x0 = max(x_min, y_min)
        diag_x1 = min(x_max, y_max)
        diagonal_df = pd.DataFrame({
            'x': [diag_x0, diag_x1],
            'y': [diag_x0, diag_x1]
        })
        diagonal_agg = canvas.line(diagonal_df, 'x', 'y')
        diagonal_img = ds.tf.shade(diagonal_agg, cmap=['black'])

        final_img = ds.transfer_functions.stack(img, diagonal_img)

        # Convert datashader image to matplotlib
        ax.imshow(final_img.to_pil(), extent=[x_min, x_max, y_min, y_max])
        if hasattr(ax, 'set_box_aspect'):
            ax.set_box_aspect(1)  # Keep square axes while allowing different x/y spans

        # Add axes labels and title
        ax.set_xlabel(f'{dm1_name} Distance (substitutions/site)', fontsize=18)
        ax.set_ylabel(f'{dm2_name} Distance (substitutions/site)', fontsize=18)
        title = 'Density Plot' if plot_type == 'density' else 'Phylum Pairs Plot'
        ax.set_title(title, fontsize=14)

        # Add color bar for density plot
        if plot_type == 'density':
            norm = matplotlib.colors.Normalize(vmin=0, vmax=agg.values.max())
            sm = plt.cm.ScalarMappable(cmap=bmy_cmap, norm=norm)
            cbar = plt.colorbar(sm, ax=ax, label='Density')
            # Make colorbar more compact
            cbar.ax.tick_params(labelsize=8)

        # Add legend for phylum plot (only self-pairs)
        elif plot_type == 'phylum':
            self_pairs_in_data = sorted({p for p in unique_pairs if parse_pair(p)[0] == parse_pair(p)[1]})
            legend_elements = [matplotlib.patches.Patch(facecolor=color_key[str(pair)],
                                                     label=str(pair))
                               for pair in self_pairs_in_data]
            if legend_elements:
                legend = ax.legend(handles=legend_elements,
                                   loc='center left',
                                   bbox_to_anchor=(1.02, 0.5),
                                   fontsize=12,
                                   ncol=1)
                legend.set_draggable(True)
                plt.setp(legend.get_texts(), linespacing=1.2)

        # Save with appropriate suffix and format
        suffix = '_density' if plot_type == 'density' else '_phylum'
        # For phylum plot, increase right margin to accommodate legend
        if plot_type == 'phylum':
            plt.subplots_adjust(right=0.85)
        # Name files as X_vs_Y where X is x-axis (dm1) and Y is y-axis (dm2)
        base = f'figures1/{safe_dm1}_vs_{safe_dm2}_datashader{suffix}'
        plt.savefig(f'{base}.png', dpi=600, bbox_inches='tight')
        plt.savefig(f'{base}.svg', format='svg', bbox_inches='tight')
        plt.close()

    print('Plots created successfully!')

def plot_dms_grid(dm_list, ref_dm, ref_name, dm_names, to_remove=None, quantile=0.9999, ncols=4, outdir='figures1', per_plot_y=False):
    """
    Create grid SVGs (density and phylum-colored) for multiple matrices vs a reference.
    - Standardizes X axis across all comparisons (global x range)
    - Optionally fits Y axis to each dataset (per_plot_y=True)
    - Keeps square plots by using equal canvas width/height and equal aspect
    """
    # Hardcoded colors for self-comparisons
    preselected_colors = {
        "Ascomycota": "#377eb8",  # Blue
        "Basidiomycota": "#e41a1c",  # Red
        "Mucoromycota": "#4daf4a",  # Green
        "Zoopagomycota": "#984ea3",  # Purple
        "Chytridiomycota": "#ff7f00",  # Orange
        "Blastocladiomycota": "#ffff33",  # Yellow
        "Cryptomycota": "#a65628",  # Brown
    }
    default_color = '#BBBBBB'

    prepared = []
    global_x_min = None
    global_x_max = None
    global_y_min = None
    global_y_max = None

    # Prepare all pairs and compute global X range (and local Y ranges)
    for dm, dm_name in zip(dm_list, dm_names):
        dm1 = dm.copy()
        dm2 = ref_dm.copy()
        dm1_name = dm_name
        dm2_name = ref_name

        dm1.iloc[:, 0] = dm1.iloc[:, 0].str[:15]
        dm2.iloc[:, 0] = dm2.iloc[:, 0].str[:15]
        dm1_order = dm1.iloc[:, 0].tolist()
        dm2_order = dm2.iloc[:, 0].tolist()
        common_entries = list(set(dm1_order) & set(dm2_order))
        keep_indices_dm1 = [i for i, acc in enumerate(dm1_order) if acc in common_entries]
        keep_indices_dm2 = [i for i, acc in enumerate(dm2_order) if acc in common_entries]
        dm1 = dm1.iloc[keep_indices_dm1].reset_index(drop=True)
        dm2 = dm2.iloc[keep_indices_dm2].reset_index(drop=True)
        col_indices_dm1 = [0] + [i + 1 for i in keep_indices_dm1]
        col_indices_dm2 = [0] + [i + 1 for i in keep_indices_dm2]
        dm1 = dm1.iloc[:, col_indices_dm1]
        dm2 = dm2.iloc[:, col_indices_dm2]
        dm1_order = dm1.iloc[:, 0].tolist()
        dm2_order = dm2.iloc[:, 0].tolist()
        order_mapping = [dm2_order.index(x) for x in dm1_order]
        dm2 = dm2.iloc[order_mapping]
        dm2_cols = dm2.columns.tolist()
        dm2 = dm2[[dm2_cols[0]] + [dm2_cols[i+1] for i in order_mapping]]
        if to_remove is not None:
            to_remove_clean = [acc.strip() for acc in to_remove]
            keep_mask = ~dm1.iloc[:, 0].isin(to_remove_clean)
            keep_indices = np.where(keep_mask)[0]
            dm1 = dm1.iloc[keep_indices].reset_index(drop=True)
            dm2 = dm2.iloc[keep_indices].reset_index(drop=True)
            col_indices = [0] + [i + 1 for i in keep_indices]
            dm1 = dm1.iloc[:, col_indices]
            dm2 = dm2.iloc[:, col_indices]
        accessions = dm1.iloc[:, 0].tolist()
        dm1 = dm1.iloc[:, 1:]
        dm2 = dm2.iloc[:, 1:]
        dm1_array = dm1.to_numpy()
        dm2_array = dm2.to_numpy()
        dm1_array, dm2_array = filter_quantiles(dm1_array, dm2_array, quantile=quantile)
        # Use taxa_csv argument instead of hardcoded path
        taxa_df = pd.read_csv(taxa_csv)
        taxa_df['Phylum'] = taxa_df['Phylum'].fillna("Unknown")
        taxa_dict = dict(zip(taxa_df['Accession'].str[:15], taxa_df['Phylum']))
        phyla = [str(taxa_dict.get(acc, "Unknown")) for acc in accessions]
        rows, cols = np.triu_indices(dm1_array.shape[0], k=1)
        dm1_flat = dm1_array[rows, cols]
        dm2_flat = dm2_array[rows, cols]
        df = pd.DataFrame({
            'x': dm1_flat,
            'y': dm2_flat,
            'phylum_pair': pd.Categorical([
                f"{min(phyla[rows[i]], phyla[cols[i]])}-{max(phyla[rows[i]], phyla[cols[i]])}"
                for i in range(len(rows))
            ])
        })
        # Update global X range
        local_x_min = df['x'].min()
        local_x_max = df['x'].max()
        global_x_min = local_x_min if global_x_min is None else min(global_x_min, local_x_min)
        global_x_max = local_x_max if global_x_max is None else max(global_x_max, local_x_max)
        # Local Y range
        local_y_min = df['y'].min()
        local_y_max = df['y'].max()
        if local_y_min == local_y_max:
            eps = 1e-9
            local_y_min -= eps
            local_y_max += eps
        global_y_min = local_y_min if global_y_min is None else min(global_y_min, local_y_min)
        global_y_max = local_y_max if global_y_max is None else max(global_y_max, local_y_max)
        prepared.append({
            'df': df,
            'dm1_name': dm1_name,
            'dm2_name': dm2_name,
            'y_min': float(local_y_min),
            'y_max': float(local_y_max)
        })

    os.makedirs(outdir, exist_ok=True)

    n = len(prepared)
    ncols = ncols if n > 0 else 1
    nrows = math.ceil(n / ncols) if n > 0 else 1

    def add_labels(ax, row_idx, col_idx, dm1_label, dm2_label):
        if col_idx == 0:
            ax.set_ylabel(f'{dm2_label} Distance (substitutions/site)', fontsize=10)
        else:
            ax.set_yticklabels([])
        if row_idx == nrows - 1:
            ax.set_xlabel(f'{dm1_label} Distance (substitutions/site)', fontsize=10)
        else:
            ax.set_xticklabels([])
        # Keep the axes box square regardless of data ranges
        if hasattr(ax, 'set_box_aspect'):
            ax.set_box_aspect(1)
        else:
            ax.set_aspect('auto', adjustable='box')

    # Create density grid
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    axes = np.atleast_2d(axes)
    for idx, item in enumerate(prepared):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        df = item['df']
        if per_plot_y:
            y_min, y_max = item['y_min'], item['y_max']
        else:
            y_min, y_max = global_y_min, global_y_max
        canvas = ds.Canvas(plot_width=800, plot_height=800,
                           x_range=(global_x_min, global_x_max),
                           y_range=(y_min, y_max))
        agg = canvas.points(df, 'x', 'y')
        img = ds.tf.shade(agg, cmap=cc.bmy)
        img = ds.tf.set_background(img, 'white')
        # Diagonal segment within current axes
        x0 = max(global_x_min, y_min)
        x1 = min(global_x_max, y_max)
        if x1 > x0:
            diagonal_df = pd.DataFrame({'x': [x0, x1], 'y': [x0, x1]})
            diagonal_agg = canvas.line(diagonal_df, 'x', 'y')
            diagonal_img = ds.tf.shade(diagonal_agg, cmap=['black'])
            img = ds.transfer_functions.stack(img, diagonal_img)
        ax.imshow(img.to_pil(), extent=[global_x_min, global_x_max, y_min, y_max], aspect='auto')
        ax.set_title(f"{item['dm1_name']} vs {item['dm2_name']}", fontsize=11)
        add_labels(ax, r, c, item['dm1_name'], item['dm2_name'])
    # Hide any unused axes
    for j in range(idx+1, nrows*ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis('off')
    plt.tight_layout()
    fig.savefig(f'{outdir}/grid_density.svg', format='svg', bbox_inches='tight')
    plt.close(fig)

    # Create phylum-colored grid
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    axes = np.atleast_2d(axes)
    legend_pairs = set()
    for idx, item in enumerate(prepared):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        df = item['df']
        if per_plot_y:
            y_min, y_max = item['y_min'], item['y_max']
        else:
            y_min, y_max = global_y_min, global_y_max
        unique_pairs = df['phylum_pair'].unique()
        def parse_pair(p):
            a, b = str(p).split('-', 1)
            return a, b
        color_key = {}
        for pair in unique_pairs:
            a, b = parse_pair(pair)
            if a == b and a in preselected_colors:
                color_key[str(pair)] = preselected_colors[a]
                legend_pairs.add(str(pair))
            else:
                color_key[str(pair)] = default_color
        canvas = ds.Canvas(plot_width=800, plot_height=800,
                           x_range=(global_x_min, global_x_max),
                           y_range=(y_min, y_max))
        agg = canvas.points(df, 'x', 'y', agg=ds.count_cat('phylum_pair'))
        img = ds.tf.shade(agg, color_key=color_key, how='eq_hist')
        img = ds.tf.set_background(img, 'white')
        x0 = max(global_x_min, y_min)
        x1 = min(global_x_max, y_max)
        if x1 > x0:
            diagonal_df = pd.DataFrame({'x': [x0, x1], 'y': [x0, x1]})
            diagonal_agg = canvas.line(diagonal_df, 'x', 'y')
            diagonal_img = ds.tf.shade(diagonal_agg, cmap=['black'])
            img = ds.transfer_functions.stack(img, diagonal_img)
        ax.imshow(img.to_pil(), extent=[global_x_min, global_x_max, y_min, y_max], aspect='auto')
        ax.set_title(f"{item['dm1_name']} vs {item['dm2_name']}", fontsize=11)
        add_labels(ax, r, c, item['dm1_name'], item['dm2_name'])
    for j in range(idx+1, nrows*ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis('off')
    if legend_pairs:
        legend_elements = [matplotlib.patches.Patch(facecolor=preselected_colors[p.split('-')[0]], label=p)
                           for p in sorted(legend_pairs)]
        fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.02, 0.5), fontsize=12)
        plt.subplots_adjust(right=0.85)
    plt.tight_layout()
    fig.savefig(f'{outdir}/grid_phylum.svg', format='svg', bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot distance matrices vs reference with datashader and deviation analysis')
    parser.add_argument('--grid-only', action='store_true', help='Only generate grid SVGs (no individual plots)')
    parser.add_argument('--individual-only', action='store_true', help='Only generate individual plots (no grids)')
    parser.add_argument('--deviation-analysis', action='store_true', help='Perform detailed deviation analysis by taxonomic groups')
    parser.add_argument('--enhanced-plots', action='store_true', help='Create enhanced plots with residual analysis')
    parser.add_argument('--taxonomic-level', choices=['Phylum', 'Class'], default='Phylum', 
                        help='Taxonomic level for deviation analysis (Phylum or Class)')
    parser.add_argument('--cols', type=int, default=4, help='Number of columns in the grid layout')
    parser.add_argument('--quantile', type=float, default=0.9999, help='Quantile for filtering outliers')
    parser.add_argument('--outdir', type=str, required=True, help='Output directory for figures')
    parser.add_argument('--taxa-csv', type=str, required=True, help='Path to taxa CSV file (deduplicated_genomes_with_traits.csv)')
    parser.add_argument('--outliers', type=str, required=True, help='File with outlier accession IDs to remove')
    args, unknown = parser.parse_known_args()

    with open(args.outliers, 'r') as file:
        to_remove = file.readlines()

    parser.add_argument('--ref-dm', required=True, help='Reference distance matrix CSV file')
    parser.add_argument('--ref-name', required=True, help='Reference matrix name')
    parser.add_argument('--dm-paths', nargs='+', required=True, help='List of distance matrix CSV files to compare')
    parser.add_argument('--dm-names', nargs='+', required=True, help='Names for each distance matrix')
    parser.add_argument('--outliers', required=True, help='File with outlier accession IDs to remove')
    args, unknown = parser.parse_known_args()

    with open(args.outliers, 'r') as file:
        to_remove = file.readlines()

    ref_dm = load_distance_matrix(args.ref_dm)
    ref_name = args.ref_name
    dm_paths = args.dm_paths
    dm_names = args.dm_names
    dm_list = [load_distance_matrix(path) for path in dm_paths]

    # Standard plots
    if not args.grid_only and not args.deviation_analysis and not args.enhanced_plots:
        plot_dms_multiple(dm_list, ref_dm, ref_name, dm_names, to_remove=to_remove, quantile=args.quantile, output_dir=args.outdir)
    if not args.individual_only and not args.deviation_analysis and not args.enhanced_plots:
        plot_dms_grid(dm_list, ref_dm, ref_name, dm_names, to_remove=to_remove, quantile=args.quantile, ncols=args.cols, outdir=args.outdir, per_plot_y=True)
    if args.enhanced_plots:
        print(f"\nCreating enhanced plots with {args.taxonomic_level} level analysis...")
        for dm, dm_name in zip(dm_list, dm_names):
            plot_dms_with_deviations(dm, ref_dm, dm_name, ref_name, to_remove=to_remove, quantile=args.quantile, group_level=args.taxonomic_level, output_dir=args.outdir, taxa_csv=args.taxa_csv)
    if args.deviation_analysis:
        print(f"\nPerforming comprehensive deviation analysis at {args.taxonomic_level} level...")
        deviation_results, summary_stats = create_deviation_focused_plots(
            dm_list, ref_dm, ref_name, dm_names, to_remove=to_remove, quantile=args.quantile, group_level=args.taxonomic_level, output_dir=args.outdir, taxa_csv=args.taxa_csv)
        import json
        with open(f'{args.outdir}/deviation_analysis_{args.taxonomic_level.lower()}.json', 'w') as f:
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            serializable_results = {}
            for comparison, stats_dict in deviation_results.items():
                serializable_results[comparison] = {}
                for comparison_type, stats in stats_dict.items():
                    serializable_results[comparison][comparison_type] = {}
                    for group, group_stats in stats.items():
                        serializable_results[comparison][comparison_type][group] = {k: convert_numpy(v) for k, v in group_stats.items()}
            json.dump({
                'deviation_results': serializable_results,
                'summary_stats': [{k: convert_numpy(v) for k, v in stat.items()} for stat in summary_stats]
            }, f, indent=2)
        
        print(f"\nDetailed results saved to {args.outdir}/deviation_analysis_{args.taxonomic_level.lower()}.json")