import argparse
import csv
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Compute phyla x localization heatmap from enzyme prediction CSV and phyla info.")
    parser.add_argument('--input_csv', required=True, help='Input enzyme prediction CSV (e.g. lys1.csv)')
    parser.add_argument('--phyla_csv', required=True, help='CSV with at least columns: Protein_ID, phylum')
    parser.add_argument('--output', required=True, help='Output TSV file for heatmap matrix')
    args = parser.parse_args()

    # Read enzyme prediction CSV
    df = pd.read_csv(args.input_csv)
    # Read phyla info CSV
    phyla_df = pd.read_csv(args.phyla_csv)


    # Merge on Accession (phyla file) and Protein_ID (enzyme file)
    merged = pd.merge(df, phyla_df[['Accession', 'Phylum']], left_on='Protein_ID', right_on='Accession', how='inner')

    # Columns 5-16 (0-based index 4-15) are the localizations
    loc_cols = list(df.columns[4:14])


    # Get the 6 most common phyla (or all if less than 6)
    phyla_counts = merged['Phylum'].value_counts().index[:6]
    phyla = list(phyla_counts)

    # Prepare output matrix: phyla x localization
    heatmap = pd.DataFrame(index=phyla, columns=loc_cols, dtype=float)

    for phylum in phyla:
        phylum_df = merged[merged['Phylum'] == phylum]
        for loc in loc_cols:
            avg = phylum_df[loc].astype(float).mean() if not phylum_df.empty else np.nan
            heatmap.loc[phylum, loc] = avg


    # Write to TSV
    heatmap.to_csv(args.output, sep='\t', float_format='%.4f', na_rep='NA')
    print(f"Heatmap written to {args.output}")

    # Plot heatmap
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import colorcet as cc
        plt.figure(figsize=(1.5*len(heatmap.columns), 1+len(heatmap.index)))
        sns.heatmap(heatmap, annot=True, fmt='.2f', cmap=cc.m_bmy, cbar_kws={'label': 'Avg. Probability'})
        plt.title('Average Predicted Probability by Phylum and Localization')
        plt.ylabel('Phylum')
        plt.xlabel('Localization')
        plt.tight_layout()
        png_out = args.output.rsplit('.', 1)[0] + '.png'
        plt.savefig(png_out, dpi=200)
        plt.close()
        print(f"Heatmap plot saved to {png_out}")
    except ImportError:
        print("matplotlib, seaborn, or colorcet not installed: skipping plot.")

if __name__ == "__main__":
    main()
