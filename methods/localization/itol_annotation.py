import argparse
import csv
import sys

def write_itol_annotation(input_csv, output_txt, color_map, legend_shape_map, legend_categories, maxcat_map):
    with open(input_csv, newline='') as csvfile, open(output_txt, 'w') as outfile:
        reader = csv.DictReader(csvfile)
        outfile.write(f"DATASET_COLORSTRIP\nSEPARATOR TAB\nDATASET_LABEL\tPredicted_Category\nCOLOR\t#ff0000\n\n")
        # Write legend
        outfile.write("LEGEND_TITLE\tPredicted_Category\n")
        outfile.write("LEGEND_SHAPES\t" + "\t".join([legend_shape_map.get(k, "1") for k in legend_categories]) + "\n")
        outfile.write("LEGEND_COLORS\t" + "\t".join([color_map.get(k, "#cccccc") for k in legend_categories]) + "\n")
        outfile.write("LEGEND_LABELS\t" + "\t".join(legend_categories) + "\n\n")
        outfile.write("DATA\n")
        for row in reader:
            protid = row['Protein_ID']
            value = maxcat_map[protid] if protid in maxcat_map else "Unknown"
            color = color_map.get(value, "#cccccc")
            outfile.write(f"{protid}\t{color}\t{value}\n")


def get_max_category_map(input_csv, prob_cols):
    maxcat_map = {}
    categories = set()
    with open(input_csv, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            maxval = -float('inf')
            maxcat = "Unknown"
            for col in prob_cols:
                try:
                    val = float(row[col])
                except (ValueError, KeyError):
                    val = float('-inf')
                if val > maxval:
                    maxval = val
                    maxcat = col
            maxcat_map[row['Protein_ID']] = maxcat
            categories.add(maxcat)
    return maxcat_map, sorted(categories)

def assign_colors_shapes(categories):
    # Color palette (extend as needed)
    palette = [
        "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a65628", "#f781bf", "#999999", "#cccccc",
        "#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99", "#e31a1c", "#fdbf6f", "#ff7f00", "#cab2d6", "#6a3d9a"
    ]
    shapes = ["1"]
    color_map = {}
    shape_map = {}
    for i, cat in enumerate(categories):
        color_map[cat] = palette[i % len(palette)]
        shape_map[cat] = shapes[i % len(shapes)]
    return color_map, shape_map

def main():
    parser = argparse.ArgumentParser(description="Generate iTOL color strip annotation from CSV.")
    parser.add_argument("--input_csv", required=True, help="Input CSV file path")
    parser.add_argument("--output_localization", required=True, help="Output iTOL file for Localizations")
    parser.add_argument("--output_signal", required=True, help="Output iTOL file for Signals")
    parser.add_argument("--output_stats", required=False, default=None, help="Output statistics file (tab-separated)")
    args = parser.parse_args()


    # Identify probability columns (columns 5-14, 0-based index 4-13)
    with open(args.input_csv, newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)

    prob_cols = header[4:14]
    legend_categories = prob_cols  # Always use all possible columns for legend order and color assignment

    # Get max category for each row and all unique categories
    maxcat_map, _ = get_max_category_map(args.input_csv, prob_cols)
    color_map, shape_map = assign_colors_shapes(legend_categories)


    write_itol_annotation(args.input_csv, args.output_localization, color_map, shape_map, legend_categories, maxcat_map)

    # --- SIGNALS OUTPUT ---
    # Collect all unique signals and build a map for each protein
    signals_map = {}
    signals_set = set()
    EMPTY_SIGNAL_LABEL = "(empty)"
    with open(args.input_csv, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            protid = row['Protein_ID']
            raw_signal = row.get('Signals', '')
            signal = raw_signal if raw_signal.strip() else EMPTY_SIGNAL_LABEL
            signals_map[protid] = signal
            signals_set.add(signal)
    signals_categories = sorted(signals_set)
    signals_color_map, signals_shape_map = assign_colors_shapes(signals_categories)
    # Assign a distinct color for empty signal (e.g., gray)
    signals_color_map[EMPTY_SIGNAL_LABEL] = "#bdbdbd"  # light gray
    # Write the signal annotation file
    write_itol_annotation(args.input_csv, args.output_signal, signals_color_map, signals_shape_map, signals_categories, signals_map)

    # Statistics output
    if args.output_stats:
        from collections import Counter
        cat_counter = Counter()
        signals_counter = Counter()
        # Read the input CSV again to get the 'Signals' column
        with open(args.input_csv, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Count predicted category
                v = maxcat_map.get(row['Protein_ID'], 'Unknown')
                cat_counter[v] += 1
                # Count Signals column if present
                if 'Signals' in row:
                    signals_counter[row['Signals']] += 1
        with open(args.output_stats, 'w') as statfile:
            statfile.write("Column\tValue\tCount\n")
            for k, v in cat_counter.items():
                statfile.write(f"Predicted_Category\t{k}\t{v}\n")
            for k, v in signals_counter.items():
                statfile.write(f"Signals\t{k}\t{v}\n")

    print(f"input_csv={args.input_csv}")
    print(f"output_localization={args.output_localization}")
    if args.output_stats:
        print(f"output_stats={args.output_stats}")

if __name__ == "__main__":
    main()