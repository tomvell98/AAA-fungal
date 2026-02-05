import csv
import argparse
import os
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description='Merge stats files into a matrix.')
    parser.add_argument('--inputs', nargs='+', required=True, help='Input stats files (tab-separated)')
    parser.add_argument('--output', required=True, help='Output merged stats matrix file')
    args = parser.parse_args()

    all_keys = set()
    file_counts = {}
    for fname in args.inputs:
        counts = defaultdict(int)
        with open(fname, newline='') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                key = (row['Column'], row['Value'])
                counts[key] += int(row['Count'])
                all_keys.add(key)
        file_counts[os.path.basename(fname)] = counts

    all_keys = sorted(all_keys)
    header = ['File'] + [f"{col}:{val}" for (col, val) in all_keys]

    with open(args.output, 'w') as out:
        out.write('\t'.join(header) + '\n')
        for fname in sorted(file_counts):
            row = [fname]
            counts = file_counts[fname]
            for key in all_keys:
                row.append(str(counts.get(key, 0)))
            out.write('\t'.join(row) + '\n')

if __name__ == '__main__':
    main()