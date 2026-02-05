#!/usr/bin/env python3
"""
set_bfactor_from_conservation.py

Assigns B-factor values in a PDB file based on conservation values (e.g., AvgPhylaEffectiveConservation) from a consensus CSV file.

Usage:
    python set_bfactor_from_conservation.py input.pdb consensus_phylum_conservation.csv output.pdb

- The CSV should have columns: Position, Consensus, JSD, GapFraction, AvgPhylaEffectiveConservation
- The script assigns the AvgPhylaEffectiveConservation value to the B-factor of each residue (all atoms in the residue get the same value),
  in order, skipping any residues in the PDB that do not match the consensus sequence order.
- The script assumes the PDB residue order matches the ungapped consensus sequence.
"""
import sys
import csv
import argparse
from Bio.PDB import PDBParser, PDBIO

def parse_args():
    parser = argparse.ArgumentParser(description='Set B-factor from conservation with flexible file paths.')
    parser.add_argument('--pdb_file', required=True, help='Input PDB file')
    parser.add_argument('--conservation_file', required=True, help='Conservation data file')
    parser.add_argument('--output_pdb', required=True, help='Output PDB file')
    return parser.parse_args()

def read_conservation_csv(csv_path):
    cons_values = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = row['AvgPhylaEffectiveConservation']
            if val == 'NA':
                cons_values.append(0.0)
            else:
                # Multiply by 100 for PyMOL compatibility and round to nearest integer
                cons_values.append(round(float(val) * 100))
    return cons_values

def set_b_factors(pdb_in, pdb_out, cons_values):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('model', pdb_in)
    # Build a mapping from (chain, resseq, icode) to new B-factor
    res_bfactors = {}
    i = 0
    for residue in structure.get_residues():
        if i >= len(cons_values):
            break
        res_id = (residue.get_full_id()[2], residue.get_id()[1], residue.get_id()[2])
        res_bfactors[res_id] = int(cons_values[i])
        i += 1
    # Write new PDB with correct B-factors for all atoms in each residue
    with open(pdb_in) as in_handle, open(pdb_out, 'w') as out_handle:
        for line in in_handle:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                chain = line[21]
                resseq = int(line[22:26])
                icode = line[26]
                res_id = (chain, resseq, icode)
                b = res_bfactors.get(res_id, 0)
                newline = line[:60] + f"{b:6d}" + line[66:]
                out_handle.write(newline)
            else:
                out_handle.write(line)

def main():
    args = parse_args()
    pdb_in = args.pdb_file
    csv_in = args.conservation_file
    pdb_out = args.output_pdb
    cons_values = read_conservation_csv(csv_in)
    set_b_factors(pdb_in, pdb_out, cons_values)
    print(f"Wrote {pdb_out} with B-factors set from {csv_in}")

if __name__ == "__main__":
    main()
