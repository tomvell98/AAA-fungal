#!/usr/bin/env python3

import sys
import csv
import subprocess
import freesasa
import argparse

def run_dssp(pdb_path, dssp_path):
    subprocess.run(['mkdssp', pdb_path, dssp_path], check=True)

def parse_dssp(dssp_path):
    dssp_data = []
    with open(dssp_path) as f:
        in_data = False
        for line in f:
            if line.startswith('  #  RESIDUE AA'):
                in_data = True
                continue
            if in_data and line.strip():
                # DSSP columns: https://swift.cmbi.umcn.nl/gv/dssp/
                try:
                    resseq = int(line[5:10].strip())
                    chain = line[11]
                    aa = line[13]
                    ss = line[16]
                    rsa = line[35:38].strip()
                    dssp_data.append({'Chain': chain, 'ResSeq': resseq, 'AA': aa, 'SecStruct': ss, 'DSSP_RSA': rsa})
                except Exception:
                    continue
    return dssp_data

def run_freesasa(pdb_path):
    structure = freesasa.Structure(pdb_path)
    result = freesasa.calc(structure)
    # Map: (chain, resseq) -> ASA
    residue_asa = {}
    for i in range(structure.nAtoms()):
        chain = structure.chainLabel(i)
        resnum = structure.residueNumber(i)
        resname = structure.residueName(i)
        # FreeSASA uses residue numbers as in the PDB, which may be string (with insertion codes)
        # To match with DSSP and output, convert to int if possible
        try:
            resnum_int = int(resnum)
        except Exception:
            resnum_int = resnum
        key = (chain, resnum_int)
        residue_asa.setdefault(key, 0)
        residue_asa[key] += result.atomArea(i)
    # Debug: print a few entries
    print("FreeSASA per-residue ASA (first 10):", list(residue_asa.items())[:10])
    return residue_asa
    return asa_map

def read_conservation_csv(consv_csv):
    consv_map = {}
    with open(consv_csv) as f:
        reader = csv.DictReader(f)
        # Determine if conservation file has Chain column
        has_chain = 'Chain' in reader.fieldnames
        # Use Position if Chain is missing
        for row in reader:
            if has_chain:
                chain = row.get('Chain')
                resseq = row.get('ResSeq')
                try:
                    resseq = int(resseq)
                except Exception:
                    pass
                key = (chain, resseq)
            else:
                # Use Position as key
                try:
                    pos = int(row.get('Position'))
                except Exception:
                    pos = row.get('Position')
                key = pos
            consv_map[key] = row
    return consv_map, has_chain

def merge_and_write(dssp_data, asa_map, out_csv, consv_map=None):
    with open(out_csv, 'w', newline='') as f:
        consv_cols = []
        has_chain = False
        if consv_map:
            # consv_map is now a tuple (map, has_chain)
            if isinstance(consv_map, tuple):
                consv_map, has_chain = consv_map
            for v in consv_map.values():
                consv_cols = [k for k in v.keys() if k not in ('Chain','ResSeq','Position')]
                break
        fieldnames = ['Chain', 'ResSeq', 'AA', 'SecStruct', 'DSSP_RSA', 'FreeSASA_ASA'] + consv_cols
        f.write(','.join(fieldnames) + '\n')
        missing = 0
        for row in dssp_data:
            chain = row['Chain']
            resseq = row['ResSeq']
            asa = asa_map.get((chain, resseq))
            if asa is None:
                try:
                    asa = asa_map.get((chain, int(resseq)))
                except Exception:
                    asa = None
            if asa is None:
                asa = 0.0
                missing += 1
            out_row = [chain, resseq, row['AA'], row['SecStruct'], row['DSSP_RSA'], asa]
            # Add conservation columns if available
            if consv_map:
                if has_chain:
                    consv = consv_map.get((chain, resseq))
                else:
                    consv = consv_map.get(resseq)
                if consv:
                    out_row += [consv.get(col, 'NA') for col in consv_cols]
                else:
                    out_row += ['NA'] * len(consv_cols)
            f.write(','.join(map(str, out_row)) + '\n')
        print(f"[merge_and_write] Residues missing FreeSASA ASA: {missing} / {len(dssp_data)}")

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze structures and merge DSSP, FreeSASA, and optional conservation data.')
    parser.add_argument('--pdb', required=True, help='Input PDB file path')
    parser.add_argument('--output', required=True, help='Output CSV file path')
    parser.add_argument('--conservation', required=False, help='Optional conservation CSV file path')
    return parser.parse_args()

def main():
    args = parse_args()
    pdb_path = args.pdb
    out_csv = args.output
    consv_map = None
    if args.conservation:
        consv_map = read_conservation_csv(args.conservation)
    dssp_path = 'out.dssp'
    run_dssp(pdb_path, dssp_path)
    dssp_data = parse_dssp(dssp_path)
    asa_map = run_freesasa(pdb_path)
    merge_and_write(dssp_data, asa_map, out_csv, consv_map)
    print(f"Wrote {out_csv}")

if __name__ == "__main__":
    main()