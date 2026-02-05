import sys
import csv
from Bio import Entrez
import multiprocessing as mp

import argparse

parser = argparse.ArgumentParser(description='Fetch lineage data from NCBI Entrez for a list of accessions.')
parser.add_argument('--email', required=True, help='Email address for NCBI Entrez')
parser.add_argument('--input_file', required=True, help='Input CSV file with accessions')
parser.add_argument('--output_file', required=True, help='Output CSV file for lineage data')
args = parser.parse_args()

Entrez.email = args.email
INPUT_FILE = args.input_file
OUTPUT_FILE = args.output_file

def fetch_lineage(accession):
    # First get assembly info to get taxid
    search_handle = Entrez.esearch(db="assembly", term=accession)
    search_record = Entrez.read(search_handle)
    search_handle.close()
    if not search_record["IdList"]:
        return {"error": "Not Found"}

    # Get assembly summary to extract taxid
    asm_id = search_record["IdList"][0]
    sum_handle = Entrez.esummary(db="assembly", id=asm_id)
    sum_record = Entrez.read(sum_handle)
    sum_handle.close()
    
    doc = sum_record["DocumentSummarySet"]["DocumentSummary"][0]
    taxid = doc.get("Taxid", "")
    if not taxid:
        return {"error": "No Taxid Found"}

    # Get taxonomy information
    tax_handle = Entrez.efetch(db="taxonomy", id=taxid)
    tax_record = Entrez.read(tax_handle)
    tax_handle.close()
    
    if not tax_record:
        return {"error": "No Taxonomy Found"}
    
    # Extract full lineage information
    lineage_data = {
        "species": tax_record[0].get("ScientificName", "Unknown"),
        "kingdom": "Unknown",
        "phylum": "Unknown", 
        "class": "Unknown",
        "order": "Unknown",
        "family": "Unknown",
        "genus": "Unknown"
    }
    
    # Parse the lineage to extract specific ranks
    lineage = tax_record[0].get("LineageEx", [])
    for item in lineage:
        rank = item.get("Rank", "").lower()
        name = item.get("ScientificName", "Unknown")
        
        if rank in lineage_data:
            lineage_data[rank] = name
    
    return lineage_data

def get_processed_accessions(output_file):
    """Read already processed accessions from output file"""
    processed = set()
    try:
        with open(output_file, 'r') as f:
            csv_reader = csv.reader(f)
            next(csv_reader)  # Skip header
            for row in csv_reader:
                if row:  # Check if row is not empty
                    processed.add(row[0])
    except FileNotFoundError:
        pass
    return processed

def main():
    # Create results directory if it doesn't exist
    import os
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # Get already processed accessions
    processed_accessions = get_processed_accessions(OUTPUT_FILE)
    
    # Open output file in CSV format
    with open(OUTPUT_FILE, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not processed_accessions:
            writer.writerow(['Accession', 'Species', 'Genus', 'Family', 'Order', 'Class', 'Phylum', 'Kingdom'])
        
        # Read input file and process each line
        with open(INPUT_FILE, 'r') as infile:
            next(infile)  # Skip the first line
            for line in infile:
                line = line.strip()
                if not line:
                    continue
                
                accession = line[:15]  # Take first 15 characters
                if accession in processed_accessions:
                    print(f"Skipping already processed accession: {accession}")
                    continue
                
                print(f"Fetching lineage for: {accession}")
                try:
                    lineage_data = fetch_lineage(accession)
                    if "error" in lineage_data:
                        print(f"Error: {lineage_data['error']}")
                        writer.writerow([accession, lineage_data['error'], "", "", "", "", "", ""])
                    else:
                        print(f"Species: {lineage_data['species']}")
                        writer.writerow([
                            accession,
                            lineage_data['species'],
                            lineage_data['genus'],
                            lineage_data['family'],
                            lineage_data['order'],
                            lineage_data['class'],
                            lineage_data['phylum'],
                            lineage_data['kingdom']
                        ])
                except Exception as e:
                    print(f"Error fetching lineage for {accession}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
