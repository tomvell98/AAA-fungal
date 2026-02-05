#!/usr/bin/env python3
"""
Script to deduplicate genomes based on species names and quality metrics.

Logic:
1. Group genomes by species name
2. For each species group:
   - If any accession starts with 'GCF', prefer those over 'GCA'
   - Among preferred type (GCF or GCA), select based on:
     a) Highest complete_buscos
     b) If tied, highest single_copy_buscos  
     c) If still tied, lowest missing_buscos
"""

import csv
import sys
from collections import defaultdict
import os

def read_lineage_busco_data(data_file):
    """Read the combined lineage and BUSCO data"""
    data = {}
    with open(data_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            accession = row['Accession']
            data[accession] = {
                'species': row['Species'],
                'genus': row['Genus'],
                'family': row['Family'],
                'order': row['Order'],
                'class': row['Class'],
                'phylum': row['Phylum'],
                'kingdom': row['Kingdom'],
                'complete_buscos': int(row['complete_buscos']),
                'single_copy_buscos': int(row['single_copy_buscos']),
                'fragmented_buscos': int(row['fragmented_buscos']),
                'missing_buscos': int(row['missing_buscos'])
            }
    return data

def get_quality_score(data):
    """Return tuple for sorting: (complete_buscos, single_copy_buscos, -missing_buscos)"""
    return (
        data['complete_buscos'],
        data['single_copy_buscos'], 
        -data['missing_buscos']  # Negative because we want lowest missing
    )

def deduplicate_genomes(data_file, output_file):
    """Main deduplication function"""
    
    # Read data
    print("Reading lineage and BUSCO data...")
    genome_data = read_lineage_busco_data(data_file)
    
    # Group by species
    species_groups = defaultdict(list)
    
    for accession, data in genome_data.items():
        species = data['species']
        species_groups[species].append(accession)
    
    print(f"Found {len(species_groups)} unique species")
    
    # Select best genome for each species
    selected_genomes = {}
    duplicates_removed = 0
    
    for species, accessions in species_groups.items():
        if len(accessions) == 1:
            # No duplicates
            selected_genomes[accessions[0]] = {
                **genome_data[accessions[0]],
                'reason': 'unique'
            }
        else:
            # Multiple genomes for this species
            duplicates_removed += len(accessions) - 1
            
            # Separate GCF and GCA accessions
            gcf_accessions = [acc for acc in accessions if acc.startswith('GCF')]
            gca_accessions = [acc for acc in accessions if acc.startswith('GCA')]
            
            # Choose which pool to select from
            if gcf_accessions:
                candidate_pool = gcf_accessions
                reason = f"GCF_preferred_from_{len(accessions)}"
            else:
                candidate_pool = gca_accessions
                reason = f"GCA_only_from_{len(accessions)}"
            
            # Sort by quality metrics
            candidate_pool.sort(
                key=lambda acc: get_quality_score(genome_data[acc]),
                reverse=True
            )
            
            best_accession = candidate_pool[0]
            selected_genomes[best_accession] = {
                **genome_data[best_accession],
                'reason': reason
            }
            
            print(f"Species: {species}")
            print(f"  Total genomes: {len(accessions)}")
            print(f"  GCF genomes: {len(gcf_accessions)}")
            print(f"  GCA genomes: {len(gca_accessions)}")
            print(f"  Selected: {best_accession}")
            print(f"  Quality: C={genome_data[best_accession]['complete_buscos']}, "
                  f"SC={genome_data[best_accession]['single_copy_buscos']}, "
                  f"M={genome_data[best_accession]['missing_buscos']}")
            print()
    
    # Write results
    print(f"Writing results to {output_file}")
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Accession', 'Species', 'Genus', 'Family', 'Order', 'Class', 'Phylum', 'Kingdom',
            'Complete_BUSCOs', 'Single_Copy_BUSCOs', 'Fragmented_BUSCOs', 'Missing_BUSCOs', 'Selection_Reason'
        ])
        
        for accession, data in selected_genomes.items():
            writer.writerow([
                accession,
                data['species'],
                data['genus'],
                data['family'],
                data['order'],
                data['class'],
                data['phylum'],
                data['kingdom'],
                data['complete_buscos'],
                data['single_copy_buscos'],
                data['fragmented_buscos'],
                data['missing_buscos'],
                data['reason']
            ])
    
    print(f"Summary:")
    print(f"  Original genomes: {len(genome_data)}")
    print(f"  Selected genomes: {len(selected_genomes)}")
    print(f"  Duplicates removed: {duplicates_removed}")
    print(f"  Unique species: {len(species_groups)}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Deduplicate genomes based on species and BUSCO metrics.')
    parser.add_argument('--input', required=True, help='Input CSV file with lineage and BUSCO data')
    parser.add_argument('--output', required=True, help='Output CSV file for deduplicated genomes')
    args = parser.parse_args()

    data_file = args.input
    output_file = args.output

    if not os.path.exists(data_file):
        print(f"Error: Input file not found: {data_file}")
        sys.exit(1)

    deduplicate_genomes(data_file, output_file)
    print(f"Deduplication complete! Results saved to: {output_file}")

if __name__ == "__main__":
    main()
