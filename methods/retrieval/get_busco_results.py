import os
import json
import pandas as pd
from tqdm import tqdm

def get_busco_results(base_dir):
    results_list = []
    
    # Only get immediate subdirectories
    for entry in tqdm(os.scandir(base_dir)):
        if entry.is_dir():
            dir_path = entry.path
            dir_name = entry.name
            
            # Look for json files in this directory
            json_files = [f for f in os.listdir(dir_path) if f.endswith('.json')]
            if len(json_files) == 1:
                json_file_path = os.path.join(dir_path, json_files[0])
                with open(json_file_path, 'r') as json_file:
                    data = json.load(json_file)
                    results = data.get('results', {})
                    
                    results_dict = {
                        'organism': dir_name,
                        'complete_buscos': results.get('Complete BUSCOs', 0),
                        'single_copy_buscos': results.get('Single copy BUSCOs', 0),
                        'fragmented_buscos': results.get('Fragmented BUSCOs', 0),
                        'missing_buscos': results.get('Missing BUSCOs', 0)
                    }
                    results_list.append(results_dict)
            else:
                print(f"WARNING: {dir_path} contains {len(json_files)} json files")
    
    return pd.DataFrame(results_list)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Extract BUSCO results from subdirectories.')
    parser.add_argument('--base_directory', required=True, help='Base directory containing BUSCO result subfolders')
    parser.add_argument('--output_csv', required=True, help='Output CSV file path')
    args = parser.parse_args()
    busco_results_df = get_busco_results(args.base_directory)
    busco_results_df.to_csv(args.output_csv, index=False)