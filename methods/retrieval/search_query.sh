
#!/bin/bash
# Template for running MMseqs searches against protein databases.
# All configuration must be provided via environment variables or arguments.

# Required variables:
#   CONDA_SH: Path to conda.sh
#   MMSEQS_ENV: Conda environment for MMseqs
#   QUERY_FASTA: Query FASTA file
#   PARENT_DIR: Directory to search for protein.faa files
#   N_THREADS: Number of parallel jobs

if [ -z "$CONDA_SH" ] || [ -z "$MMSEQS_ENV" ] || [ -z "$QUERY_FASTA" ] || [ -z "$PARENT_DIR" ] || [ -z "$N_THREADS" ]; then
  echo "Error: One or more required variables are not set."
  echo "Set CONDA_SH, MMSEQS_ENV, QUERY_FASTA, PARENT_DIR, N_THREADS."
  exit 1
fi

source "$CONDA_SH"
conda activate "$MMSEQS_ENV"

process_file() {
    protein_file="$1"
    dir_path=$(dirname "$protein_file")
    base_name=$(basename "$dir_path")
    if [ -f "${dir_path}/results.tsv" ]; then
        return 0
    fi
    mkdir -p "${dir_path}/mmseqs_results"
    echo "Searching against: $protein_file"
    mmseqs easy-search \
        "$QUERY_FASTA" \
        "$protein_file" \
        "${dir_path}/results.tsv" \
        "${dir_path}/tmp" \
        --format-output "query,target,evalue,tseq" \
        -s 7.5 \
        -e 1e-5 \
        --threads 1 
    rm -rf "${dir_path}/tmp"
}

export -f process_file

find "$PARENT_DIR" -name "protein.faa" -print0 | xargs -0 -I {} -P "$N_THREADS" bash -c 'process_file "{}"'
