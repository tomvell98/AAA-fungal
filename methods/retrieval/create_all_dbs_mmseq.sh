
#!/bin/bash
# Template for creating MMseqs databases for all protein.faa files.
# All configuration must be provided via environment variables or arguments.

# Required variables:
#   CONDA_SH: Path to conda.sh
#   MMSEQS_ENV: Conda environment for MMseqs
#   PARENT_DIR: Directory to search for protein.faa files

if [ -z "$CONDA_SH" ] || [ -z "$MMSEQS_ENV" ] || [ -z "$PARENT_DIR" ]; then
    echo "Error: One or more required variables are not set."
    echo "Set CONDA_SH, MMSEQS_ENV, PARENT_DIR."
    exit 1
fi

source "$CONDA_SH"
conda activate "$MMSEQS_ENV"

find "$PARENT_DIR" -name "protein.faa" | while read -r protein_file; do
        dir_path=$(dirname "$protein_file")
        echo "Creating database for: $protein_file"
        mmseqs createdb "$protein_file" "$dir_path/db"
        mmseqs createindex "$dir_path/db" "$dir_path/tmp"
done

echo "All databases created successfully."
