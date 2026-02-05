
#!/bin/bash
# Template for running IQ-TREE on a supermatrix alignment.
# All configuration must be provided via environment variables or arguments.

# Required variables:
#   CONDA_SH: Path to conda.sh
#   IQTREE_ENV: Conda environment for IQ-TREE
#   IQTREE_INPUT: Input alignment file
#   IQTREE_THREADS: Number of threads
#   IQTREE_MEM: Memory for IQ-TREE
#   IQTREE_MODEL: Substitution model
#   IQTREE_PRE: Output prefix

if [ -z "$CONDA_SH" ] || [ -z "$IQTREE_ENV" ] || [ -z "$IQTREE_INPUT" ] || [ -z "$IQTREE_THREADS" ] || [ -z "$IQTREE_MEM" ] || [ -z "$IQTREE_MODEL" ] || [ -z "$IQTREE_PRE" ]; then
	echo "Error: One or more required variables are not set."
	echo "Set CONDA_SH, IQTREE_ENV, IQTREE_INPUT, IQTREE_THREADS, IQTREE_MEM, IQTREE_MODEL, IQTREE_PRE."
	exit 1
fi

source "$CONDA_SH"
conda activate "$IQTREE_ENV"

iqtree -s "$IQTREE_INPUT" -T "$IQTREE_THREADS" -mem "$IQTREE_MEM" -m "$IQTREE_MODEL" -fast -pre "$IQTREE_PRE"