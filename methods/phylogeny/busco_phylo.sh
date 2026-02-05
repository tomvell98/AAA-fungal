
#!/bin/bash
# Template for running BUSCO phylogenomics analysis.
# All configuration must be provided via environment variables or arguments.

# Required variables:
#   CONDA_SH: Path to conda.sh
#   BUSCO_PHYLO_ENV: Conda environment for BUSCO phylogenomics
#   BUSCO_PHYLO_INPUT: Input directory for BUSCO runs
#   BUSCO_PHYLO_OUTPUT: Output directory for results
#   BUSCO_PHYLO_THREADS: Number of threads
#   BUSCO_PHYLO_PSC: Percent single-copy threshold

if [ -z "$CONDA_SH" ] || [ -z "$BUSCO_PHYLO_ENV" ] || [ -z "$BUSCO_PHYLO_INPUT" ] || [ -z "$BUSCO_PHYLO_OUTPUT" ] || [ -z "$BUSCO_PHYLO_THREADS" ] || [ -z "$BUSCO_PHYLO_PSC" ]; then
	echo "Error: One or more required variables are not set."
	echo "Set CONDA_SH, BUSCO_PHYLO_ENV, BUSCO_PHYLO_INPUT, BUSCO_PHYLO_OUTPUT, BUSCO_PHYLO_THREADS, BUSCO_PHYLO_PSC."
	exit 1
fi

source "$CONDA_SH"
conda activate "$BUSCO_PHYLO_ENV"

BUSCO_phylogenomics.py -i "$BUSCO_PHYLO_INPUT" -o "$BUSCO_PHYLO_OUTPUT" -t "$BUSCO_PHYLO_THREADS" --supermatrix_only -psc "$BUSCO_PHYLO_PSC"

