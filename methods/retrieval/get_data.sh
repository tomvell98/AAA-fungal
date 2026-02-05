
#!/bin/bash
# Template for downloading fungal genomes using NCBI datasets.
# All configuration must be provided via environment variables or arguments.

# Required variable:
#   FUNGI_ZIP: Output zip file for downloaded genomes

if [ -z "$FUNGI_ZIP" ]; then
	echo "Error: FUNGI_ZIP variable is not set."
	echo "Set FUNGI_ZIP to the output zip file name."
	exit 1
fi

datasets download genome taxon fungi --include genome --annotated --filename "$FUNGI_ZIP"




