
#!/bin/bash
# Template for downloading protein data for a set of accessions.
# All configuration must be provided via environment variables or arguments.

# Required variables:
#   PROTEIN_DIR: Directory for protein data
#   GENOMES_CSV: Path to deduplicated genomes CSV

if [ -z "$PROTEIN_DIR" ] || [ -z "$GENOMES_CSV" ]; then
  echo "Error: One or more required variables are not set."
  echo "Set PROTEIN_DIR and GENOMES_CSV."
  exit 1
fi

mkdir -p "$PROTEIN_DIR"

ACCESSIONS=$(tail -n +2 "$GENOMES_CSV" | cut -c 1-15)

mapfile -t EXISTING_FOLDERS < <(ls -d ${PROTEIN_DIR}/*/ 2>/dev/null | xargs -n 1 basename)
echo "Found ${#EXISTING_FOLDERS[@]} existing folders"

MISSING_ACCESSIONS=()
for ACCESSION in $ACCESSIONS; do
    FOUND=0
    for FOLDER in "${EXISTING_FOLDERS[@]}"; do
        if [ "$ACCESSION" = "$FOLDER" ]; then
            FOUND=1
            break
        fi
    done
    if [ $FOUND -eq 0 ]; then
        MISSING_ACCESSIONS+=("$ACCESSION")
    fi
done

echo "Total accessions to process: $(echo "$ACCESSIONS" | wc -w)"
echo "Number of missing accessions: ${#MISSING_ACCESSIONS[@]}"

for ACCESSION in "${MISSING_ACCESSIONS[@]}"; do
    echo "Downloading proteins for: $ACCESSION"
    if ! datasets download genome accession "$ACCESSION" \
        --include protein \
        --filename "${PROTEIN_DIR}/${ACCESSION}_proteins.zip"; then
        echo "Failed to download proteins for: $ACCESSION"
    fi
done





