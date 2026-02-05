
#!/bin/bash
# Template for running trimAl on alignments.
# All configuration must be provided via environment variables or arguments.

# Required variables:
#   MAX_JOBS: Maximum concurrent jobs
#   SOURCE_DIR: Directory with input .aln files
#   OUTPUT_DIR: Output directory for trimmed alignments

if [ -z "$MAX_JOBS" ] || [ -z "$SOURCE_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "Error: One or more required variables are not set."
  echo "Set MAX_JOBS, SOURCE_DIR, OUTPUT_DIR."
  exit 1
fi

echo "Running trimAl with $MAX_JOBS concurrent jobs"
echo "Source directory: $SOURCE_DIR"
echo "Output directory: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

count_running_jobs() {
    ps aux | grep "trimal.*${SOURCE_DIR}" | grep -v grep | wc -l
}

for aln_file in "${SOURCE_DIR}"/*.aln; do
    if [ ! -f "$aln_file" ]; then
        echo "No .aln files found in ${SOURCE_DIR}"
        exit 1
    fi
    filename=$(basename "$aln_file" .aln)
    while [ $(count_running_jobs) -ge $MAX_JOBS ]; do
        echo "Maximum concurrent jobs ($MAX_JOBS) reached, waiting..."
        sleep 5
    done
    echo "Processing trimAl for ${filename}..."
    (
        trimal -in "${aln_file}" -out "${OUTPUT_DIR}/${filename}.aln" -gappyout 2>/dev/null
        echo "Completed: ${filename}"
    ) &
done

echo "Waiting for all trimAl jobs to complete..."
wait
echo "All trimAl trimming completed!"
