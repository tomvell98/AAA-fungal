
#!/bin/bash
# Template for running MAFFT alignments for enzyme sequences.
# All configuration must be provided via environment variables or arguments.

# Required variables:
#   MAX_JOBS: Maximum concurrent jobs
#   THREADS_PER_JOB: Threads per MAFFT job
#   SEQUENCE_DIR: Directory with input .fasta files
#   OUTPUT_DIR: Output directory for alignments

if [ -z "$MAX_JOBS" ] || [ -z "$THREADS_PER_JOB" ] || [ -z "$SEQUENCE_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "Error: One or more required variables are not set."
  echo "Set MAX_JOBS, THREADS_PER_JOB, SEQUENCE_DIR, OUTPUT_DIR."
  exit 1
fi

echo "Running MAFFT alignment for enzymes with $MAX_JOBS concurrent jobs, $THREADS_PER_JOB threads each"
echo "Sequence directory: $SEQUENCE_DIR"
echo "Output directory: $OUTPUT_DIR"

mkdir -p "${OUTPUT_DIR}/alignments"

count_running_jobs() {
    ps aux | grep "mafft.*${SEQUENCE_DIR}" | grep -v grep | wc -l
}

for seq_file in "${SEQUENCE_DIR}"/*.fasta; do
    if [ ! -f "$seq_file" ]; then
        echo "No .fasta files found in ${SEQUENCE_DIR}"
        exit 1
    fi
    filename=$(basename "$seq_file" .fasta)
    while [ $(count_running_jobs) -ge $MAX_JOBS ]; do
        echo "Maximum concurrent jobs ($MAX_JOBS) reached, waiting..."
        sleep 5
    done
    echo "Processing MAFFT alignment for enzyme ${filename}..."
    (
        mafft --retree 2 --maxiterate 0 --thread ${THREADS_PER_JOB} "${seq_file}" > "${OUTPUT_DIR}/alignments/${filename}.aln" 2>/dev/null
        echo "Completed: ${filename}"
    ) &
done

echo "Waiting for all MAFFT enzyme alignment jobs to complete..."
wait
echo "All enzyme MAFFT alignments completed!"
