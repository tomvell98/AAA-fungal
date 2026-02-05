
#!/bin/bash
# Template for running BUSCO in batches.
# All configuration must be provided via environment variables or arguments.

# Required variables:
#   OUT_DIR: Output directory for logs
#   BUSCO_OUT_DIR: Output directory for BUSCO results
#   START_BATCH: First batch number
#   END_BATCH: Last batch number
#   MAX_CONCURRENT: Maximum concurrent jobs
#   BUSCO_BATCH_PARENT: Parent directory containing batch folders
#   BUSCO_LINEAGE: BUSCO lineage dataset
#   BUSCO_ENV: Conda environment for BUSCO
#   BUSCO_MODE: BUSCO mode (e.g., genome)
#   BUSCO_THREADS: Number of threads
#   CONDA_SH: Path to conda.sh

if [ -z "$OUT_DIR" ] || [ -z "$BUSCO_OUT_DIR" ] || [ -z "$START_BATCH" ] || [ -z "$END_BATCH" ] || [ -z "$MAX_CONCURRENT" ] || [ -z "$BUSCO_BATCH_PARENT" ] || [ -z "$BUSCO_LINEAGE" ] || [ -z "$BUSCO_ENV" ] || [ -z "$BUSCO_MODE" ] || [ -z "$BUSCO_THREADS" ] || [ -z "$CONDA_SH" ]; then
  echo "Error: One or more required variables are not set."
  echo "Set OUT_DIR, BUSCO_OUT_DIR, START_BATCH, END_BATCH, MAX_CONCURRENT, BUSCO_BATCH_PARENT, BUSCO_LINEAGE, BUSCO_ENV, BUSCO_MODE, BUSCO_THREADS, CONDA_SH."
  exit 1
fi

mkdir -p "$OUT_DIR"
mkdir -p "$BUSCO_OUT_DIR"

count_running_jobs() {
    pgrep -f "busco.*${BUSCO_BATCH_PARENT}" | wc -l
}

for batch_num in $(seq $START_BATCH $END_BATCH); do
    batch_dir="${BUSCO_BATCH_PARENT}/batch_l_${batch_num}"
    out_dir="${BUSCO_OUT_DIR}/busco_batch_l_${batch_num}"
    if [ -d "$batch_dir" ]; then
        while [ $(count_running_jobs) -ge $MAX_CONCURRENT ]; do
            echo "Maximum concurrent jobs reached, waiting..."
            sleep 60
        done
        echo "Running BUSCO for batch ${batch_num}"
        (
            source "$CONDA_SH"
            conda activate "$BUSCO_ENV"
            busco -i "$batch_dir" \
                  -m "$BUSCO_MODE" \
                  -l "$BUSCO_LINEAGE" \
                  -f \
                  -q \
                  -o "$out_dir" \
                  -c "$BUSCO_THREADS" \
                  --metaeuk
        ) &
    else
        echo "Batch directory ${batch_num} not found"
    fi
done

wait
echo "All BUSCO batch jobs completed."
