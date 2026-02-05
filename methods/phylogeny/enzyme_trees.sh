#!/bin/bash
# LSF job headers (customize via env or config)
LSF_JOB_NAME="${LSF_JOB_NAME:-submit_iqtree}"
LSF_OUT="${LSF_OUT:-out/submit_iqtree_%J.out}"
LSF_ERR="${LSF_ERR:-out/submit_iqtree_%J.err}"
LSF_QUEUE="${LSF_QUEUE:-hpc}"

#!/bin/bash
# Template for running IQ-TREE jobs for enzyme alignments.
# All configuration must be provided via environment variables or arguments.

# Required variables:
#   OUT_IQTREE_DIR: Output directory for IQ-TREE logs
#   ENZYME_TREES_DIR: Output directory for enzyme trees
#   TRIM_DIR: Directory containing .aln alignment files
#   IQTREE_ENV: Conda environment for IQ-TREE
#   IQTREE_THREADS: Number of threads for IQ-TREE
#   IQTREE_MODEL: Substitution model for IQ-TREE
#   MAX_CONCURRENT: Maximum concurrent jobs

if [ -z "$OUT_IQTREE_DIR" ] || [ -z "$ENZYME_TREES_DIR" ] || [ -z "$TRIM_DIR" ] || [ -z "$IQTREE_ENV" ] || [ -z "$IQTREE_THREADS" ] || [ -z "$IQTREE_MODEL" ] || [ -z "$MAX_CONCURRENT" ]; then
  echo "Error: One or more required variables are not set."
  echo "Set OUT_IQTREE_DIR, ENZYME_TREES_DIR, TRIM_DIR, IQTREE_ENV, IQTREE_THREADS, IQTREE_MODEL, MAX_CONCURRENT."
  exit 1
fi

mkdir -p "$OUT_IQTREE_DIR"
mkdir -p "$ENZYME_TREES_DIR"

count_running_jobs() {
    pgrep -f "iqtree.*${TRIM_DIR}" | wc -l
}

for aln in "$TRIM_DIR"/*.aln; do
    enzyme=$(basename "$aln" .aln)
    while [ $(count_running_jobs) -ge $MAX_CONCURRENT ]; do
        echo "Maximum concurrent jobs reached, waiting..."
        sleep 60
    done
    out_dir="$ENZYME_TREES_DIR/$enzyme"
    mkdir -p "$out_dir"
    echo "Running IQ-TREE for ${enzyme}"
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate "$IQTREE_ENV"
    iqtree -s "$aln" -T "$IQTREE_THREADS" -m "$IQTREE_MODEL" -fast -pre "${out_dir}/tree_iq_multi_LGI" &
done

wait
echo "All IQ-TREE jobs submitted."
fi

mkdir -p "$OUT_IQTREE_DIR"
mkdir -p "$ENZYME_TREES_DIR"

count_running_jobs() {
    bjobs -w | grep "iqtree_" | wc -l
}

for aln in "$TRIM_DIR"/*.aln; do
    enzyme=$(basename "$aln" .aln)
    while [ $(count_running_jobs) -ge $MAX_CONCURRENT ]; do
        echo "Maximum concurrent jobs reached, waiting..."
        sleep 60
    done
    out_dir="$ENZYME_TREES_DIR/$enzyme"
    mkdir -p "$out_dir"
    echo "Submitting IQ-TREE job for ${enzyme}"
    bsub -J "iqtree_${enzyme}" \
         -o "$OUT_IQTREE_DIR/iqtree_${enzyme}_%J.out" \
         -e "$OUT_IQTREE_DIR/iqtree_${enzyme}_%J.err" \
         -n "$IQTREE_THREADS" \
         -R "span[hosts=1] rusage[mem=$IQTREE_MEM]" \
         -q "$IQTREE_QUEUE" \
         -W "$IQTREE_WALLTIME" \
         "source \$HOME/miniconda3/etc/profile.d/conda.sh && \
          conda activate $IQTREE_ENV && \
          iqtree -s ${aln} -T $IQTREE_THREADS -m $IQTREE_MODEL -fast -pre ${out_dir}/tree_iq_multi_LGI"
done