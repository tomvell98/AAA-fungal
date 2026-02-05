

#!/bin/bash
# Template for running AlphaFold3 on multiple input directories.
# All configuration must be provided via environment variables or arguments.

# Required variables:
#   INPUT_PARENT: Parent directory containing input folders
#   INPUT_GLOB: Glob pattern for input directories
#   MODELS_DIR: Directory with AlphaFold3 models
#   DB_DIR: Directory with AlphaFold3 databases
#   DOCKER_IMAGE: Docker image for AlphaFold3
#   CPUS: Number of CPUs to allocate
#   GPUS: GPU specification for Docker
#   NUM_DIFFUSION_SAMPLES: Number of diffusion samples
#   NUM_RECYCLES: Number of recycles
#   PY_SCRIPT: Python script to run inside the container

if [ -z "$INPUT_PARENT" ] || [ -z "$INPUT_GLOB" ] || [ -z "$MODELS_DIR" ] || [ -z "$DB_DIR" ] || [ -z "$DOCKER_IMAGE" ] || [ -z "$CPUS" ] || [ -z "$GPUS" ] || [ -z "$NUM_DIFFUSION_SAMPLES" ] || [ -z "$NUM_RECYCLES" ] || [ -z "$PY_SCRIPT" ]; then
  echo "Error: One or more required variables are not set."
  echo "Set INPUT_PARENT, INPUT_GLOB, MODELS_DIR, DB_DIR, DOCKER_IMAGE, CPUS, GPUS, NUM_DIFFUSION_SAMPLES, NUM_RECYCLES, PY_SCRIPT."
  exit 1
fi

for fastas_dir in "$INPUT_PARENT"/$INPUT_GLOB; do
    if [ -d "$fastas_dir" ]; then
        output_dir="$fastas_dir/output"
        mkdir -p "$output_dir"
        echo "Running AlphaFold3 for $fastas_dir -> $output_dir"
        docker run -it \
            --user $(id -u):$(id -g) \
            --volume "$fastas_dir":/input:ro \
            --volume "$output_dir":/output \
            --volume "$MODELS_DIR":/models:ro \
            --volume "$DB_DIR":/database:ro \
            --cpus="$CPUS" \
            --gpus "$GPUS" \
            "$DOCKER_IMAGE" \
            python "$PY_SCRIPT" \
            --input_dir=/input \
            --model_dir=/models \
            --db_dir=/database \
            --num_diffusion_samples="$NUM_DIFFUSION_SAMPLES" \
            --num_recycles="$NUM_RECYCLES" \
            --output_dir=/output
    fi
done
