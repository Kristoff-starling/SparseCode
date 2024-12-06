#!/bin/bash

#SBATCH -J LCA-eval           # Job name
#SBATCH -o job.%j.out         # Name of stdout output file (%j expands to jobId)
#SBATCH -N 1                  # Total number of nodes requested
#SBATCH -t 12:00:00           # Run time (hh:mm:ss) - 12 hours
#SBATCH -p mi2104x            # Desired partition

module load pytorch
poetry run python -m eval.eval_pipeline
