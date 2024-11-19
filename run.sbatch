#!/bin/bash

#SBATCH -J notebook           # Job name
#SBATCH -o job.%j.out         # Name of stdout output file (%j expands to jobId)
#SBATCH -N 1                  # Total number of nodes requested
#SBATCH -t 12:00:00           # Run time (hh:mm:ss) - 1.5 hours
#SBATCH -p mi2104x            # Desired partition

module load pytorch
python3 model.py