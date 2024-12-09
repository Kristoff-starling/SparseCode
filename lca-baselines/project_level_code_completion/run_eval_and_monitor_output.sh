#!/bin/bash

output=$(sbatch run_eval.sbatch 2>&1)

if [[ $output =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
    job_id=${BASH_REMATCH[1]}
    echo "Job ID: $job_id"
else
    echo "Failed to submit job"
    exit 1
fi

log_file="job.${job_id}.out"
echo "Monitoring log file: $log_file"

while [ ! -f $log_file ]; do
    sleep 0.5
done

tail -f $log_file