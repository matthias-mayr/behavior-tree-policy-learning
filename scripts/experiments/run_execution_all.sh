#!/bin/bash
cwd=$(pwd)
# Exit on ctrl-c
trap "exit" INT
for i in {1..15}
do
    ./scripts/experiments/move_to_experiment_start.py peg$i
    ./scripts/experiments/run_execution.sh peg /tmp/params.csv
done
