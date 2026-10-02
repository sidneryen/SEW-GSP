#!/bin/bash
# Run all pending (dataset, method) jobs, 2 at a time, each in its own process.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 run_main.py --list | xargs -P ${1:-2} -L 1 sh -c 'python3 run_main.py --job "$0" "$1" || echo "FAILED $0 $1"'
