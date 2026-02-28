#!/bin/bash
#SBATCH -p RM
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH -o outputs/slurm_%j.out

set -euo pipefail

export OMP_PROC_BIND=true
export OMP_PLACES=cores

make clean all > outputs/build.log 2>&1
bash bench_within.sh
cp outputs/within_wires_bench.csv outputs/psc_wires_bench.csv