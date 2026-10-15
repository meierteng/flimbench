#!/bin/bash
#SBATCH --job-name=sb-perturb
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/export/fs01/cli229/style_bench/logs/%x-%j.out
B=/export/fs01/cli229/style_bench
cd $B/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
$B/venv/bin/python -W ignore perturb.py $B/clip_list_6dir.csv $B/data/clips $B/perturb --clips 18 --workers 8
