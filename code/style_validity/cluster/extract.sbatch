#!/bin/bash
#SBATCH --job-name=sb-extract
#SBATCH --partition=cpu
#SBATCH --array=0-7
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=06:00:00
#SBATCH --output=/export/fs01/cli229/style_bench/logs/%x-%A_%a.out
# Phase 1 extraction, 8 shards in parallel (resumable: finished clips are skipped).
B=/export/fs01/cli229/style_bench
cd $B/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
$B/venv/bin/python -W ignore extract.py $B/clip_list_6dir.csv $B/data/clips $B/extract --workers 8 --shard $SLURM_ARRAY_TASK_ID --nshards 8
