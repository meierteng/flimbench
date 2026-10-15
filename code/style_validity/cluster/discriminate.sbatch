#!/bin/bash
#SBATCH --job-name=sb-disc
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/export/fs01/cli229/style_bench/logs/%x-%j.out
B=/export/fs01/cli229/style_bench
cd $B/code
$B/venv/bin/python -W ignore discriminate.py $B/extract $B/discriminate --perturb $B/perturb/sensitivity.csv
