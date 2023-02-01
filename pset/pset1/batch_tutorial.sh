#!/bin/bash

#SBATCH --job-name=deep_learn
#SBATCH --output=gpu_job.txt
#SBATCH --time=1:00
#SBATCH --mem-per-cpu=2MB
#SBATCH --mail-type=ALL
#SBATCH -o ./Report/output.%a.out # STDOUT

module load miniconda

source activate deep_learning

python message_decode_tutorial.py