#!/bin/bash
#SBATCH --account notchpeak-gpu
#SBATCH --partition notchpeak-gpu
#SBATCH --nodes=1
#SBATCH --time=5:00:00
#SBATCH --mem=8GB
#SBATCH --gres=gpu:1
#SBATCH --mail-user=u1471339@umail.utah.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -o assignment_1-%j
#SBATCH --export=ALL

cd mp1_release

source ~/miniconda3/etc/profile.d/conda.sh
conda activate condaenv

OUT_DIR=/scratch/general/vast/u1471339/cs6957/assignment1/models

mkdir -p ${OUT_DIR}

python evaluation.py --output_dir ${OUT_DIR}



