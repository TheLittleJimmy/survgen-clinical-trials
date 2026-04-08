#!/bin/bash
#SBATCH --job-name=actg320_step0_preprocess
#SBATCH --output=/project/Stat/s1155202253/myproject/survgen-clinical-trials/reproduce/ACTG320/logs/step0_preprocess_%j.out
#SBATCH --error=/project/Stat/s1155202253/myproject/survgen-clinical-trials/reproduce/ACTG320/logs/step0_preprocess_%j.err
#SBATCH -w chpc-gpu029
#SBATCH -p statgp1
#SBATCH -c 10
#SBATCH --mem=108GB
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:59

echo "### Step 0: Preprocessing ACTG320 ###"
eval "$(conda shell.bash hook)"
conda activate hivae

cd /project/Stat/s1155202253/myproject/survgen-clinical-trials/script
python3 realdataset_preprocessing_trainfull.py

echo "### Step 0 done ###"
