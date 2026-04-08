#!/bin/bash
#SBATCH --job-name=actg320_step1_hyperopt_traincontrol
#SBATCH --output=/project/Stat/s1155202253/myproject/survgen-clinical-trials/reproduce/ACTG320/logs/step1_hyperopt_traincontrol_%A_%a.out
#SBATCH --error=/project/Stat/s1155202253/myproject/survgen-clinical-trials/reproduce/ACTG320/logs/step1_hyperopt_traincontrol_%A_%a.err
#SBATCH --array=0-5
#SBATCH -w chpc-gpu029
#SBATCH -p statgp1
#SBATCH -c 10
#SBATCH --mem=108GB
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:59

echo "### Step 1: Hyperopt traincontrol, generator=$SLURM_ARRAY_TASK_ID ###"
eval "$(conda shell.bash hook)"
conda activate hivae

cd /project/Stat/s1155202253/myproject/survgen-clinical-trials/script
python3 ACTG320_hyperopt_traincontrol_parallel.py $SLURM_ARRAY_TASK_ID

echo "### Step 1, generator=$SLURM_ARRAY_TASK_ID done ###"
