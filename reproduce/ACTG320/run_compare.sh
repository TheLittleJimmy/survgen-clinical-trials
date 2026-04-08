#!/bin/bash
set -e

PROJ=/project/Stat/s1155202253/myproject/survgen-clinical-trials
SCRIPT=$PROJ/script
REPRO=$PROJ/reproduce/ACTG320
LOGDIR=$REPRO/logs

mkdir -p $LOGDIR

eval "$(conda shell.bash hook)"
conda activate hivae

GENERATORS="['HI-VAE_weibull', 'HI-VAE_piecewise', 'Surv-GAN', 'Surv-VAE', 'HI-VAE_weibull_prior', 'HI-VAE_piecewise_prior']"

echo "=========================================="
echo "Step 1: Compare traincontrol"
echo "=========================================="
cd $SCRIPT
python3 -u -c "
from realdataset_compare_traincontrol_parallel import run
run('ACTG320', $GENERATORS)
" 2>&1 | tee $LOGDIR/compare_traincontrol.log
echo "[$(date)] Done"

echo "=========================================="
echo "Step 2: Compare trainfull"
echo "=========================================="
cd $SCRIPT
python3 -u -c "
from realdataset_compare_trainfull_parallel import run
run('ACTG320', $GENERATORS)
" 2>&1 | tee $LOGDIR/compare_trainfull.log
echo "[$(date)] Done"

echo "=========================================="
echo "Step 3: Compare aug traincontrol"
echo "=========================================="
cd $SCRIPT
python3 -u -c "
from realdataset_compare_aug_traincontrol_parallel import run
run('ACTG320', $GENERATORS)
" 2>&1 | tee $LOGDIR/compare_aug_traincontrol.log
echo "[$(date)] Done"

echo "=========================================="
echo "Step 4: Compare aug trainfull"
echo "=========================================="
cd $SCRIPT
python3 -u -c "
from realdataset_compare_aug_trainfull_parallel import run
run('ACTG320', $GENERATORS)
" 2>&1 | tee $LOGDIR/compare_aug_trainfull.log
echo "[$(date)] Done"

echo "=========================================="
echo "Step 5: Copy all results to reproduce/ACTG320"
echo "=========================================="
cp -r $PROJ/dataset/ACTG320/optuna_results $REPRO/
cp -r $PROJ/dataset/ACTG320/metric_results $REPRO/

echo "=========================================="
echo "[$(date)] ALL REPRODUCTION COMPLETE"
echo "Results in: $REPRO/"
ls -la $REPRO/metric_results/
echo "=========================================="
