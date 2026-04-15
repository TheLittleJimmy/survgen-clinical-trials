#!/bin/bash
set -e

# Master reproduction script for ACTG320 experiments
# Runs all hyperopt + compare steps sequentially in terminal

PROJ=/project/Stat/s1155202253/myproject/pfizer_projects/survgen-clinical-trials
SCRIPT=$PROJ/script
REPRO=$PROJ/reproduce/ACTG320
LOGDIR=$REPRO/logs

mkdir -p $LOGDIR

eval "$(conda shell.bash hook)"
conda activate hivae

GENERATORS=(0 1 2 3 4 5)
# 0=HI-VAE_weibull 1=HI-VAE_piecewise 2=Surv-GAN 3=Surv-VAE 4=HI-VAE_weibull_prior 5=HI-VAE_piecewise_prior

echo "=========================================="
echo "Step 1: Hyperopt traincontrol (6 generators)"
echo "=========================================="
for g in "${GENERATORS[@]}"; do
    echo "[$(date)] Starting hyperopt traincontrol generator=$g"
    cd $SCRIPT
    python3 ACTG320_hyperopt_traincontrol_parallel.py $g 2>&1 | tee $LOGDIR/step1_hyperopt_traincontrol_gen${g}.log
    echo "[$(date)] Finished hyperopt traincontrol generator=$g"
done

echo "=========================================="
echo "Step 2: Hyperopt trainfull (6 generators)"
echo "=========================================="
for g in "${GENERATORS[@]}"; do
    echo "[$(date)] Starting hyperopt trainfull generator=$g"
    cd $SCRIPT
    python3 ACTG320_hyperopt_trainfull_parallel.py $g 2>&1 | tee $LOGDIR/step2_hyperopt_trainfull_gen${g}.log
    echo "[$(date)] Finished hyperopt trainfull generator=$g"
done

echo "=========================================="
echo "Step 3: Hyperopt aug traincontrol (6 generators)"
echo "=========================================="
for g in "${GENERATORS[@]}"; do
    echo "[$(date)] Starting hyperopt aug traincontrol generator=$g"
    cd $SCRIPT
    python3 ACTG320_hyperopt_aug_traincontrol_parallel.py $g 2>&1 | tee $LOGDIR/step3_hyperopt_aug_traincontrol_gen${g}.log
    echo "[$(date)] Finished hyperopt aug traincontrol generator=$g"
done

echo "=========================================="
echo "Step 4: Hyperopt aug trainfull (6 generators)"
echo "=========================================="
for g in "${GENERATORS[@]}"; do
    echo "[$(date)] Starting hyperopt aug trainfull generator=$g"
    cd $SCRIPT
    python3 ACTG320_hyperopt_aug_trainfull_parallel.py $g 2>&1 | tee $LOGDIR/step4_hyperopt_aug_trainfull_gen${g}.log
    echo "[$(date)] Finished hyperopt aug trainfull generator=$g"
done

echo "=========================================="
echo "Step 5: Compare traincontrol"
echo "=========================================="
cd $SCRIPT
python3 -c "
from realdataset_compare_traincontrol_parallel import run
generators_sel = ['HI-VAE_weibull', 'HI-VAE_piecewise', 'Surv-GAN', 'Surv-VAE', 'HI-VAE_weibull_prior', 'HI-VAE_piecewise_prior']
run('ACTG320', generators_sel)
" 2>&1 | tee $LOGDIR/step5_compare_traincontrol.log

echo "=========================================="
echo "Step 6: Compare trainfull"
echo "=========================================="
cd $SCRIPT
python3 -c "
from realdataset_compare_trainfull_parallel import run
generators_sel = ['HI-VAE_weibull', 'HI-VAE_piecewise', 'Surv-GAN', 'Surv-VAE', 'HI-VAE_weibull_prior', 'HI-VAE_piecewise_prior']
run('ACTG320', generators_sel)
" 2>&1 | tee $LOGDIR/step6_compare_trainfull.log

echo "=========================================="
echo "Step 7: Compare aug traincontrol"
echo "=========================================="
cd $SCRIPT
python3 -c "
from realdataset_compare_aug_traincontrol_parallel import run
generators_sel = ['HI-VAE_weibull', 'HI-VAE_piecewise', 'Surv-GAN', 'Surv-VAE', 'HI-VAE_weibull_prior', 'HI-VAE_piecewise_prior']
run('ACTG320', generators_sel)
" 2>&1 | tee $LOGDIR/step7_compare_aug_traincontrol.log

echo "=========================================="
echo "Step 8: Compare aug trainfull"
echo "=========================================="
cd $SCRIPT
python3 -c "
from realdataset_compare_aug_trainfull_parallel import run
generators_sel = ['HI-VAE_weibull', 'HI-VAE_piecewise', 'Surv-GAN', 'Surv-VAE', 'HI-VAE_weibull_prior', 'HI-VAE_piecewise_prior']
run('ACTG320', generators_sel)
" 2>&1 | tee $LOGDIR/step8_compare_aug_trainfull.log

echo "=========================================="
echo "Step 9: Copy results to reproduce/ACTG320"
echo "=========================================="
cp -r $PROJ/dataset/ACTG320/optuna_results $REPRO/
cp -r $PROJ/dataset/ACTG320/metric_results $REPRO/

echo "=========================================="
echo "[$(date)] ALL ACTG320 EXPERIMENTS COMPLETE"
echo "Results in: $REPRO/"
echo "=========================================="
