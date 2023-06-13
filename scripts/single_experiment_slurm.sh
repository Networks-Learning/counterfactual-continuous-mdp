#!/bin/bash

# Replace PLACEHOLDER with your working directory 
source {PLACEHOLDER}/env/bin/activate

temp_directory={PLACEHOLDER}/outputs/temp_outputs/
processed_data_directory={PLACEHOLDER}/data/processed/
device=cuda
seed=42

# name of the experiment
name=$1
experiment_directory={PLACEHOLDER}/outputs/experiments/${name}_

# lipschitz value for the location network
lipschitz=$2
model_filename={PLACEHOLDER}/outputs/models/mimic_transitions_hl_1_hu_200_lr_0.001_bs_256_lipschitzloc_${lipschitz}_lipschitzscale_0.1_prior_multigaussian_maxepochs_100.pt

# k value for actions to change
k=$3

# number of montecarlo samples for the anchor set
anchor_samples=$4

# algorithm to execute
algo=$5

# anchor set generation strategy
anchor_method=$6
exact_anchors=$7

pid=${SLURM_ARRAY_TASK_ID}
echo $name $lipschitz $k $anchor_samples $pid

# if exact_anchors is set to 0 or 1 set the flag --exact to generate anchor sets based on their exact size instead of the number of monte carlo samples
if [ $exact_anchors -eq 0 ]
then
    python -m src.mimic_mdp --model_filename=$model_filename --temp_directory=$temp_directory --processed_data_directory=$processed_data_directory --experiment_directory=$experiment_directory --device=$device --pid=$pid --k=$k --seed=$seed --anchor_method=$anchor_method --anchor_samples=$anchor_samples --algo=$algo --nologging #&    
else
    python -m src.mimic_mdp --model_filename=$model_filename --temp_directory=$temp_directory --processed_data_directory=$processed_data_directory --experiment_directory=$experiment_directory --device=$device --pid=$pid --k=$k --seed=$seed --anchor_method=$anchor_method --anchor_samples=$anchor_samples --algo=$algo --nologging --exact #&
fi

deactivate
