#!/bin/bash

model_filename=outputs/models/mimic_transitions_hl_1_hu_200_lr_0.001_bs_256_lipschitzloc_1.0_lipschitzscale_0.02_prior_laplace_maxepochs_100.pt
temp_directory=outputs/temp_outputs/
processed_data_directory=data/processed/
experiment_directory=outputs/experiments/bench_
device=cpu
seed=42
anchor_method=montecarlo-proportional
anchor_samples=200
algos=(random topk greedy astar)
ks=(0 1 2 3 4 5)

horizon=12
num_of_episodes=1
ids_file=${processed_data_directory}horizon_${horizon}.txt

counter=0
# open the file with the ids and read the first num_of_episodes lines
for algo in "${algos[@]}"; do
    for k in "${ks[@]}"; do
        for pid in $(head -n $num_of_episodes $ids_file); do
            python -m src.mimic_mdp --model_filename=$model_filename --temp_directory=$temp_directory --processed_data_directory=$processed_data_directory --experiment_directory=$experiment_directory --device=$device --pid=$pid --k=$k --seed=$seed --anchor_method=$anchor_method --anchor_samples=$anchor_samples --algo=$algo #&
            counter=$((counter+1))
            # if device=cuda:0 and counter equals to 4, set the device to cuda:1
            if [ $device = "cuda:0" ] && [ $counter -eq 4 ]; then
                device="cuda:1"
            fi
            # if device=cuda:1 and counter equals to 8, set the device to cuda:0 and wait
            if [ $device = "cuda:1" ] && [ $counter -eq 8 ]; then
                device="cuda:0"
                counter=0
                wait
            fi
        done
    done
done
