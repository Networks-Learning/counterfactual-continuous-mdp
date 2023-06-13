#!/bin/bash

# This script is used to submit an array of jobs to a slurm scheduler
# The various configurations of the experiments are set here

# Replace PLACEHOLDER with the full path of your working directory 
processed_data_directory={PLACEHOLDER}/data/processed/
single_experiment_file={PLACEHOLDER}/scripts/single_experiment_slurm.sh
# Specify the required memory, time and partition for the jobs. These values depend on the size and configuration of each instance
# and they should be figured out manually with a few test runs. For the computing resources reported in the paper, the below preset
# values are sufficient for the majority of the experiments.
job_mem=8G
job_time=0-00:59
job_partition=a100

# Adjust the following lines to set the experiment configuration

###################
name=grande # specifies the name of the experiment which is subsequently used to generate the respective figure
# Figure 2(a): lipschitz
# Figure 2(b): anchors
# Figures 2(c), 3(a): kappas
# Figures 3(b,c): grande
# Figure 4: size
###################

###################
horizon=18  # specifies that the experiment will be run only for patients with a horizon equal to the given value
# Figures 3(b,c): run with all values from 10 to 20
# All remaining figures: run with horizon=12 
###################

###################
lipschitz=1.0   # specifies that the experiment will be run using an SCM whose location network Lipschitz constant is equal to the given value
# Figure 2(a): run with values in {0.8, 0.9, 1.0, 1.1, 1.2, 1.3}
# All remaining figures: run with lipschitz=1.0
###################

###################
k=3 # specifies that the number of action changes in the counterfactual episode will be at most equal to the given value
# Figure 2(c): run with all values from 0 to 6
# All remaining figures: run with k=3
###################

###################
anchor_samples=2000 # specifies that the anchor set will be generated using the given number of monte carlo samples
exact_anchors=0     # if exact_anchors=1, the above number (anchor_samples) sets the desired size of the anchor set instead of the number of monte carlo samples
anchor_method=montecarlo-proportional   # specifies the anchor set selection strategy {montecarlo-proportional, montecarlo-uniform, facility-location}
# Figure 2(b): run with anchor_samples in {500, 1000, 1500, 2000, 2500, 3000}, exact_anchors=0 and anchor_method=montecarlo-proportional
# Figure 4: run with anchor_samples in {10000, 15000, 20000, 25000, 30000}, exact_anchors=1 and anchor_method in {montecarlo-proportional, montecarlo-uniform, facility-location}
# All remaining figures: run with anchor_samples=2000, exact_anchors=0 and anchor_method=montecarlo-proportional
###################

algo=astar  # specifies the algorithm used to find the counterfactually optimal action sequence (keep as is)

ids_file=${processed_data_directory}horizon_${horizon}.txt

# Find the patient IDs (PIDs) that have a horizon as specified above
pid_list=""
while read pid; do
    pid_list+="$pid,"
done < "$ids_file"
pid_list=${pid_list%,}

# Uncomment these lines to run the experiments for a subset of 200 patient IDs
batch_size=10
n_batches=20

# Uncomment the following 2 lines to run the experiments for all patient IDs
# batch_size=4
# n_batches=$((($(echo "$pid_list" | tr ',' '\n' | wc -l) - 1) / $batch_size + 1))

for ((i = 0; i < $n_batches; i++)); do
    # Extract a batch of PIDs from the PID list
    start_idx=$(($i * $batch_size))
    end_idx=$((($i + 1) * $batch_size - 1))
    batch_pids=$(echo "$pid_list" | cut -d',' -f$(($start_idx+1))-$((end_idx+1)))
    # Submit a job array for the batch
    sbatch -a $batch_pids -c 1 --gres gpu:1 -o slurm_name_${name}_horizon_${horizon}_lipschitz_${lipschitz}_k_${k}_anchors_${anchor_samples}_algo_${algo}_batch_${i}.out -e slurm_name_${name}_horizon_${horizon}_lipschitz_${lipschitz}_k_${k}_anchors_${anchor_samples}_algo_${algo}_batch_${i}.err --mem=$job_mem --time=$job_time --partition=$job_partition $single_experiment_file $name $lipschitz $k $anchor_samples $algo $anchor_method $exact_anchors
done