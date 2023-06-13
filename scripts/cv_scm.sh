#!/bin/bash

# set the name of the prediction task and the dataset file
prediction_task="mimic_transitions"
data_filename="data/processed/dataset_transitions.csv"

# set the epoch strategy
max_epochs=100

# set the hyperparameter values
hidden_layers=1
hidden_units=200
loc_lipschitz_seq=(0.01 0.02 0.05 0.1 0.2 0.5 1.0 1.1 1.2 1.5)
scale_lipschitz_seq=(0.01 0.02 0.05 0.1 0.2 0.5 1.0 1.1 1.2 1.5)
priors=('laplace' 'gaussian' 'multigaussian')
batch_size=256
learning_rate=0.001
cuda=0
counter=0
# search over hyperparameter space
for prior_type in "${priors[@]}"; do
    for lipschitz_loc in "${loc_lipschitz_seq[@]}"; do
        for lipschitz_scale in "${scale_lipschitz_seq[@]}"; do
            python -m src.cv_scm --prediction_task=$prediction_task --data_filename=$data_filename --temp_directory=outputs/cv_reports/ --hidden_layers=$hidden_layers --hidden_units=$hidden_units --learning_rate=$learning_rate --batch_size=$batch_size --max_epochs=$max_epochs --lipschitz_loc=$lipschitz_loc --lipschitz_scale=$lipschitz_scale --prior_type=$prior_type --cuda=$cuda #&
            counter=$((counter+1))
            # if counter equals to 5, switch to cuda=1
            if [ $counter -eq 5 ]; then
                cuda=1
            fi
            # if counter equals to 10, set cuda to 0 and wait
            if [ $counter -eq 10 ]; then
                cuda=0
                counter=0
                wait
            fi
        done
    done
done

# train models without lipschitz constraint
for prior_type in "${priors[@]}"; do
    python -m src.cv_scm --prediction_task=$prediction_task --data_filename=$data_filename --temp_directory=outputs/cv_reports/ --hidden_layers=$hidden_layers --hidden_units=$hidden_units --learning_rate=$learning_rate --batch_size=$batch_size --max_epochs=$max_epochs --prior_type=$prior_type #&
done