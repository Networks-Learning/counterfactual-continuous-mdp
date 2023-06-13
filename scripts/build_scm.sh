#!/bin/bash

# set the name of the prediction task and the dataset file
prediction_task="mimic_transitions"
data_filename="data/processed/dataset_transitions.csv"
model_directory="outputs/models/"

# set the epoch strategy
max_epochs=100

# set the hyperparameter values
hidden_layers=1
hidden_units=200
lipschitz_loc=1.0
lipschitz_scale=0.1
prior_type='multigaussian'
batch_size=256
learning_rate=0.001
cuda=0

# train final model
python -m src.cv_scm --prediction_task=$prediction_task --data_filename=$data_filename --hidden_layers=$hidden_layers --hidden_units=$hidden_units --learning_rate=$learning_rate --batch_size=$batch_size --max_epochs=$max_epochs --lipschitz_loc=$lipschitz_loc --lipschitz_scale=$lipschitz_scale --prior_type=$prior_type --cuda=$cuda --final --model_directory=$model_directory #&
