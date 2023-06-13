#!/bin/bash

data_filename=data/mimic-rebuild/mimic-table.csv
temp_directory=outputs/temp_outputs/
processed_data_directory=data/processed/
min_horizon=10

python -m src.build_datasets --data_filename=$data_filename --temp_directory=$temp_directory --processed_data_directory=$processed_data_directory --min_horizon=$min_horizon