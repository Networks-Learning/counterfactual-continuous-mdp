#!/bin/bash

temp_directory=outputs/temp_outputs/
device=cpu
seed=42
max_facility_num=30000

python -m src.facility --temp_directory=$temp_directory --device=$device --seed=$seed --max_facility_num=$max_facility_num