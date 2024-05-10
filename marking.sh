#!/bin/bash

# Change this to the path of your Python script
script_path="main.py"

# Change these to the paths you want to test
model_path="/path/to/your/model"
path_folder="/path/to/your/folder"

# Run the script with the test paths
python $script_path -m $model_path -pf $path_folder