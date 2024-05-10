#!/bin/bash


# Assign arguments to variables for better readability
model_path="D:/cg_play/vis/gaussian_splatting/output/train_naive"
folder_path="mypaths"

# Run the python script
python gen_dense_video.py -m $model_path -pf $folder_path