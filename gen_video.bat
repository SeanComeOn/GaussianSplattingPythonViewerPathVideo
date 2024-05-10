@echo off
set model_path=D:/cg_play/vis/gaussian_splatting/output/train_naive
set folder_path=mypaths

python gen_dense_video.py -m %model_path% -pf %folder_path%