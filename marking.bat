@REM @echo off

REM Change this to the path of your Python script
set script_path=main.py

REM Change these to the paths you want to test
set model_path=D:/cg_play/vis/gaussian_splatting/output/train_naive
set path_folder=ourpathss

REM Run the script with the test paths
python %script_path% -m %model_path% -pf %path_folder%