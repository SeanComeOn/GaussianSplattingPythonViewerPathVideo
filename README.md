# GaussianSplattingPythonViewerPathVideo
Intended to be a independent submodule in the original [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) repository. Aiming to provide:
1. Waypoint marking and camera pose exporting
2. camera pose interpolating
3. Video generation

with Python language.

The project is based on [Tiny Gaussian Splatting Viewer](https://github.com/limacv/GaussianSplattingViewer).

## How to Use
### Environment setup
Navigate to the Origianal Gaussian Splatting repository
```shell
cd /path/to/your/gaussian-splatting/
```
Get the repo and enter it
```shell
git clone https://github.com/SeanComeOn/GaussianSplattingPythonViewerPathVideo
cd GaussianSplattingPythonViewerPathVideo
```
### Mark the waypoints
Run the interactive viewer
```shell
python main.py -m <your_models_absolute_path> -pf <folder_for_path_to_be_generated>
```
You can navigate in the scene with keys and mouse, and record the waypoint.
The path will be saved to the directory under `<folder_for_path_to_be_generated>`. `mypaths/SavedCameraWaypoint.json` by default.
`paths/SavedCameraWaypoint.json` contains the json file for camera parameters defined by original 3D gaussian splatting project.

### Interpolate paths
```shell
python interpolate.py -pf <folder_for_path>
```
The script will read the `SavedCameraWaypoint.txt` under `<folder_for_path>`, `mypaths/SavedCameraWaypoint.txt` by default. It will generate a new file `DensePath.txt` under `<folder_for_path>`, containing the interpolated pose (position of 3D vector and rotation in quaternions) and camera intrinsics.

### Generate video
```shell
python gen_dense_video.py -m <your_models_absolute_path> -pf <folder_for_path>
```
The script will read `DensePath.txt` under `<folder_for_path>` and generate the video for the path.



-----------

Below is the original readme of [Tiny Gaussian Splatting Viewer](https://github.com/limacv/GaussianSplattingViewer)

# Tiny Gaussian Splatting Viewer
![UI demo](assets/teaser.png)
This is a simple Gaussian Splatting Viewer built with PyOpenGL / CUDARasterizer. It's easy to install with minimum dependencies. The goal of this project is to provide a minimum example of the viewer for research and study purpose. 

# News!
1/10/2024: The OpenGL renderer has faster sorting backend with `torch.argsort` & `cupy.argsort`. With cuda based sorting, it achieves nearly real-time sorting with OpenGL backend.

12/21/2023: Now we support rendering using the official cuda rasterizer!

# Usage
Install the dependencies:
```
pip install -r requirements.txt
```

Launch the viewer:
```
python main.py
```

You can check how to use UI in the "help" panel.

The Gaussian file loader is compatiable with the official implementation. 
Therefore, download pretrained Gaussian PLY file from [this official link](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip), and select the "point_cloud.ply" you like by clicking the 'open ply' button, and you are all set!


# Optional dependencies:

- If you want to use `cuda` backend for rendering, please install the [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization) following the guidance [here](https://github.com/graphdeco-inria/gaussian-splatting). And also install the following package:
```
pip install cuda-python
```

- For sorting, we provide three backend: `torch`, `cupy`, and `cpu`. The implementation will choose the first available one based on this priority order: `torch -> cupy -> cpu`. If you have `torch` or `cupy` backend, turning on `auto sort` will achieve nearly real-time sorting.
    - If you want to use `torch` as sorting backend, install any version of [PyTorch](https://pytorch.org/get-started/locally/).

    - If you want to use `cupy` to accelerate sorting, you should install the following package:
    ```
    pip install cupy-cuda11x // for cuda 11
    pip install cupy-cuda12x // for cuda 12
    ```


# Troubleshoot

The rendering speed of is comparable to the official CUDA renderer. If you're experiencing slow rendering, it's likely you are using integrated graphics card instead of a high-performance one. You can configure python to use high-performance graphics card in system settings. In Windows, you can set in Setting > System > Display > Graphics. See the screenshot below for example.

![Setting > System > Display > Graphics](assets/setting.png)

# Limitations
- The implementation utilizes SSBO, which is only support by OpenGL version >= 4.3. Although this version is widely adopted, MacOS is an exception. As a result, this viewer does not support MacOS.

- The `cuda` backend currently does not support other visualizations.

- Based on the flip test between the two backends, the unofficial implementation seems producing slightly different results compared with the official cuda version.

# TODO
- Make the projection matrix compatiable with official cuda implementation
- Tighter billboard to reduce number of fragments
- Save viewing parameters
