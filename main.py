# coding: utf-8
import glfw
import OpenGL.GL as gl
from imgui.integrations.glfw import GlfwRenderer
import imgui
import numpy as np
import util
import imageio
import util_gau
import tkinter as tk
from tkinter import filedialog
import os
import sys
import argparse
from renderer_ogl import OpenGLRenderer, GaussianRenderBase
import json

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

from scene.colmap_loader import qvec2rotmat, rotmat2qvec
import numpy as np
from torch import nn
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

from util import CUDA_Camera_Light
# # Add the directory containing main.py to the Python path
# dir_path = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(dir_path)

# # Change the current working directory to the script's directory
# os.chdir(os.path.dirname(os.path.abspath(__file__)))




g_camera = util.Camera(720, 1280)
now_view = None
BACKEND_OGL=0
BACKEND_CUDA=1
g_renderer_list = [
    None, # ogl
]
g_renderer_idx = BACKEND_OGL
g_renderer = g_renderer_list[g_renderer_idx]
g_scale_modifier = 1.
g_auto_sort = False
g_show_control_win = True
g_show_cuda_raster_settings = True
g_show_help_win = True
g_show_camera_win = False
g_render_mode_tables = ["Gaussian Ball", "Flat Ball", "Billboard", "Depth", "SH:0", "SH:0~1", "SH:0~2", "SH:0~3 (default)"]
g_render_mode = 7

g_recorded_point_num = 0
g_recorded_raster_setting = []

g_T = np.array([0.0, 0.0, 0.0])
g_R = np.array([[1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]])

def impl_glfw_init():
    # 窗口名称
    window_name = "NeUVF editor"

    # 创建OpenGL上下文。这些是由glfw完成的。
    if not glfw.init():
        print("Could not initialize OpenGL context")
        exit(1)

    # 创建相关的版本信息
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    # glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

    # Create a windowed mode window and its OpenGL context
    global window
    window = glfw.create_window(
        g_camera.w, g_camera.h, window_name, None, None
    )
    glfw.make_context_current(window)
    glfw.swap_interval(0)
    # glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL);
    if not window:
        glfw.terminate()
        print("Could not initialize Window")
        exit(1)

    return window

# 此回调函数，鼠标控制位姿
def cursor_pos_callback(window, xpos, ypos):
    if imgui.get_io().want_capture_mouse:
        g_camera.is_leftmouse_pressed = False
        g_camera.is_rightmouse_pressed = False
    g_camera.process_mouse(xpos, ypos)

def mouse_button_callback(window, button, action, mod):
    if imgui.get_io().want_capture_mouse:
        return
    pressed = action == glfw.PRESS
    g_camera.is_leftmouse_pressed = (button == glfw.MOUSE_BUTTON_LEFT and pressed)
    g_camera.is_rightmouse_pressed = (button == glfw.MOUSE_BUTTON_RIGHT and pressed)

def wheel_callback(window, dx, dy):
    g_camera.process_wheel(dx, dy)

def key_callback(window, key, scancode, action, mods):
    if action == glfw.REPEAT or action == glfw.PRESS:
        if key == glfw.KEY_U:
            g_camera.process_roll_key(1)
        elif key == glfw.KEY_O:
            g_camera.process_roll_key(-1)
        elif key == glfw.KEY_W:
            g_camera.process_mov_forward_key(1)
        elif key == glfw.KEY_S:
            g_camera.process_mov_forward_key(-1)
        elif key == glfw.KEY_A:
            g_camera.process_mov_right_key(-1)
        elif key == glfw.KEY_D:
            g_camera.process_mov_right_key(1)
        elif key == glfw.KEY_Q:
            g_camera.process_mov_up_key(-1)
        elif key == glfw.KEY_E:
            g_camera.process_mov_up_key(1)
        
        if key == glfw.KEY_Y:
            now_view.process_rollz_key(1)
        elif key == glfw.KEY_R:
            now_view.process_rollz_key(-1)
        elif key == glfw.KEY_T:
            now_view.process_rollx_key(1)
        elif key == glfw.KEY_G:
            now_view.process_rollx_key(-1)
        elif key == glfw.KEY_H:
            now_view.process_rolly_key(1)
        elif key == glfw.KEY_F:
            now_view.process_rolly_key(-1)
        elif key == glfw.KEY_W:
            now_view.process_mov_z_key(1)
        elif key == glfw.KEY_S:
            now_view.process_mov_z_key(-1)
        elif key == glfw.KEY_A:
            now_view.process_mov_x_key(-1)
        elif key == glfw.KEY_D:
            now_view.process_mov_x_key(1)
        elif key == glfw.KEY_Q:
            now_view.process_mov_y_key(-1)
        elif key == glfw.KEY_E:
            now_view.process_mov_y_key(1)


def update_camera_pose_lazy():
    if g_camera.is_pose_dirty:
        g_renderer.update_camera_pose(g_camera)
        g_camera.is_pose_dirty = False
    if now_view.is_pose_dirty:
        now_view.update_camera_pose()
        now_view.is_pose_dirty = False

def update_camera_intrin_lazy():
    if g_camera.is_intrin_dirty:
        g_renderer.update_camera_intrin(g_camera)
        g_camera.is_intrin_dirty = False

def update_activated_renderer_state(gaus: util_gau.GaussianData):
    g_renderer.update_gaussian_data(gaus)
    g_renderer.sort_and_update(g_camera)
    g_renderer.set_scale_modifier(g_scale_modifier)
    g_renderer.set_render_mod(g_render_mode - 3)
    g_renderer.update_camera_pose(g_camera)
    g_renderer.update_camera_intrin(g_camera)
    g_renderer.set_render_reso(g_camera.w, g_camera.h)

def window_resize_callback(window, width, height):
    gl.glViewport(0, 0, width, height)
    g_camera.update_resolution(height, width)
    g_renderer.set_render_reso(width, height)

def main():
    # 全局的参数
    global g_camera, g_renderer, g_renderer_list, g_renderer_idx, g_scale_modifier, g_auto_sort, \
        g_show_control_win, g_show_help_win, g_show_camera_win, g_show_cuda_raster_settings, \
        g_render_mode, g_render_mode_tables
    global g_recorded_point_num
    global g_recorded_raster_setting

    global g_T, g_R
    global now_view

    #################################################################
    # load gaussian scene
    global args

    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--hidpi", action="store_true", help="Enable HiDPI scaling for the interface.")
    parser.add_argument('-pf', '--path_folder', type=str, help='Folder for path to be generated')
    # args = parser.parse_args()
    # Use the arguments

    # Set up command line argument parser
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    path_folder = args.path_folder
    # render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
    dataset = model.extract(args)
    iteration = args.iteration
    pipeline = pipeline.extract(args)
    skip_train = args.skip_train
    skip_test = args.skip_test

    with torch.no_grad():
        my_gaussian = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, my_gaussian, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        views = scene.getTrainCameras()

        init_view = views[0]
        fovx = init_view.FoVx
        fovy = init_view.FoVy
        R = init_view.R
        T = init_view.T
        # w = init_view.image_width
        # h = init_view.image_height
        w = 1280
        h = 720
        print(f"fovx = {fovx}, fovy = {fovy}")
        print(f"R = {R}")
        print(f"T = {T}")
        print(f"w = {w}, h = {h}")


        now_view = CUDA_Camera_Light(R=R, T=T, FoVx=fovx, FoVy=fovy, w=w, h=h)
        rendering = render(now_view, my_gaussian, pipeline, background)["render"]
        torchvision.utils.save_image(rendering, "aaa.png")
        



    # 创建界面上下文
    imgui.create_context()
    if args.hidpi:
        imgui.get_io().font_global_scale = 1.5

    # 创建窗口
    window = impl_glfw_init()
    impl = GlfwRenderer(window)
    root = tk.Tk()  # used for file dialog
    root.withdraw()
    
    # 设置几个回调函数
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_scroll_callback(window, wheel_callback)
    glfw.set_key_callback(window, key_callback)
    
    # 窗口大小变化的回调函数
    glfw.set_window_size_callback(window, window_resize_callback)

    # init renderer
    # 初始化OpenGL的渲染器
    g_renderer_list[BACKEND_OGL] = OpenGLRenderer(g_camera.w, g_camera.h)
    # 选择渲染器
    g_renderer_idx = BACKEND_OGL
    try:
        from renderer_cuda import CUDARenderer
        # 尝试装载CUDARenderer
        g_renderer_list += [CUDARenderer(g_camera.w, g_camera.h)]
        g_renderer_idx = BACKEND_CUDA
    except ImportError:
        print("Info: No CUDARenderer")
        pass
    
    g_renderer = g_renderer_list[g_renderer_idx]

    # 装载最初的几个示例gaussian
    # gaussian data
    gaussians = util_gau.naive_gaussian()
    try:
        gaussians = util_gau.load_ply("D:/cg_play/2023_12_19_style_transfer_and_gaussian/2024_02_28_second_trial/tocp/point_cloud/iteration_7000/point_cloud.ply")
    except RuntimeError as e:
        pass
    update_activated_renderer_state(gaussians)
    
    # settings
    # 只要不关闭窗口
    while not glfw.window_should_close(window):
        # 处理所有的待处理事件
        glfw.poll_events()
        # 处理输入
        impl.process_inputs()
        imgui.new_frame()
        
        gl.glClearColor(0, 0, 0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        # dirty的时候再更新
        update_camera_pose_lazy()
        update_camera_intrin_lazy()
        
        # 调用renderer的draw方法
        # g_renderer.draw()
        g_renderer.draw_scene_setting(now_view, my_gaussian, pipeline, background)

        # imgui ui
        # update control information
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("Window", True):
                clicked, g_show_control_win = imgui.menu_item(
                    "Show Control", None, g_show_control_win
                )
                clicked, g_show_help_win = imgui.menu_item(
                    "Show Help", None, g_show_help_win
                )
                clicked, g_show_camera_win = imgui.menu_item(
                    "Show Camera Control", None, g_show_camera_win
                )
                
                imgui.end_menu()
            imgui.end_main_menu_bar()
        
        if g_show_cuda_raster_settings:
            if imgui.begin("Raster_settings", True):
                if g_renderer_idx == BACKEND_CUDA:
                    # ras = g_renderer.get_raster_setting_dict_for_json()
                    # ras = g_camera.get_view_matrix().tolist()
                    t,r = now_view.get_real_tr()
                    ras = {'t': t.tolist(), 'r': r.tolist()}
                    if imgui.button(label='Record Current Point'):
                        g_recorded_raster_setting.append({f"{g_recorded_point_num}":ras})
                        g_recorded_point_num += 1
                    imgui.text(f"Recorded num = {g_recorded_point_num}")
                    if imgui.button(label='Clear'):
                        g_recorded_point_num = 0
                        g_recorded_raster_setting = []
                    if imgui.button(label='Save Mark'):
                        # create directory if not exist
                        path_folder = "mypaths"
                        if not os.path.exists(path_folder):
                            os.makedirs(path_folder)
                        filename = os.path.join(path_folder, "SavedCameraWaypoint.json")
                        with open(filename, 'w') as f:
                            # 写入字符串
                            f.write(json.dumps(g_recorded_raster_setting, indent=4))
                    if imgui.button(label='Gen Video From Path'):
                        
                        pass
                    imgui.text(json.dumps(ras, indent=4))
                    # Intrinsics:
                    imgui.text(f"fovy = {imgui.get_io().framerate:.1f}")
                    # Extrinsics:
                    
                else:
                    imgui.text(f"Not CUDA rasterizing now.")
                imgui.end()
        if g_show_control_win:
            if imgui.begin("Control", True):
                # rendering backend
                changed, g_renderer_idx = imgui.combo("backend", g_renderer_idx, ["ogl", "cuda"][:len(g_renderer_list)])
                if changed:
                    g_renderer = g_renderer_list[g_renderer_idx]
                    update_activated_renderer_state(gaussians)

                imgui.text(f"fps = {imgui.get_io().framerate:.1f}")
                imgui.text(f"# of Gaus = {len(gaussians)}")
                if imgui.button(label='open ply'):
                    file_path = filedialog.askopenfilename(title="open ply",
                        initialdir="D:/cg_play/2023_12_19_style_transfer_and_gaussian/2024_02_28_second_trial/tocp/point_cloud/iteration_7000",
                        filetypes=[('ply file', '.ply')]
                        )
                    if file_path:
                        try:
                            gaussians = util_gau.load_ply(file_path)
                            g_renderer.update_gaussian_data(gaussians)
                            g_renderer.sort_and_update(g_camera)
                        except RuntimeError as e:
                            pass
                
                # camera fov
                changed, g_camera.fovy = imgui.slider_float(
                    "fov", g_camera.fovy, 0.001, np.pi - 0.001, "fov = %.3f"
                )
                g_camera.is_intrin_dirty = changed
                # update_camera_intrin_lazy()
                
                # scale modifier
                changed, g_scale_modifier = imgui.slider_float(
                    "", g_scale_modifier, 0.1, 10, "scale modifier = %.3f"
                )
                imgui.same_line()
                if imgui.button(label="reset"):
                    g_scale_modifier = 1.
                    changed = True
                    
                if changed:
                    g_renderer.set_scale_modifier(g_scale_modifier)
                
                # render mode
                changed, g_render_mode = imgui.combo("shading", g_render_mode, g_render_mode_tables)
                if changed:
                    g_renderer.set_render_mod(g_render_mode - 4)
                
                # sort button
                if imgui.button(label='sort Gaussians'):
                    g_renderer.sort_and_update(g_camera)
                imgui.same_line()
                changed, g_auto_sort = imgui.checkbox(
                        "auto sort", g_auto_sort,
                    )
                if g_auto_sort:
                    g_renderer.sort_and_update(g_camera)
                
                if imgui.button(label='save image'):
                    width, height = glfw.get_framebuffer_size(window)
                    nrChannels = 3;
                    stride = nrChannels * width;
                    stride += (4 - stride % 4) if stride % 4 else 0
                    gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 4)
                    gl.glReadBuffer(gl.GL_FRONT)
                    bufferdata = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
                    img = np.frombuffer(bufferdata, np.uint8, -1).reshape(height, width, 3)
                    imageio.imwrite("save.png", img[::-1])
                    # save intermediate information
                    # np.savez(
                    #     "save.npz",
                    #     gau_xyz=gaussians.xyz,
                    #     gau_s=gaussians.scale,
                    #     gau_rot=gaussians.rot,
                    #     gau_c=gaussians.sh,
                    #     gau_a=gaussians.opacity,
                    #     viewmat=g_camera.get_view_matrix(),
                    #     projmat=g_camera.get_project_matrix(),
                    #     hfovxyfocal=g_camera.get_htanfovxy_focal()
                    # )
                
                imgui.end()

        if g_show_camera_win:
            if imgui.button(label='rot 180'):
                g_camera.flip_ground()

            changed, g_camera.target_dist = imgui.slider_float(
                    "t", g_camera.target_dist, 1., 8., "target dist = %.3f"
                )
            if changed:
                g_camera.update_target_distance()

            changed, g_camera.rot_sensitivity = imgui.slider_float(
                    "r", g_camera.rot_sensitivity, 0.002, 0.1, "rotate speed = %.3f"
                )
            imgui.same_line()
            if imgui.button(label="reset r"):
                g_camera.rot_sensitivity = 0.02

            changed, g_camera.trans_sensitivity = imgui.slider_float(
                    "m", g_camera.trans_sensitivity, 0.001, 0.03, "move speed = %.3f"
                )
            imgui.same_line()
            if imgui.button(label="reset m"):
                g_camera.trans_sensitivity = 0.01

            changed, g_camera.zoom_sensitivity = imgui.slider_float(
                    "z", g_camera.zoom_sensitivity, 0.001, 0.05, "zoom speed = %.3f"
                )
            imgui.same_line()
            if imgui.button(label="reset z"):
                g_camera.zoom_sensitivity = 0.01

            changed, g_camera.roll_sensitivity = imgui.slider_float(
                    "ro", g_camera.roll_sensitivity, 0.003, 0.1, "roll speed = %.3f"
                )
            imgui.same_line()
            if imgui.button(label="reset ro"):
                g_camera.roll_sensitivity = 0.03



        if g_show_help_win:
            imgui.begin("Help", True)
            imgui.text("Open Gaussian Splatting PLY file \n  by click 'open ply' button")
            imgui.text("Use left click & move to rotate camera")
            imgui.text("Use right click & move to translate camera")
            imgui.text("Press Q/E to roll camera")
            imgui.text("Use scroll to zoom in/out")
            imgui.text("Use control panel to change setting")
            imgui.end()
        
        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

    impl.shutdown()
    glfw.terminate()


if __name__ == "__main__":



    main()
