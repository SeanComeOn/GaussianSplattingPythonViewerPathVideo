import numpy as np

# from https://github.com/isl-org/FreeViewSynthesis

def quat_from_rotm(R):
    R = R.reshape(-1, 3, 3)
    w = np.sqrt(np.maximum(0, 1 + R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]))
    x = np.sqrt(np.maximum(0, 1 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2]))
    y = np.sqrt(np.maximum(0, 1 - R[:, 0, 0] + R[:, 1, 1] - R[:, 2, 2]))
    z = np.sqrt(np.maximum(0, 1 - R[:, 0, 0] - R[:, 1, 1] + R[:, 2, 2]))
    q0 = np.empty((R.shape[0], 4), dtype=R.dtype)
    q0[:, 0] = w
    q0[:, 1] = x * np.sign(x * (R[:, 2, 1] - R[:, 1, 2]))
    q0[:, 2] = y * np.sign(y * (R[:, 0, 2] - R[:, 2, 0]))
    q0[:, 3] = z * np.sign(z * (R[:, 1, 0] - R[:, 0, 1]))
    q1 = np.empty((R.shape[0], 4), dtype=R.dtype)
    q1[:, 0] = w * np.sign(w * (R[:, 2, 1] - R[:, 1, 2]))
    q1[:, 1] = x
    q1[:, 2] = y * np.sign(y * (R[:, 1, 0] + R[:, 0, 1]))
    q1[:, 3] = z * np.sign(z * (R[:, 0, 2] + R[:, 2, 0]))
    q2 = np.empty((R.shape[0], 4), dtype=R.dtype)
    q2[:, 0] = w * np.sign(w * (R[:, 0, 2] - R[:, 2, 0]))
    q2[:, 1] = x * np.sign(x * (R[:, 0, 1] + R[:, 1, 0]))
    q2[:, 2] = y
    q2[:, 3] = z * np.sign(z * (R[:, 1, 2] + R[:, 2, 1]))
    q3 = np.empty((R.shape[0], 4), dtype=R.dtype)
    q3[:, 0] = w * np.sign(w * (R[:, 1, 0] - R[:, 0, 1]))
    q3[:, 1] = x * np.sign(x * (R[:, 0, 2] + R[:, 2, 0]))
    q3[:, 2] = y * np.sign(y * (R[:, 1, 2] + R[:, 2, 1]))
    q3[:, 3] = z
    q = q0 * (w[:, None] > 0) + (w[:, None] == 0) * (
        q1 * (x[:, None] > 0)
        + (x[:, None] == 0) * (q2 * (y[:, None] > 0) + (y[:, None] == 0) * (q3))
    )
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q.squeeze()


def cameracenter_from_translation(R, t):
    t = t.reshape(-1, 3, 1)
    R = R.reshape(-1, 3, 3)
    C = -R.transpose(0, 2, 1) @ t
    return C.squeeze()


def quat_slerp_space(q0, q1, t=None, num=100, endpoint=True):
    q0 = q0.reshape(-1, 4)
    q1 = q1.reshape(-1, 4)
    dot = (q0 * q1).sum(axis=1)

    ma = dot < 0
    q1[ma] *= -1
    dot[ma] *= -1

    if t is None:
        t = np.linspace(0, 1, num=num, endpoint=endpoint, dtype=q0.dtype)
    t = t.reshape((-1, 1))
    num = t.shape[0]

    res = np.empty((q0.shape[0], num, 4), dtype=q0.dtype)

    ma = dot > 0.9995
    if np.any(ma):
        res[ma] = (q0[ma] + t[..., None] * (q1[ma] - q0[ma])).transpose(1, 0, 2)

    ma = ~ma
    if np.any(ma):
        q0 = q0[ma]
        q1 = q1[ma]
        dot = dot[ma]

        dot = np.clip(dot, -1, 1)
        theta0 = np.arccos(dot)
        theta = theta0 * t
        s0 = np.cos(theta) - dot * np.sin(theta) / np.sin(theta0)
        s1 = np.sin(theta) / np.sin(theta0)
        res[ma] = ((s0[..., None] * q0) + (s1[..., None] * q1)).transpose(
            1, 0, 2
        )
    return res.squeeze()


def rotm_from_quat(q):
    q = q.reshape(-1, 4)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.array(
        [
            [
                1 - 2 * y * y - 2 * z * z,
                2 * x * y - 2 * z * w,
                2 * x * z + 2 * y * w,
            ],
            [
                2 * x * y + 2 * z * w,
                1 - 2 * x * x - 2 * z * z,
                2 * y * z - 2 * x * w,
            ],
            [
                2 * x * z - 2 * y * w,
                2 * y * z + 2 * x * w,
                1 - 2 * x * x - 2 * y * y,
            ],
        ],
        dtype=q.dtype,
    )
    R = R.transpose((2, 0, 1))
    return R.squeeze()


def translation_from_cameracenter(R, C):
    C = C.reshape(-1, 3, 1)
    R = R.reshape(-1, 3, 3)
    t = -R @ C
    return t.squeeze()


def interpolate_waypoints(wpRs, wpts, steps=25):
    wpRs = np.array(wpRs)
    wpts = np.array(wpts)
    wpqs = quat_from_rotm(wpRs)
    wpCs = cameracenter_from_translation(wpRs, wpts)
    qs, Cs = [], []
    for idx in range(wpRs.shape[0] - 1):
        q0, q1 = wpqs[idx], wpqs[idx + 1]
        C0, C1 = wpCs[idx], wpCs[idx + 1]
        alphas = np.linspace(0, 1, num=steps, endpoint=False)
        Cs.append((1 - alphas[:, None]) * C0 + alphas[:, None] * C1)
        qs.append(quat_slerp_space(q0, q1, t=alphas))
    Rs = rotm_from_quat(np.vstack(qs))
    ts = translation_from_cameracenter(Rs, np.vstack(Cs))
    return Rs, ts

import sys
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-pf', '--path_folder', default='mypaths', help='Folder for path')
    return parser.parse_args()

import json

def read_waypoints(file_path):
    with open(file_path, 'r') as f:
        waypoints = json.load(f)
    # Convert the waypoints to a numpy array
    ts = []
    rs = []
    for waypoint in waypoints:
        for value in waypoint.values():
            ts.append(np.array(value['t']))
            rs.append(np.array(value['r']))
            
    return ts, rs

def write_interpolated(interpolated, file_path):
    with open(file_path, 'w') as f:
        for line in interpolated:
            f.write(' '.join(map(str, line)) + '\n')


def main():
    args = parse_args()
    wpts, wprs = read_waypoints(f'{args.path_folder}/SavedCameraWaypoint.json')
    rs, ts = interpolate_waypoints(wprs, wpts, steps=40)

    allpoints = []
    for i in range(len(rs)):
        ras = {"r": rs[i].tolist(), "t": ts[i].tolist()}
        allpoints.append(ras)
    json.dump(allpoints, open(f'{args.path_folder}/DensePath.json', 'w'), indent=4)
    # write_interpolated(interpolated, f'{args.path_folder}/DensePath.txt')

if __name__ == '__main__':
    main()