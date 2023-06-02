import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pytorch3d
import torch
from pytorch3d.renderer import (
    PerspectiveCameras,
)
from pytorch3d.renderer.cameras import get_ndc_to_screen_transform

def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]

    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

def convert_NDC_to_screen(
    im_w, im_h, fx_ndc, fy_ndc, px_ndc, py_ndc
):
    s = min(im_w, im_h)
    px_screen = -(px_ndc * s / 2) + im_w / 2
    py_screen = -(py_ndc * s / 2) + im_h / 2
    fx_screen = fx_ndc * s / 2
    fy_screen = fy_ndc * s / 2
    return fx_screen, fy_screen, px_screen, py_screen


params = np.load('/home/frc-ag-3/harry_ws/learning_3d_vis/final_project/NeuS/public_data/plant/cameras_sphere.npz')

fig=plt.figure()
ax = plt.axes(projection='3d')
for i in range(400):
    suffix = f'{i+1:03d}'

    try:
        world_mat = params['world_mat_' + suffix]
    except:
        continue
        
    scale_mat = params['scale_mat_' + suffix]

    #P = world_mat @ scale_mat
    P = world_mat
    P_inv = np.linalg.inv(P)
    P = P[:3, :4]
    intrinsics, pose = load_K_Rt_from_P(None, P)

    intrinsics_inv = np.linalg.inv(intrinsics)
    p_x = 2454 / 2
    p_y = 2056 / 2
    p = np.array([p_x, p_y, 1])
    p = intrinsics_inv[0:3, 0:3] @ p
    v = pose[0:3, 0:3] @ p
    v = v / np.linalg.norm(v)
    o = pose[0:3, 3] #o matches cam_x, cam_y, cam_z which is good

    p0 = o
    p1 = o + v
    p2 = o + 0.1*v

    _ = ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], c='cyan')
    _ = ax.plot([p0[0], p2[0]], [p0[1], p2[1]], [p0[2], p2[2]], c='red')


plt.show()


###
#this also works
# ext_0 = np.eye(4)
# ext_0[0:3, 0:3] = R
# ext_0[0:3, 3] = T

# ext_1 = np.eye(4)
# ext_1[0:3, 0:3] = R.T
# ext_1[0:3, 3] = -R.T @ T

# print(ext_0 @ ext_1)
# print('')
# print(ext_1 @ ext_0)
###
