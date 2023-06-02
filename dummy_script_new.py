import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

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


params = np.load('/home/frc-ag-3/harry_ws/learning_3d_vis/final_project/fork_NeuS/public_data/fruitlet/cameras_sphere.npz')

fig=plt.figure()
ax = plt.axes(projection='3d')
idx = 0
for i in range(100):
    suffix = f'{i+1:03d}'

    try:
        world_mat = params['world_mat_' + str(int(suffix))]
        cam_mat = params['camera_mat_' + str(int(suffix))]
    except:
        #raise RuntimeError('here')
        continue
        
    #scale_mat = params['scale_mat_' + suffix]
    #P = world_mat @ scale_mat
    P = world_mat
    P = P[:3, :4]
    intrinsics, pose = load_K_Rt_from_P(None, P)

    idx+=1

    h = 1920
    w = 1080
    ff = np.array([cam_mat[0, 0], cam_mat[1, 1]])
    pp = np.array([cam_mat[0, 2], cam_mat[1, 2]])

    K = np.zeros((3, 3))
    K[0, 0], K[1, 1], K[0, 2], K[1, 2] = convert_NDC_to_screen(w, h, ff[0], ff[1], pp[0], pp[1])
    K[2, 2] = 1

    cam_x, cam_y, cam_z = pose[0:3, 3]

    intrinsics_inv = np.linalg.inv(intrinsics)
    p_x = w / 2
    p_y = h / 2
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


    ###
    #these match
    # print(intrinsics)
    # print('')
    # print(K)
    # print('')
    # print(np.max(np.abs(K - intrinsics[0:3, 0:3])))
    ###

    ###
    #first and last are the same
    # print(pose[:, 3])
    # print(T)
    # print(-R.T@T)
    ###

plt.show()

