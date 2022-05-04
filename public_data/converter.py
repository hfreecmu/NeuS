import pdb

import os
import numpy as np
import cv2


# https://github.com/facebookresearch/co3d/blob/7ee9f5ba0b87b22e1dfe92c4d2010cb14dd467a6/dataset/co3d_dataset.py#L490
def co3d_rescale(principal_point, focal_length, im_wh):

    # first, we convert from the legacy Pytorch3D NDC convention
    # (currently used in CO3D for storing intrinsics) to pixels
    half_image_size_wh_orig = im_wh / 2.0

    # principal point and focal length in pixels
    principal_point_px = -1.0 * (principal_point - 1.0) * half_image_size_wh_orig
    focal_length_px = focal_length * half_image_size_wh_orig
    return principal_point_px, focal_length_px


def convert_NDC_to_screen_old_NDC(im_w, im_h, fx_ndc, fy_ndc, px_ndc, py_ndc):
    principal_point_ndc = np.array((px_ndc, py_ndc))
    focal_length_ndc = np.array((fx_ndc, fy_ndc))
    im_wh = np.array((im_w, im_h))

    principal_point_px, focal_length_px = co3d_rescale(
        principal_point_ndc, focal_length_ndc, im_wh
    )
    fx_screen, fy_screen = focal_length_px
    px_screen, py_screen = principal_point_px
    return fx_screen, fy_screen, px_screen, py_screen


def main():
    """Given a directory of CO3D data, converts it into nice NeUS data."""
    # Filtering based on whether "image" exist instead of images
    cases = [n for n in os.listdir(".") if os.path.isdir(n)]
    # The CO3D data has "images" instead of "image"
    cases = [n for n in cases if not os.path.exists(os.path.join(n, "image"))]

    for idx, case in enumerate(cases):
        print(f"Processing {case}: ({idx + 1}/{len(cases)})", end='\r')
        # Renaming stuff
        i_dir = os.path.join(case, "image")
        m_dir = os.path.join(case, "mask")
        os.rename(os.path.join(case, "images"), i_dir)
        os.rename(os.path.join(case, "masks"), m_dir)
        for im_name in [os.path.join(i_dir, n) for n in os.listdir(i_dir)]:
            os.rename(im_name, im_name.replace("frame", ""))
        for im_name in [os.path.join(m_dir, n) for n in os.listdir(m_dir)]:
            os.rename(im_name, im_name.replace("frame", ""))

        # Changing images to png
        for im_name in [os.path.join(i_dir, n) for n in os.listdir(i_dir)]:
            cv2.imwrite(im_name.replace(".jpg", ".png"), cv2.imread(im_name))
            os.remove(im_name)
        # Changing masks to binary
        for im_name in [os.path.join(m_dir, n) for n in os.listdir(m_dir)]:
            img = cv2.imread(im_name)
            cv2.imwrite(im_name, (img > 127).astype(np.uint8) * 255)

        # Camera parameters
        co_param = np.load(f"{case}/params.npz", allow_pickle=True)
        params = co_param['arr_0'].item()['frame_params']
        scale_mat = np.array(co_param['arr_0'].item()['scale'])
        scale_mat_zeros = np.zeros((4, 4), dtype=scale_mat.dtype)
        scale_mat_zeros[3, 3] = 1
        scale_mat_zeros[:3, :3] = scale_mat
        scale_mat = scale_mat_zeros
        neus_param = {}
        for data in params:
            name = data['path'].stem
            name_idx = name.replace("frame", "")
            #if int(name_idx) > 9:
            #    continue

            h, w = data['size']
            R = np.array(data['R']).T
            T = np.array(data['T'])
            ff = np.array(data['focal_length'])
            pp = np.array(data['principal_point'])

            K = np.zeros((3, 3), dtype=R.dtype)
            K[0, 0], K[1, 1], K[0, 2], K[1, 2] = convert_NDC_to_screen_old_NDC(
                                                    im_h=h,
                                                    im_w=w,
                                                    fx_ndc=ff[0],
                                                    fy_ndc=ff[1],
                                                    px_ndc=pp[0],
                                                    py_ndc=pp[1],
                                                )
            K[2, 2] = 1

            K[0, 0] = -K[0, 0]
            K[1, 1] = -K[1, 1]

            P = (K @ np.concatenate((R, T[:, None]), axis=1))
            P = np.concatenate((P, np.zeros((1, 4))), axis=0)
            P[3, 3] = 1

            neus_param[f'world_mat_{name_idx}'] = P
            neus_param[f'scale_mat_{name_idx}'] = scale_mat
        np.savez(f"{case}/cameras_sphere.npz", **neus_param)

    for idx, case in enumerate(sorted(cases)):
        print(f"Renamed case {case} to {idx:05d}")
        os.rename(case, f"{idx:05d}")


if __name__ == "__main__":
    main()
