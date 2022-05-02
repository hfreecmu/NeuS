calib_bin=./build/calibration
work_dir=./plants/plant2_halfsize
n_images=34

# Define the region of interest (ROI).
# We assume the center of the ROI is the center of the calibration board.
# The lower the value, the larger the ROI is. When the value is 1, the ROI is a sphere that exactly bound the calibration board.
board_scale=0.9

# Crop the pixels (to remove the black region) after undistortion.
crop_pixels=10

mkdir ${work_dir}/tmp
mkdir ${work_dir}/tmp/image_undistort
mkdir ${work_dir}/tmp/image_for_aruco

sudo ${calib_bin} ${work_dir} ${n_images} ${board_scale}

python gen_cameras.py ${work_dir} ${crop_pixels}
