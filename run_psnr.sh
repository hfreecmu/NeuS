python compute_metrics.py --conf ./confs/wmask.conf --mode psnr --case 00022_no_meta_full --weight_path exp/00022_no_meta_full/wmask/checkpoints/ckpt_020000.pth --is_continue --training-directory public_data/00022_no_meta_full/image/ --ref-directory public_data/00022_no_meta_full/image/
python compute_metrics.py --conf ./confs/wmask.conf --mode psnr --case 00022_no_meta_full --weight_path exp/00022_no_meta_10/wmask/checkpoints/ckpt_010000.pth   --is_continue --training-directory public_data/00022_no_meta_10/image/ --ref-directory public_data/00022_no_meta_full/image/
python compute_metrics.py --conf ./confs/wmask.conf --mode psnr --case 00022_no_meta_full --weight_path exp/00022_no_meta_30/wmask/checkpoints/ckpt_010000.pth   --is_continue --training-directory public_data/00022_no_meta_30/image/ --ref-directory public_data/00022_no_meta_full/image/
python compute_metrics.py --conf ./confs/wmask.conf --mode psnr --case 00022_no_meta_full --weight_path exp/00022_with_meta_10/wmask/checkpoints/ckpt_010000.pth --is_continue --training-directory public_data/00022_with_meta_10/image/ --ref-directory public_data/00022_no_meta_full/image/
python compute_metrics.py --conf ./confs/wmask.conf --mode psnr --case 00022_no_meta_full --weight_path exp/00022_with_meta_30/wmask/checkpoints/ckpt_010000.pth --is_continue --training-directory public_data/00022_with_meta_30/image/ --ref-directory public_data/00022_no_meta_full/image/

python compute_metrics.py --conf ./confs/wmask.conf --mode psnr --case 00032_no_meta_full --weight_path exp/00032_no_meta_full/wmask/checkpoints/ckpt_020000.pth --is_continue   --training-directory public_data/00032_no_meta_full/image/ --ref-directory public_data/00032_no_meta_full/image/
python compute_metrics.py --conf ./confs/wmask.conf --mode psnr --case 00032_no_meta_full --weight_path exp/00032_no_meta_10/wmask/checkpoints/ckpt_010000.pth   --is_continue   --training-directory public_data/00032_no_meta_10/image/ --ref-directory public_data/00032_no_meta_full/image/
python compute_metrics.py --conf ./confs/wmask.conf --mode psnr --case 00032_no_meta_full --weight_path exp/00032_no_meta_30/wmask/checkpoints/ckpt_010000.pth   --is_continue   --training-directory public_data/00032_no_meta_30/image/ --ref-directory public_data/00032_no_meta_full/image/
python compute_metrics.py --conf ./confs/wmask.conf --mode psnr --case 00032_no_meta_full --weight_path exp/00032_with_meta_10/wmask/checkpoints/ckpt_010000.pth --is_continue --training-directory public_data/00032_with_meta_10/image/ --ref-directory public_data/00032_no_meta_full/image/
python compute_metrics.py --conf ./confs/wmask.conf --mode psnr --case 00032_no_meta_full --weight_path exp/00032_with_meta_30/wmask/checkpoints/ckpt_010000.pth --is_continue --training-directory public_data/00032_with_meta_30/image/ --ref-directory public_data/00032_no_meta_full/image/

public_data/00022_no_meta_full/wmask/checkpoints/ckpt_020000.pth