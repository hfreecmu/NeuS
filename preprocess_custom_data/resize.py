import os
import glob
import sys
import cv2


names = sorted([n for n in glob.glob(os.path.join(sys.argv[1], "*")) if n.endswith(".png")])
for idx, n in enumerate(names):
    out_name = os.path.join("resized", os.path.split(n)[1])
    out_img = cv2.resize(cv2.imread(n), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(out_name, out_img)









