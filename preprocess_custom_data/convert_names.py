import os
import glob
import sys

names = sorted([n for n in glob.glob(os.path.join(sys.argv[1], "*")) if n.endswith(".png")])
for idx, n in enumerate(names):
    new_n = os.path.join(os.path.split(n)[0], f"{idx:03d}.png")
    os.rename(n, new_n)







