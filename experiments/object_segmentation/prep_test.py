from os import getcwd, mkdir, listdir
from shutil import copyfile
import numpy as np

np.random.seed(2189)

CWD = getcwd()
BOTH = f"{CWD}/experiments/object_segmentation/datasets/both/"
CUB = f"{CWD}/experiments/object_segmentation/datasets/CUB_200_2011/"
DUTS = f"{CWD}/experiments/object_segmentation/datasets/DUTS-TE/"

CUB_IMG_DIR = f"{CUB}test_images/"
CUB_SEG_DIR = f"{CUB}test_segmentations/"

DUTS_IMG_DIR = f"{DUTS}test_images/"
DUTS_SEG_DIR = f"{DUTS}test_segmentations/"

for out_dirs in [f"{BOTH}test_images", f"{BOTH}test_segmentations"]:
    try:
        mkdir(out_dirs)
    except FileExistsError:
        print("Out directories exist, passing")

n_duts_imgs = len(listdir(DUTS_IMG_DIR))
n_cubs_imgs = len(listdir(CUB_IMG_DIR))

N_IMGS = 200
duts_idx = np.random.choice(n_duts_imgs, N_IMGS, replace=False)
cubs_idx = np.random.choice(n_cubs_imgs, N_IMGS, replace=False)
idxs = [duts_idx, cubs_idx]
img_dirs = [DUTS_IMG_DIR, CUB_IMG_DIR]
seg_dirs = [DUTS_SEG_DIR, CUB_SEG_DIR]

for i in range(N_IMGS):
    for offset in [0, 1]:
        old_idx = idxs[offset][i]
        new_idx = 2 * i + offset
        img_dir, seg_dir = img_dirs[offset], seg_dirs[offset]

        copyfile(f"{img_dir}{old_idx}.jpg", f"{BOTH}/test_images/{new_idx}.jpg")
        copyfile(f"{seg_dir}{old_idx}.png", f"{BOTH}/test_segmentations/{new_idx}.png")
