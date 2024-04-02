from os import getcwd, mkdir, listdir
from shutil import copyfile

CWD = getcwd()
DUTS = f"{CWD}/experiments/object_segmentation/datasets/DUTS-TE/"
IMG_DIR = f"{DUTS}DUTS-TE-Image/"
SEG_DIR = f"{DUTS}DUTS-TE-Mask/"

for out_dirs in [f"{DUTS}test_images", f"{DUTS}test_segmentations"]:
    try:
        mkdir(out_dirs)
    except FileExistsError:
        print("Out directories exist, passing")

img_files = listdir(IMG_DIR)
seg_files = listdir(SEG_DIR)

test_idx = 0
for i in range(len(img_files)):
    img_name, seg_name = img_files[i], seg_files[i]
    name = img_name[:-4]
    if "test" in img_name:
        in_img_path = f"{IMG_DIR}{img_name}"
        in_seg_path = f"{SEG_DIR}{name}.png"
        out_img_path = f"{DUTS}test_images/{test_idx}.jpg"
        out_seg_path = f"{DUTS}test_segmentations/{test_idx}.png"

        copyfile(in_img_path, out_img_path)
        copyfile(in_seg_path, out_seg_path)

        test_idx += 1
