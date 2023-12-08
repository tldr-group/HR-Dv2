from os import getcwd, mkdir
from shutil import copyfile

CWD = getcwd()
CUB = f"{CWD}/experiments/object_segmentation/datasets/CUB_200_2011/"
IMG_LIST_PATH = f"{CUB}images.txt"
TEST_TRAIN_PATH = f"{CUB}train_test_split.txt"

for out_dirs in [f"{CUB}test_images", f"{CUB}test_segmentations"]:
    try:
        mkdir(out_dirs)
    except FileExistsError:
        print("Out directories exist, passing")

with open(IMG_LIST_PATH, "r") as f:
    img_strs = f.read().splitlines()

with open(TEST_TRAIN_PATH, "r") as f:
    img_test_train_strs = f.read().splitlines()

for i, string in enumerate(img_strs):
    idx, path = string.split(" ")
    name = path[:-4]
    img_path = f"{CUB}images/{path}"
    seg_path = f"{CUB}segmentations/{name}.png"

    out_img_path = f"{CUB}test_images/{i}.jpg"
    out_seg_path = f"{CUB}test_segmentations/{i}.png"

    test_int = int(img_test_train_strs[i][-1])
    test: bool = True if test_int else False

    if test:
        copyfile(img_path, out_img_path)
        copyfile(seg_path, out_seg_path)
