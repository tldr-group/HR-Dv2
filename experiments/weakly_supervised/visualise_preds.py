
from skimage.color import label2rgb
from scipy.ndimage import zoom
from skimage.transform import resize
import matplotlib.pyplot as plt
from os import getcwd

from tifffile import imread

import numpy as np


examples = ["Frame5_4.6x.tif", # lighter
            "Box13_P1_JK_4600x_0118.tif", # weird shape
            "Box13_P1_JK_4600x_0103.tif", 
            "Box13_P1_JK_4600x_0099.tif",
            "Box13_O2_JK_4600x_0004.tif", # darker
            ]

PREFIX = "experiments/weakly_supervised"
MASK_SUFFIX = "_segmentation.tifnomalized.tif"

PRED_FOLDERS = ["masks", "preds/dv2_out", "preds/rf_out_no_crf"]

color_list = [[255, 255, 255], [31, 119, 180], [255, 127, 14], [44, 160, 44]]
COLORS = np.array(color_list) / 255.0

def remap_label_arr(arr: np.ndarray) -> np.ndarray:
    return (arr // np.unique(arr)[0]) - 1

print(getcwd())

fig, axs = plt.subplots(nrows=3, ncols=len(examples))

fig.set_size_inches(25, 15)
plt.rcParams["font.family"] = "serif"
titles = ["Ground Truth", "Deep Features", "Classical Features"]

for col, fname in enumerate(examples):
    original = imread(f"{PREFIX}/data/{fname}")
    

    for row, pred_folder in enumerate(PRED_FOLDERS):
        ax = axs[row, col]

        if (col == 0):
            ax.set_ylabel(titles[row], fontsize=26)

        suffix = MASK_SUFFIX if row == 0 else ""
        
        low_res_data = imread(f"{PREFIX}/{pred_folder}/{fname}{suffix}")
        low_res_data = remap_label_arr(low_res_data)
        
        data = resize(low_res_data, (1024, 1024), preserve_range=True) if row > 0 else low_res_data
        data = data.astype(np.uint8)
        overlay = label2rgb(data, original, colors=COLORS[1:], alpha=0.2, bg_label=-1)
        
        ax.imshow(overlay)
        ax.set_xticks([])
        ax.set_yticks([])
        #ax.set_axis_off()
plt.tight_layout()
plt.savefig(f"{PREFIX}/out/poster_figure.png")