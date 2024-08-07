from os import listdir
from os.path import join
from tifffile import imread, imwrite
from PIL import Image
import numpy as np

np.random.seed(1001)

import torch
from torchmetrics.classification.jaccard import JaccardIndex
from torch.nn.functional import interpolate

from multiprocessing import Queue
from classifiers import DeepFeaturesModel, WekaFeaturesModel, get_featuriser_classifier
from data_model import resize_longest_side

from skimage.util import random_noise

# TODO: add poisson noise=shotnoise (and also gaussian=thermal?) and show 0-shot perf as noise increases?


default_mapping = [(255, 0), (170, 1), (85, 2)]


def tiff_to_labels(tiff: np.ndarray, mapping: list[tuple[int, int]]) -> np.ndarray:
    out = tiff
    for val, i in mapping:
        out = np.where(out == val, i, out)
    return out


def save_seg(seg: np.ndarray, fname: str):
    rescaled = seg
    for i in range(3):
        rescaled = np.where(rescaled == i, default_mapping[i][0], rescaled)

    imwrite(fname, rescaled.astype(np.uint8), photometric="minisblack")

def add_noise(input_arr: np.ndarray, possion: bool=True, gauss_level: float=0) -> np.ndarray:
    result = input_arr / 255.0
    if possion:
        result = random_noise(result, mode='poisson')
    if gauss_level != 0:
        result = random_noise(result, mode='gaussian', var=gauss_level)
    result = ((result)  * 255)
    return result


L = 518

do_dino = False
if do_dino:
    model_name = "DINO-S-8" #DINO-S-8 DINOv2-S-14
    model_path = "/home/ronan/HR-Dv2/experiments/weakly_supervised/models/d_real.pkl"
    out_folder = "preds/d_out_no_crf"
else:
    model_name = "random_forest"
    model_path = "/home/ronan/HR-Dv2/experiments/weakly_supervised/models/rf_real.pkl"
    out_folder = "preds/rf_out"



def main_loop(model_path: str, model_name: str, prefix: str, mapping: list[tuple[int, int]], save: bool=False, noise_params: tuple[bool, float]=(False, 0)) -> None:
    model = get_featuriser_classifier(model_name, Queue(2), Queue(2))
    model.load_model(model_path)

    model.do_crf = True

    jac = JaccardIndex(num_classes=3, task="multiclass")
    expr_folder = "/home/ronan/HR-Dv2/experiments/weakly_supervised"

    vals = []

    for i, f in enumerate(listdir(join(expr_folder, "data"))):
        if i % 30 == 0:
            print(f"{i}/150")
        inp_path = join(expr_folder, "data", f)
        out_path = join(expr_folder, "masks", f"{f[:-4]}.tif_segmentation.tifnomalized.tif")

        inp_data = imread(inp_path)
        if noise_params != (False, 0):
            inp_data = add_noise(inp_data, noise_params[0], noise_params[1])
        out_data = tiff_to_labels(imread(out_path), default_mapping)

        inp_img = Image.fromarray(inp_data).convert("RGB")
        if save:
            inp_img.save(f'foo_{noise_params[1]}.png')

        inp_img = resize_longest_side(inp_img, L)
        features = model.img_to_features(inp_img)
        out_segs = model.segment([features], [inp_img], [0], False)
        out_seg = out_segs[0] - 1

        gt_tensor = torch.tensor(out_data, dtype=torch.float32)
        gt_tensor = interpolate(
            gt_tensor.unsqueeze(0).unsqueeze(0), (L, L), mode="nearest-exact"
        )
        gt_tensor = gt_tensor.squeeze(0).to(torch.int32)
        pred_tensor = torch.tensor(out_seg, dtype=torch.int32).unsqueeze(0)

        if save:
            save_seg(out_seg, "pred_foo.tiff") #
            save_seg(out_data, "gt_foo.tiff")

        val = jac(pred_tensor, gt_tensor)
        vals.append(val.item())
        #print(jac.compute())
        #jac.update(pred_tensor, gt_tensor)
    print(f"{prefix}: {jac.compute()} +/- {np.std(vals)}")

noise_vals = [0, 0.001, 0.01, 0.1, 0.2]
noise_params = [(True, n) for n in noise_vals]

for n in noise_params:
    for models in ["DINOv2-S-14", "random_forest"]:
        name = "dv2" if models == "DINOv2-S-14" else "rf"
        model_path = f"/home/ronan/HR-Dv2/experiments/weakly_supervised/models/{name}_real.pkl"
        main_loop(model_path, models, f"{models}, {n[1]}: ", [(255, 2), (170, 1), (85, 0)], noise_params=n, save=False)

"""
for n_data in [1, 2, 4, 8, 16]:
    for models in ["DINOv2-S-14", "random_forest"]:
        name = "dv2" if models == "DINOv2-S-14" else "rf"
        model_path = f"/home/ronan/HR-Dv2/experiments/weakly_supervised/models/{n_data}/{name}.pkl"
        main_loop(model_path, models, f"{models}, {n_data}: ", [(255, 2), (170, 1), (85, 0)])
"""
        

"""
(inc. bg as a class) with old classifiers
dv2 mIoU: 0.7915431261062622
weka mIoU: 0.3780558109283447


(inc. bg as a class) with new classifiers
dv2 mIoU: 0.8129842281341553
dv mIoU: 0.8047828674316406
weka mIoU: 0.3105025291442871

dv2 mIoU (no crf): 0.7758936882019043
dv mIoU (no crf): 0.7737023234367371
weka mIoU (no crf): 0.3527997136116028


N_data run w/out std dev
0/150
30/150
60/150
90/150
120/150
DINOv2-S-14, 1: : 0.7994146347045898
0/150
30/150
60/150
90/150
120/150
random_forest, 1: : 0.34948796033859253
Using cache found in /home/ronan/.cache/torch/hub/facebookresearch_dinov2_main
0/150
30/150
60/150
90/150
120/150
DINOv2-S-14, 2: : 0.7780811190605164
0/150
30/150
60/150
90/150
120/150
random_forest, 2: : 0.37990739941596985
Using cache found in /home/ronan/.cache/torch/hub/facebookresearch_dinov2_main
0/150
30/150
60/150
90/150
120/150
DINOv2-S-14, 4: : 0.7993969917297363
0/150
30/150
60/150
90/150
120/150
random_forest, 4: : 0.39271098375320435
Using cache found in /home/ronan/.cache/torch/hub/facebookresearch_dinov2_main
0/150
30/150
60/150
90/150
120/150
DINOv2-S-14, 8: : 0.8266112804412842
0/150
30/150
60/150
90/150
120/150
random_forest, 8: : 0.397940993309021
Using cache found in /home/ronan/.cache/torch/hub/facebookresearch_dinov2_main
0/150
30/150
60/150
90/150
120/150
DINOv2-S-14, 16: : 0.8581506013870239
0/150
30/150
60/150
90/150
120/150
random_forest, 16: : 0.40712738037109375
"""



