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

# TODO: add poisson noise=shotnoise (and also gaussian=thermal?) and show 0-shot perf as noise increases?


mapping = [(255, 0), (170, 1), (85, 2)]


def tiff_to_labels(tiff: np.ndarray) -> np.ndarray:
    out = tiff
    for val, i in mapping:
        out = np.where(out == val, i, out)
    return out


def save_seg(seg: np.ndarray, fname: str):
    rescaled = seg
    for i in range(3):
        rescaled = np.where(rescaled == i, mapping[i][0], rescaled)

    imwrite(fname, rescaled.astype(np.uint8), photometric="minisblack")


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


model = get_featuriser_classifier(model_name, Queue(2), Queue(2))
model.load_model(model_path)

model.do_crf = True

jac = JaccardIndex(num_classes=3, task="multiclass")
expr_folder = "/home/ronan/HR-Dv2/experiments/weakly_supervised"

for i, f in enumerate(listdir(join(expr_folder, "data"))):
    print(f"{i}/150")
    inp_path = join(expr_folder, "data", f)
    out_path = join(expr_folder, "masks", f"{f[:-4]}.tif_segmentation.tifnomalized.tif")

    inp_data = imread(inp_path)
    out_data = tiff_to_labels(imread(out_path))

    inp_img = Image.fromarray(inp_data).convert("RGB")
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

    save_seg(out_seg, join(expr_folder, out_folder, f))

    jac.update(pred_tensor, gt_tensor)
print(f"{jac.compute()}")


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

"""
