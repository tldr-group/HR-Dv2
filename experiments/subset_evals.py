# tasks:
# fg segmentation on 200 duts/cubs mix
# object detection on 200 VOC07/12
# 'semantic sgementation' on 200 coco sutff/VOC07 mix - get n_classes from gt and use that for crf, hungarian match segmentations
# visualisation of pcas

# measure relevant metrics, peak memory and time usage (B=1)

# specify with json file
import torch

torch.cuda.empty_cache()
from torch import nn
from hr_dv2 import HighResDV2
from hr_dv2 import transform as tr

from object_localization.main import loop as obj_loop
from object_segmentation.main import loop as seg_loop
from object_localization.dataset import Dataset

import numpy as np
from json import load

torch.manual_seed(10)
np.random.seed(10)


def get_net(json: dict) -> nn.Module:
    dtype = torch.float16 if json["dtype"] == "float16" else torch.float32
    net = HighResDV2(json["featurizer"], json["stride_l"], json["pca_dim"], dtype)
    if json["do_flips"] and json["do_shifts"]:
        shift_dists = [i for i in range(1, json["max_shift_px"] + 1)]
        fwd_flip, inv_flip = tr.get_flip_transforms()
        fwd_shift, inv_shift = tr.get_shift_transforms(shift_dists, "Moore")
        fwd, inv = tr.combine_transforms(fwd_shift, fwd_flip, inv_shift, inv_flip)
    elif json["do_shifts"]:
        print("no flips!")
        shift_dists = [i for i in range(1, json["max_shift_px"] + 1)]
        fwd, inv = tr.get_shift_transforms(shift_dists, "Moore")
    elif json["do_flips"]:
        fwd, inv = tr.get_flip_transforms()
    else:
        print("no transforms!")
        fwd, inv = [], []
    net.set_transforms(fwd, inv)
    net.cuda()
    net.eval()

    return net


def object_localize(net: nn.Module, n: int, json: dict, single: bool = False) -> None:
    dataset = Dataset(
        "VOC12",
        "trainval",
        False,
        tr.to_norm_tensor,
        "experiments/object_localization/",
    )
    obj_loop(net, dataset, n, json, "experiments/subsets/dv2/voc12s_230824", 20, single)


def object_segment_cub(net: nn.Module, n: int, json: dict) -> None:
    seg_loop(
        net,
        "experiments/object_segmentation/datasets/CUB_200_2011/",
        n,
        json,
        "experiments/subsets/dv2/cubs_090824/",
        5,
    )


def object_segment_dut(net: nn.Module, n: int, json: dict) -> None:
    seg_loop(
        net,
        "experiments/object_segmentation/datasets/DUTS-TE/",
        n,
        json,
        "experiments/subsets/dv2/duts_02924_no_shift/",
        10,
    )


n_cub = 5993
n_duts = 3203
n_voc7 = 10000
n_voc12 = 5823
if __name__ == "__main__":
    with open("experiments/exprs.json") as f:
        expr_json = load(f)

    # json = expr_json["dv2_strict_crf"]
    json = expr_json["dv2_no_stride_no_trs"]
    print(json)
    net = get_net(json)
    # object_segment_dut(net, n_duts, json)
    object_localize(net, n_voc7, json, True)
    # object_localize(net, 40, json)


"""

vanilla dv2, \rho_a > mean, CUB IoU = 0.7847+/-0.1361
vanilla dv2, \rho_a > mean, DUTS IoU = 0.6541+/-0.2560

VOC07 340 good example
it might be worth just doing multi with every class except min(attention_density) one,
maybe limiting n boxes to 5 for fair comparison to MOST?

vanilla dv2, \rho_a > mean, VOC07 CorLoc = 0.716 w/ 3.49 boxes
3900 / 5823: CorLoc=0.7246 with 3.3826 boxes

vanilla dv2, no tr, DUTs mIoU=0.6381+/-0.2558


"""
