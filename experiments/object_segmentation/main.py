import os
import torch
from PIL import Image
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
import matplotlib.patches as patches
import csv

torch.cuda.empty_cache()

from hr_dv2 import HighResDV2, tr
from hr_dv2.utils import *
from hr_dv2.segment import (
    foreground_segment,
    multi_object_foreground_segment,
    multi_class_bboxes,
    get_seg_bboxes,
    largest_connected_component,
    get_bbox,
    default_crf_params,
)

torch.manual_seed(2189)
np.random.seed(2189)

CWD = os.getcwd()
DIR = f"{CWD}/experiments/object_segmentation/"
CUB = f"{CWD}/experiments/object_segmentation/datasets/CUB_200_2011/"
IMGS = f"{CUB}/test_images/"
SEGS = f"{CUB}/test_segmentations/"

PATCH_SIZE = 14
PLOT_PER = 10
SAVE_PER = 5
N_IMGS = 5994


def compute_iou(pred, target):
    # From https://github.com/lukemelas/deep-spectral-segmentation/blob/main/object-segmentation/metrics.py
    pred, target = pred.to(torch.bool), target.to(torch.bool)
    intersection = torch.sum(pred * (pred == target), dim=[-1, -2]).squeeze()
    union = torch.sum(pred + target, dim=[-1, -2]).squeeze()
    iou = (intersection.to(torch.float) / union).mean()
    iou = (
        iou.item() if (iou == iou) else 0
    )  # deal with nans, i.e. torch.nan_to_num(iou, nan=0.0)
    return iou


def plot_result(
    pred_seg: np.ndarray, gt_seg: np.ndarray, img_arr: np.ndarray, idx: int, iou: float
) -> None:
    fig, axs = plt.subplots(nrows=1, ncols=2)
    img_ax, seg_ax = axs
    h, w, c = img_arr.shape

    alpha_seg = pred_seg.reshape((h, w, 1))
    alpha_mask = np.where(alpha_seg >= 1, [1, 1, 1, 1], [0.25, 0.25, 0.25, 0.95])
    img = Image.fromarray(img_arr).convert("RGBA")
    masked = (img * alpha_mask).astype(np.uint8)
    img_ax.imshow(masked)

    tp = np.where(pred_seg == gt_seg, 1, 0) * np.where(gt_seg == 1, 1, 0)
    tn = np.where(pred_seg == gt_seg, 1, 0) * np.where(gt_seg == 0, 1, 0)

    fp = np.where(pred_seg == 1, 1, 0) * np.where(gt_seg == 0, 1, 0)
    fn = np.where(pred_seg == 0, 1, 0) * np.where(gt_seg == 1, 1, 0)

    colours: List[List[float]] = [
        [1, 1, 1],
        [0, 0, 0],
        [235 / 255, 52 / 255, 225 / 255],
        [83 / 255, 205 / 255, 230 / 255],
    ]
    out_seg = np.zeros((h, w, c), dtype=np.uint8)
    for colour, arr in zip(colours, [tp, tn, fp, fn]):
        arr = arr.reshape((h, w, 1))
        adjusted_colour = 255 * np.array(colour)
        out_seg = (np.where(arr == 1, adjusted_colour, out_seg)).astype(np.uint8)
    seg_ax.imshow(out_seg)
    for ax in [img_ax, seg_ax]:
        ax.set_axis_off()
    plt.suptitle(f"{idx} mIoU: {iou :.4f}")
    plt.tight_layout()
    plt.savefig(f"{DIR}/out/{idx}.png")


def save_result(save_data: List, new: bool = False) -> None:
    with open(f"{DIR}/out/results.csv", "w+", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if new:
            writer.writerow(["Img idx", "Img path", "mIoU", "Dataset mIoU"])
        for row in save_data:
            img_id, img_name, miou, d_miou = row
            writer.writerow([img_id, img_name, miou, d_miou])


def main():
    net = HighResDV2("dinov2_vits14_reg", 4, dtype=torch.float16)
    # net.interpolation_mode = "bicubic"
    shift_dists = [i for i in range(1, 3)]
    transforms, inv_transforms = tr.get_shift_transforms(shift_dists, "Moore")
    # transforms, inv_transforms = tr.get_flip_transforms()
    net.set_transforms(transforms, inv_transforms)
    net.cuda()
    net.eval()

    ious = []
    save_data = []

    for i in range(N_IMGS):
        img_path = f"{IMGS}{i}.jpg"
        seg_path = f"{SEGS}{i}.png"

        pil_img = Image.open(img_path)
        pil_seg = Image.open(seg_path).convert("L")

        ih, iw = pil_img.height, pil_img.width
        img_tr = tr.closest_crop(ih, iw, PATCH_SIZE, True)
        seg_tr = tr.closest_crop(ih, iw, PATCH_SIZE, False)

        img, crop_pil_img = tr.load_image(img_path, img_tr)
        img_arr = np.array(crop_pil_img)
        seg_tensor = tr.to_tensor(seg_tr(pil_seg))
        seg_arr = tr.to_numpy(seg_tensor)

        img = img.cuda()
        # maybe just use attention ,map as unary and not unary from labels
        refined, density_map, unrefined = foreground_segment(
            net, img_arr, img, 80, default_crf_params
        )
        img = img.cpu()
        refined = np.squeeze(refined, -1)
        refined = largest_connected_component(refined)

        # same as in DSS for comparison
        gt = np.where(seg_arr > 0.5, 1, 0)  # np.zeros_like(refined) +

        iou = compute_iou(torch.Tensor(refined), torch.Tensor(gt))
        ious.append(iou)
        save_data.append([i, img_path, iou, np.mean(ious)])

        if i % PLOT_PER == 0:
            plot_result(refined, gt, img_arr, i, iou)
        if i % SAVE_PER == 0 and i > 0:
            new = i == SAVE_PER
            print(f"{i}/{N_IMGS}: mIoU={np.mean(ious):.4f}+/-{np.std(ious):.4f}")
            save_result(save_data, new)


if __name__ == "__main__":
    main()
