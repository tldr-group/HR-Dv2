import os
from time import time
import torch
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image, ImageDraw

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
import matplotlib.patches as patches

torch.cuda.empty_cache()

from hr_dv2 import HighResDV2, tr
from hr_dv2.utils import *
from hr_dv2.segment import (
    foreground_segment,
    multi_object_foreground_segment,
    multi_class_bboxes,
    get_seg_bboxes,
    default_crf_params,
)
from dataset import ImageDataset, Dataset, bbox_iou, extract_gt_VOC

torch.manual_seed(2189)
np.random.seed(2189)


DIR = os.getcwd() + "/experiments/object_localization"
PATCH_SIZE = 14


def plot_results(
    img_arr: np.ndarray,
    seg: np.ndarray,
    gt_bboxes: np.ndarray,
    pred_bboxes: np.ndarray,
    bbox_matches: List[bool],
    offset: Tuple[int, int],
    density_map: np.ndarray,
    semantic: np.ndarray,
    idx: int,
) -> None:
    def _draw_bboxes(bbox_list, colours: List[str], ax) -> None:
        for i, bbox in enumerate(bbox_list):
            x0, y0, x1, y1 = bbox
            rect = patches.Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                linewidth=1,
                edgecolor=colours[i],
                facecolor="none",
            )
            ax.add_patch(rect)

    fig, axs = plt.subplots(nrows=2, ncols=2)
    gt_colours = ["b" for i in range(len(gt_bboxes))]
    img_ax, bbox_ax, density_ax, semantic_ax = (
        axs[0, 0],
        axs[0, 1],
        axs[1, 0],
        axs[1, 1],
    )
    img_ax.imshow(img_arr)
    _draw_bboxes(gt_bboxes, gt_colours, img_ax)

    ox, oy = offset
    print(seg.shape, img_arr.shape)
    h, w, c = seg.shape
    ih, iw, c = img_arr.shape
    dy, dx = (ih - (h + oy)), (iw - (w + ox))
    seg = np.pad(seg, ((oy, dy), (ox, dx), (0, 0)))
    seg = seg.reshape((ih, iw, 1))
    alpha_mask = np.where(seg >= 1, [1, 1, 1, 1], [0.25, 0.25, 0.25, 0.95])
    img = Image.fromarray(img_arr).convert("RGBA")
    masked = (img * alpha_mask).astype(np.uint8)
    bbox_ax.imshow(masked)
    pred_colours = ["g" if i else "r" for i in bbox_matches]
    _draw_bboxes(pred_bboxes, pred_colours, bbox_ax)

    density_ax.imshow(density_map)
    semantic_ax.imshow(semantic)

    for ax in [img_ax, bbox_ax, density_ax, semantic_ax]:
        ax.set_axis_off()
        ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(f"{DIR}/out/{idx}.png")


def get_corloc(gt_bbxs, pred_bboxes) -> Tuple[bool, List[bool], List[float]]:
    ious = []
    matches: List[bool] = []
    corloc = False
    for pred_bbox in pred_bboxes:
        match = False
        for gt_bbox in gt_bbxs:
            iou: float = bbox_iou(torch.Tensor(gt_bbox), torch.Tensor(pred_bbox))  # type: ignore
            if iou > 0.5:
                corloc = True
                match = True
            ious.append(iou)
        matches.append(match)
    return corloc, matches, ious


def main() -> None:
    net = HighResDV2("dinov2_vits14_reg", 4, dtype=torch.float16)
    # net.interpolation_mode = "bicubic"
    shift_dists = [i for i in range(1, 3)]
    transforms, inv_transforms = tr.get_shift_transforms(shift_dists, "Moore")
    # transforms, inv_transforms = tr.get_flip_transforms()
    net.set_transforms(transforms, inv_transforms)
    net.cuda()
    net.eval()

    dataset = Dataset("VOC07", "test", True, tr.to_norm_tensor, DIR)
    corlocs = []
    best_ious = []

    img_idx: int = 0
    n_imgs = len(dataset.dataloader)
    for im_id, inp in enumerate(dataset.dataloader):
        img = inp[0]

        im_name = dataset.get_image_name(inp[1])

        # pass if no image name
        if im_name is None:
            continue
        gt_bbxs, gt_cls = dataset.extract_gt(inp[1], im_name)

        if gt_bbxs is not None:
            # pass if no bbox
            if gt_bbxs.shape[0] == 0:
                continue

        c, h, w = img.shape
        sub_h: int = h % PATCH_SIZE
        sub_w: int = w % PATCH_SIZE
        oy, ox = sub_h // 2, sub_w // 2

        transform = tr.closest_crop(h, w, PATCH_SIZE, to_tensor=False)

        pil_img: Image.Image = tr.to_img(tr.unnormalize(img))
        uncropped_img_arr = np.array(pil_img)

        img = transform(img)
        img_arr = np.array(tr.to_img(tr.unnormalize(img)))
        img = img.cuda()

        """
        seg = foreground_segment(net, img_arr, img, 40, default_crf_params)
        pred_bboxes = get_seg_bboxes(seg, (ox, oy))
        """
        try:
            seg, semantic, density_map = multi_object_foreground_segment(
                net, img_arr, img, 80, default_crf_params
            )
            pred_bboxes = multi_class_bboxes(seg, (ox, oy))
            img = img.cpu()

            corloc, matches, ious = get_corloc(gt_bbxs, pred_bboxes)

            if img_idx % 10 == 0 and img_idx > 0:
                plot_results(
                    uncropped_img_arr,
                    seg,
                    gt_bbxs,
                    pred_bboxes,
                    matches,
                    (ox, oy),
                    density_map,
                    semantic,
                    img_idx,
                )
        except:
            corloc = False
            ious = [0]
        corlocs.append(corloc)
        best_ious.append(max(ious))

        img_idx += 1

        if img_idx % 10 == 0 and img_idx > 0:
            avg_corloc = np.mean(corlocs)
            avg_bIoU = np.mean(best_ious)
            std_bIoU = np.std(best_ious)
            print(
                f"{img_idx} / {n_imgs}: CorLoc={avg_corloc :.4f}, Avg Best IoU={avg_bIoU:.4f} +/- {std_bIoU}"
            )


if __name__ == "__main__":
    main()
