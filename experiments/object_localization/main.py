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
from dataset import ImageDataset, Dataset, bbox_iou, extract_gt_VOC

torch.manual_seed(2189)
np.random.seed(2189)


DIR = os.getcwd() + "/experiments/object_localization"
PATCH_SIZE = 14
PLOT_PER = 10
SAVE_PER = 10


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


def save_results(save_data: List, new: bool = False) -> None:
    mode = "w+" if new else "a"
    with open(f"{DIR}/out/results.csv", mode, newline="") as csvfile:
        writer = csv.writer(csvfile)
        if new:
            writer.writerow(
                ["Img idx", "Match", "N boxes gt", "N boxes pred", "Img path", "IoUs"]
            )
        for row in save_data:
            img_id, img_name, match, n_gt, n_pred, ious = row
            writer.writerow([img_id, match, n_gt, n_pred, img_name, *ious])


def deduplicate_superbox(pred_bboxes: np.ndarray, superbox: np.ndarray) -> np.ndarray:
    deduplicated_masks: List[np.ndarray] = [superbox]
    for pred_bbox in pred_bboxes:
        iou: float = bbox_iou(torch.Tensor(superbox), torch.Tensor(pred_bbox))  # type: ignore
        if iou > 0.8:
            pass
        else:
            deduplicated_masks.append(pred_bbox)
    return np.stack(deduplicated_masks)


def get_corloc(
    gt_bbxs: np.ndarray, pred_bboxes: np.ndarray
) -> Tuple[bool, List[bool], List[Tuple[int, float]]]:
    best_ious: List[Tuple[int, float]] = []
    matches: List[bool] = []
    corloc = False
    for pred_bbox in pred_bboxes:
        match = False
        pred_ious = []
        for gt_bbox in gt_bbxs:
            iou: float = bbox_iou(torch.Tensor(gt_bbox), torch.Tensor(pred_bbox))  # type: ignore
            pred_ious.append(iou)
            if iou >= 0.5:
                corloc = True
                match = True
        best_box_iou = (int(np.argmax(pred_ious)), float(np.amax(pred_ious)))
        best_ious.append(best_box_iou)
        matches.append(match)
    return corloc, matches, best_ious


def main() -> None:
    net = HighResDV2("dinov2_vits14_reg", 4, dtype=torch.float16)
    # net.interpolation_mode = "bicubic"
    shift_dists = [i for i in range(1, 3)]
    transforms, inv_transforms = tr.get_shift_transforms(shift_dists, "Moore")
    # transforms, inv_transforms = tr.get_flip_transforms()
    net.set_transforms(transforms, inv_transforms)
    net.cuda()
    net.eval()

    # VOC07, test     VOC12, trainval
    dataset = Dataset("VOC12", "trainval", True, tr.to_norm_tensor, DIR)
    corlocs = []
    n_boxes = []
    save_data = []

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
            n_gt_bboxes = gt_bbxs.shape[0]
            if n_gt_bboxes == 0:
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

        seg, semantic, density_map, binary = multi_object_foreground_segment(
            net, img_arr, img, 80, default_crf_params
        )
        largest_connected = largest_connected_component(binary)
        superbox = np.array(get_bbox(largest_connected, (ox, oy)))
        pred_bboxes = multi_class_bboxes(seg, (ox, oy))
        pred_bboxes = deduplicate_superbox(pred_bboxes, superbox)
        n_pred_boxes = pred_bboxes.shape[0]
        img = img.cpu()

        corloc, matches, ious = get_corloc(gt_bbxs, pred_bboxes)
        corlocs.append(corloc)
        n_boxes.append(n_pred_boxes)

        try:
            if img_idx % PLOT_PER == 0 and img_idx > 0:
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
            pass

        data = [img_idx, im_name, corloc, n_gt_bboxes, n_pred_boxes, ious]
        save_data.append(data)
        img_idx += 1
        if img_idx % SAVE_PER == 0 and img_idx > 0:
            avg_corloc = np.mean(corlocs)
            print(
                f"{img_idx} / {n_imgs}: CorLoc={avg_corloc :.4f} with {np.mean(n_boxes) :.4f} boxes"
            )
            new_file = img_idx == SAVE_PER
            save_results(save_data, new_file)
            save_data = []


if __name__ == "__main__":
    main()
