import os
from time import time
import torch
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from skimage.measure import label
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels
from skimage import segmentation, color
from skimage import graph

torch.cuda.empty_cache()

from hr_dv2 import HighResDV2, tr
from hr_dv2.utils import *
from dataset import ImageDataset, Dataset, bbox_iou, extract_gt_VOC


LABEL_CONFIDENCE = 0.7
SXY_G, SXY_B = (3, 3), (80, 80)
SRGB = (13, 13, 13)
COMPAT_G, COMPAT_B = 10, 10
KERNEL = dcrf.FULL_KERNEL
PATCH_SIZE = 14


def find_fg_mask(seg: np.ndarray, verbose=False) -> np.ndarray:
    # ASSUMES BG SPANS WHOLE IMAGE AND FG DOESN'T
    seg = seg.astype(bool)
    h, w, c = seg.shape
    seg_area = h * w
    masks = [seg, ~seg]
    areas = []
    bg_idx = 0
    if verbose:
        print(f"Seg area={seg_area}")
    for i, m in enumerate(masks):
        idxs = np.nonzero(m)
        min_y, max_y = np.amin(idxs[0]), np.amax(idxs[0])
        min_x, max_x = np.amin(idxs[1]), np.amax(idxs[1])
        area = (max_x - min_x) * (max_y - min_y)
        if verbose:
            print(f"{i}: area={area} ({min_x}, {min_y}, {max_x}, {max_y})")
        areas.append(area)
    bg_idx = int(np.argmax(areas))
    return masks[1 - bg_idx]


def object_segment(
    net: HighResDV2,
    img_arr: np.ndarray,
    img_tensor: torch.Tensor,
    single_component: bool = True,
    verbose: bool = False,
    crop_coords: Tuple[int, int] = (0, 0),
) -> np.ndarray:
    start = time()
    ih, iw, ic = img_arr.shape
    hr_tensor, _ = net.forward(img_tensor, attn="none")
    # hr_tensor = TF.crop(hr_tensor, crop_coords[0], crop_coords[1], ih, iw)
    features: np.ndarray
    b, c, fh, fw = hr_tensor.shape
    features = tr.to_numpy(hr_tensor)
    reshaped = features.reshape((c, fh * fw)).T

    normed = normalise_pca(reshaped)
    cluster = KMeans(n_clusters=2, n_init="auto")
    cluster.fit(normed)
    k_means_labels = cluster.labels_

    unary = unary_from_labels(k_means_labels, 2, LABEL_CONFIDENCE, zero_unsure=False)
    d = dcrf.DenseCRF2D(iw, ih, 2)
    u = np.ascontiguousarray(unary)
    d.setUnaryEnergy(u)
    d.addPairwiseGaussian(
        sxy=SXY_G,
        compat=COMPAT_G,
        kernel=KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC,
    )
    d.addPairwiseBilateral(
        sxy=SXY_B,
        srgb=SRGB,
        rgbim=img_arr,
        compat=COMPAT_B,
        kernel=KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC,
    )
    Q = d.inference(10)
    crf_seg = np.argmax(Q, axis=0)
    crf_seg = crf_seg.reshape((ih, iw, 1))
    fg_seg = find_fg_mask(crf_seg, verbose=verbose)
    fg_seg = fg_seg.astype(np.uint8)
    if single_component:
        labels, n_components = label(fg_seg, background=0, return_num=True)
        fg_seg = labels == np.argmax(np.bincount(labels.flat, weights=fg_seg.flat))
    end = time()
    if verbose:
        print(f"Finished in {end-start}s")
    return fg_seg


def do_pca(
    net: HighResDV2,
    img_tensor: torch.Tensor,
    rescale: bool = True,
    reshape: bool = True,
) -> np.ndarray:
    hr_tensor, _ = net.forward(img_tensor, attn="none")
    # hr_tensor = TF.crop(hr_tensor, crop_coords[0], crop_coords[1], ih, iw)
    features: np.ndarray
    b, c, fh, fw = hr_tensor.shape
    features = tr.to_numpy(hr_tensor)
    reshaped = features.reshape((c, fh * fw)).T

    standard = standardize_img(reshaped)
    pca = PCA(
        n_components=3, svd_solver="randomized", n_oversamples=5, iterated_power=3
    )
    data = pca.fit_transform(standard)
    if rescale:
        data = rescale_pca(data)
    if reshape:
        data = data.reshape((fh, fw, 3))
    return data


def RAG(
    net: HighResDV2,
    img_arr: np.ndarray,
    img_tensor: torch.Tensor,
    k: int = 3,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    This seems to be working relatively well. Pipeline is:
    1) HRDv2 for features
    2) PCA down to 3 components
    3) SLIC superpixel segmentation (i.e k-means in colour space)
    could just k-means?
    => n class segmentation
    4) CRF to improve segmentation (using labels rather than confidence)

    Could really replace 3 with anything that clusters the features and considers spatial
    effects. Need to try varying PCA n_dim, SLIC parameters, CRF parameters

    :param net: _description_
    :type net: HighResDV2
    :param img_arr: _description_
    :type img_arr: np.ndarray
    :param img_tensor: _description_
    :type img_tensor: torch.Tensor
    :param k: _description_, defaults to 3
    :type k: int, optional
    :param verbose: _description_, defaults to False
    :type verbose: bool, optional
    :return: _description_
    :rtype: Tuple[np.ndarray, np.ndarray]
    """

    def _weight_mean_color(graph, src, dst, n):
        diff = graph.nodes[dst]["mean color"] - graph.nodes[n]["mean color"]
        diff = np.linalg.norm(diff)
        return {"weight": diff}

    def merge_mean_color(graph, src, dst):
        graph.nodes[dst]["total color"] += graph.nodes[src]["total color"]
        graph.nodes[dst]["pixel count"] += graph.nodes[src]["pixel count"]
        graph.nodes[dst]["mean color"] = (
            graph.nodes[dst]["total color"] / graph.nodes[dst]["pixel count"]
        )

    start = time()
    ih, iw, ic = img_arr.shape
    hr_tensor, _ = net.forward(img_tensor, attn="none")
    features: np.ndarray
    b, c, fh, fw = hr_tensor.shape
    features = tr.to_numpy(hr_tensor)
    reshaped = features.reshape((c, fh * fw)).T

    standard = standardize_img(reshaped)
    pca = PCA(
        n_components=k, svd_solver="randomized", n_oversamples=5, iterated_power=3
    )
    pca.fit(standard)
    standard = pca.transform(standard)
    rescaled = rescale_pca(standard).reshape((ih, iw, k))
    rescaled = (255 * rescaled).astype(np.uint8)

    labels = segmentation.slic(
        rescaled,
        compactness=30,
        n_segments=20,
        start_label=0,
        convert2lab=True,
        sigma=1,
        enforce_connectivity=False,
    )
    g = graph.rag_mean_color(rescaled, labels)

    labels2 = graph.merge_hierarchical(
        labels,
        g,
        thresh=110,
        rag_copy=False,
        in_place_merge=True,
        merge_func=merge_mean_color,
        weight_func=_weight_mean_color,
    )

    n_classes = np.amax(labels2) + 1
    unary = unary_from_labels(labels2, n_classes, LABEL_CONFIDENCE, zero_unsure=False)
    d = dcrf.DenseCRF2D(iw, ih, n_classes)
    u = np.ascontiguousarray(unary)
    d.setUnaryEnergy(u)
    d.addPairwiseGaussian(
        sxy=SXY_G,
        compat=COMPAT_G,
        kernel=KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC,
    )
    d.addPairwiseBilateral(
        sxy=SXY_B,
        srgb=SRGB,
        rgbim=img_arr,
        compat=COMPAT_B,
        kernel=KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC,
    )
    Q = d.inference(10)
    crf_seg = np.argmax(Q, axis=0)
    crf_seg = crf_seg.reshape((ih, iw, 1))
    end = time()
    if verbose:
        print(f"Finished in {end-start}s")
    return rescaled, crf_seg


DIR = os.getcwd() + "/experiments/object_localization"


def main() -> None:
    net = HighResDV2("dinov2_vits14_reg", 4, dtype=torch.float16)
    shift_dists = [i for i in range(1, 3)]
    # transforms, inv_transforms = tr.get_shift_transforms(shift_dists, "Moore")
    transforms, inv_transforms = tr.get_flip_transforms()
    net.set_transforms(transforms, inv_transforms)
    net.cuda()
    net.eval()

    dataset = Dataset("VOC07", "test", True, tr.to_norm_tensor, DIR)

    img_idx: int = 0
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
        add_h: int = PATCH_SIZE - (h % PATCH_SIZE)
        add_w: int = PATCH_SIZE - (w % PATCH_SIZE)

        transform = tr.closest_crop(
            h, w, 14, to_tensor=False
        )  # tr.closest_pad(h, w, 4)

        img = transform(img)
        pil_img: Image.Image = tr.to_img(tr.unnormalize(img))
        img_arr = np.array(pil_img)
        new_h, new_w, c = img_arr.shape

        img = img.cuda()

        pca, seg = RAG(net, img_arr, img, verbose=False)

        reshaped_seg = seg.reshape((new_h, new_w)) + 1
        recolored_seg = color.label2rgb(reshaped_seg, img_arr, kind="avg")

        stacked = np.hstack((img_arr, pca, recolored_seg))
        pil_tri_img = Image.fromarray(stacked)
        pil_tri_img.save(f"{DIR}/out/{img_idx}_processed.png")
        img = img.cpu()

        """
        feature_map = do_pca(net, img)
        pca_img_arr = (255 * feature_map).astype(np.uint8)
        pca_pil_img = Image.fromarray(pca_img_arr)
        pca_pil_img.save(f"{DIR}/out/{img_idx}_pca.png")

        
        seg = object_segment(net, img_arr, img, False, (add_h, add_w))
        grey = (seg * 255).squeeze(-1).astype(np.uint8)

        seg_img = Image.fromarray(grey, "L")
        seg_img.save(f"{DIR}/out/{img_idx}_seg.png")
        """

        img_idx += 1
        if img_idx > 300:
            break


if __name__ == "__main__":
    main()

# a few problems: seems to be detecting foreground when sometimes image sits
# between foreground and background. also the 'largest area = background'
# heurisitic is not robust enough. may need to cluster/decompose into
# more than 2 clusters. Maybe PCA down to three and cluster 1 component? otsu on red channel?
# maybe detect multiple objects?
