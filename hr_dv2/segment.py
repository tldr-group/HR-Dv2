import torch

import numpy as np
from time import time
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels
from skimage.measure import label


from .high_res import HighResDV2
from . import transform as tr
from .utils import *

from dataclasses import dataclass
from typing import Tuple

SMALL_OBJECT_AREA_CUTOFF = 50


def get_dv2_features(
    net: HighResDV2,
    tensor: torch.Tensor,
    flatten: bool = True,
    sequential: bool = False,
) -> np.ndarray:
    """Given a HR-DV2 net with set transforms, get features. Either
    # Use for high memory (large image and/or n_transforms) situations.

    :param net: net to get features from
    :type net: HighResDV2
    :param tensor: image tensor, (C, H, W)
    :type tensor: torch.Tensor
    :param flatten: flatten output feature array, defaults to True
    :type flatten: bool, optional
    :param sequential: use for high memory (large image and/or n_transforms) situations, defaults to False
    :type sequential: bool, optional
    :return: np array, either (c, h, w) or (h * w, c) depending on flatten.
    :rtype: np.ndarray
    """

    if sequential:
        hr_tensor, _ = net.forward_sequential(tensor, attn="none")
    else:
        hr_tensor, _ = net.forward(tensor, attn="none")
    features: np.ndarray
    b, c, fh, fw = hr_tensor.shape
    features = tr.to_numpy(hr_tensor)
    out: np.ndarray = features
    if flatten:
        out = features.reshape((c, fh * fw)).T
    return out


def do_pca(features: np.ndarray, k: int = 3, standardize: bool = True) -> np.ndarray:
    """k-component dimensionality reduction of (n_samples, n_channels) features.

    :param features: np array, (n_samples, n_channels)
    :type features: np.ndarray
    :param k: number of PCA components, defaults to 3
    :type k: int, optional
    :param standardize: whether to standardize data before PCA, defaults to True
    :type standardize: bool, optional
    :return: k-component PCA of features, (n_samples, k)
    :rtype: np.ndarray
    """

    if standardize:
        features = standardize_img(features)
    pca = PCA(
        n_components=k, svd_solver="randomized", n_oversamples=5, iterated_power=3
    )
    pca.fit(features)
    projection = pca.transform(features)
    return projection


@dataclass
class CRFParams:
    label_confidence: float = 0.6
    sxy_g: Tuple[int, int] = (3, 3)
    sxy_b: Tuple[int, int] = (80, 80)
    s_rgb: Tuple[int, int, int] = (13, 13, 13)
    compat_g: float = 10
    compat_b: float = 10
    n_infer: int = 10


KERNEL = dcrf.FULL_KERNEL
default_crf_params = CRFParams()


def do_crf_from_labels(
    labels_arr: np.ndarray, img_arr: np.ndarray, n_classes: int, crf: CRFParams
) -> np.ndarray:
    h, w, c = img_arr.shape
    unary = unary_from_labels(
        labels_arr, n_classes, crf.label_confidence, zero_unsure=False
    )
    d = dcrf.DenseCRF2D(w, h, n_classes)
    u = np.ascontiguousarray(unary)
    d.setUnaryEnergy(u)
    d.addPairwiseGaussian(
        sxy=crf.sxy_g,
        compat=crf.compat_g,
        kernel=KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC,
    )
    d.addPairwiseBilateral(
        sxy=crf.sxy_b,
        srgb=crf.s_rgb,
        rgbim=img_arr,
        compat=crf.compat_b,
        kernel=KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC,
    )
    Q = d.inference(crf.n_infer)
    crf_seg = np.argmax(Q, axis=0)
    crf_seg = crf_seg.reshape((h, w, 1))
    return crf_seg


def cluster(
    net: HighResDV2,
    img_arr: np.ndarray,
    img_tensor: torch.Tensor,
    clusters: List[int],
    get_attn: bool = True,
    verbose: bool = False,
) -> Tuple[List[np.ndarray], np.ndarray]:
    start = time()
    attn: np.ndarray
    if get_attn:
        # Cache old transforms
        fwd, inv = net.transforms, net.inverse_transforms
        net.set_transforms([], [])
        attn_tensor, _ = net.forward(img_tensor, attn="cls")
        attn = tr.to_numpy(attn_tensor)
        # Add them back
        net.set_transforms(fwd, inv)
    else:
        attn = np.zeros_like(img_arr)
    print(attn.shape)
    hr_tensor, _ = net.forward(img_tensor, attn="none")
    features: np.ndarray
    b, c, fh, fw = hr_tensor.shape
    features = tr.to_numpy(hr_tensor)
    reshaped = features.reshape((c, fh * fw)).T

    normed = normalise_pca(reshaped)

    labels = []
    for n in clusters:
        cluster = KMeans(n_clusters=n, n_init="auto", max_iter=300)
        label = cluster.fit_predict(normed)
        # refined_seg = do_crf_from_labels(label, img_arr, n, default_crf_params)
        refined_seg = label
        labels.append(refined_seg)

    end = time()
    if verbose:
        print(f"Finished in {end-start}s")
    return labels, attn


def get_attn_density(
    labels_arr: np.ndarray, attn: np.ndarray
) -> Tuple[np.ndarray, List[float]]:
    densities = []
    attention_density_map = np.zeros_like(labels_arr).astype(np.float64)
    n_clusters = np.amax(labels_arr)
    for n in range(n_clusters):
        binary_mask = np.where(labels_arr == n, 1, 0)
        n_pix = np.sum(binary_mask)
        cluster_attn = np.sum(attn * binary_mask)
        cluster_attn_density = cluster_attn / n_pix
        densities.append(cluster_attn_density)
        attention_density_map += cluster_attn_density * binary_mask
    return attention_density_map, densities


def get_attn_cutoff(densities: np.ndarray | List[float], offset: int = 2) -> float:
    n, bins = np.histogram(densities, bins=10)
    max_loc = int(np.argmax(n))
    cutoff = bins[max_loc + offset]
    return cutoff


def foreground_segment(
    net: HighResDV2,
    img_arr: np.ndarray,
    img_tensor: torch.Tensor,
    clusters: List[int],
    crf_params: CRFParams,
) -> np.ndarray:
    h, w, c = img_arr.shape
    segs, attn = cluster(net, img_arr, img_tensor, clusters, True, False)
    seg = segs[0].reshape((h, w))
    sum_cls = np.sum(attn, axis=0)
    density_map, densities = get_attn_density(seg, sum_cls)
    cutoff = get_attn_cutoff(densities, 2)
    fg_seg = (density_map > cutoff).astype(np.uint8)
    refined = do_crf_from_labels(fg_seg, img_arr, 2, crf_params)
    # if crf fails, fall back on (thresholded) attn denisty map
    if np.sum(refined) < SMALL_OBJECT_AREA_CUTOFF:
        refined = fg_seg
    return refined


def get_bbox(arr: np.ndarray, offsets: Tuple[int, int] = (0, 0)) -> List[int]:
    idxs = np.nonzero(arr)
    y_min, y_max = np.amin(idxs[0]), np.amax(idxs[0])
    x_min, x_max = np.amin(idxs[1]), np.amax(idxs[1])

    ox, oy = offsets
    x0, y0 = int(x_min + ox), int(y_min + oy)
    x1, y1 = int(x_max + ox), int(y_max + oy)
    return [x0, y0, x1, y1]


def get_seg_bboxes(fg_seg: np.ndarray, offsets: Tuple[int, int] = (0, 0)) -> np.ndarray:
    bboxes: List[List[int]] = []
    separated, n_components = label(fg_seg, return_num=True)  # type: ignore
    for i in range(1, n_components + 1):
        current_obj = np.where(separated == i, 1, 0).astype(np.uint8)
        if np.sum(current_obj) > SMALL_OBJECT_AREA_CUTOFF:
            obj_bbox = get_bbox(current_obj, offsets)
            bboxes.append(obj_bbox)
    bboxes_arr = np.array(bboxes)
    return bboxes_arr
