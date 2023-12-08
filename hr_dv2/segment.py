import torch

import numpy as np
from time import time
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels
from skimage.measure import label


from .high_res import HighResDV2
from . import transform as tr
from .utils import *

from dataclasses import dataclass
from typing import Tuple

SMALL_OBJECT_AREA_CUTOFF = 200


def get_dv2_features(
    net: HighResDV2,
    tensor: torch.Tensor,
    flatten: bool = True,
    sequential: bool = False,
) -> np.ndarray:
    """Given a HR-DV2 net with set transforms, get features.

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
    n_clusters: int,
    get_attn: bool = True,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get and (over) cluster ViT features and optionally return attention.

    :param net: ViT to extract features from
    :type net: HighResDV2
    :param img_arr: original img arr, shape (h, w, 3)
    :type img_arr: np.ndarray
    :param img_tensor: tensor of img, shape (3, h, w)
    :type img_tensor: torch.Tensor
    :param n_clusters: number of k-means clusters
    :type n_clusters: int
    :param get_attn: get attention map of ViT, defaults to True
    :type get_attn: bool, optional
    :param verbose: whether to print timings, defaults to False
    :type verbose: bool, optional
    :return: cluster assignment arr, attn map, cluster centroids
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
    """
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
    hr_tensor, _ = net.forward(img_tensor, attn="none")
    features: np.ndarray
    b, c, fh, fw = hr_tensor.shape
    features = tr.to_numpy(hr_tensor)
    reshaped = features.reshape((c, fh * fw)).T

    normed = normalise_pca(reshaped)

    cluster = KMeans(n_clusters=n_clusters, n_init="auto", max_iter=300)
    labels = cluster.fit_predict(normed)

    end = time()
    if verbose:
        print(f"Finished in {end-start}s")
    return labels, attn, cluster.cluster_centers_


def get_attn_density(
    labels_arr: np.ndarray, attn: np.ndarray
) -> Tuple[np.ndarray, List[float]]:
    """For each cluster, get attention per unit area and return a density arr
    where each entry is the pixel's cluster attention density.

    :param labels_arr: arr shape (h, w) where each entry is the cluster index
    :type labels_arr: np.ndarray
    :param attn: arr shape (h, w) of the attention (usually sum(CLS))
    :type attn: np.ndarray
    :return: attention density map and list of all cluster densities
    :rtype: Tuple[np.ndarray, List[float]]
    """
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


def mag(vec: np.ndarray) -> np.ndarray:
    return np.sqrt(np.dot(vec, vec))


def l2(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum((v1 - v2) * (v1 - v2)))


def get_feature_similarities(
    fg_clusters: np.ndarray, bg_clusters: np.ndarray
) -> Tuple[List[float], List[float]]:
    """Get list of all fg <-> bg and fg <-> fg similarities for merging later. O(n^2).

    :param fg_clusters: all foreground cluster centres
    :type fg_clusters: np.ndarray
    :param bg_clusters: all background cluster centres
    :type bg_clusters: np.ndarray
    :return: fg <-> bg similarities, fg <-> fg similarities
    :rtype: Tuple[List[float], List[float]]
    """
    fg_bg_similarities: List[float] = []
    for i, c1 in enumerate(fg_clusters):
        for j, c2 in enumerate(bg_clusters):
            similarity = float(l2(c1, c2))
            fg_bg_similarities.append(similarity)

    fg_fg_similarities: List[float] = []
    for i, c1 in enumerate(fg_clusters):
        for j, c2 in enumerate(fg_clusters):
            if i == j:
                pass
            else:
                similarity = float(l2(c1, c2))
                fg_fg_similarities.append(similarity)
    return fg_bg_similarities, fg_fg_similarities


def get_similarity_cutoff(fg_bg_similarities: List[float]) -> float:
    """Get similarity cutoff/distance threshold from max of foreground/
    background distribution.

    :param fg_bg_similarities: flat list of similarites of all foreground and background clusters.
    :type fg_bg_similarities: List[float]
    :return: similarity cutoff
    :rtype: float
    """
    bins, edges = np.histogram(fg_bg_similarities, bins=20)
    similarity_cutoff = edges[np.argmax(bins)]  # + edges[np.argmax(bins) + 1]) / 2
    return similarity_cutoff


def merge_foreground_clusters(
    fg_clusters: np.ndarray, distance_cutoff: float, offset: int = 1
) -> np.ndarray:
    """Merge foreground clusters with agglomerative clustering based on
    a similarity threshold (determined from fg/bg distribution).

    :param fg_clusters: foreground cluster centres
    :type fg_clusters: np.ndarray
    :param distance_cutoff: threshold above which clusters not merged
    :type distance_cutoff: float
    :param offset: value to add to returned classes, defaults to 1
    :type offset: int, optional
    :return: class predictions for each foreground cluster
    :rtype: np.ndarray
    """
    # distance_cutoff = 1 - similarity_cutoff
    cluster = AgglomerativeClustering(
        n_clusters=None,
        metric="euclidean",
        linkage="complete",
        distance_threshold=distance_cutoff,
    )
    fg_clustered = cluster.fit_predict(fg_clusters) + offset
    return fg_clustered


def split_foreground_and_refine(
    fg_mask: np.ndarray,
    fg_clustered: np.ndarray,
    clustered_arr: np.ndarray,
    img_arr: np.ndarray,
    crf_params: CRFParams,
) -> Tuple[np.ndarray, np.ndarray]:
    """Replace foreground clusters in clustered_arr with their
    merged values and refine it with a crf.

    :param fg_mask: arr of which clusters are foreground clusters
    :type fg_mask: np.ndarray
    :param fg_clustered: merged foreground cluster class assignments
    :type fg_clustered: np.ndarray
    :param clustered_arr: (h, w) arr where each entry is its cluster
    :type clustered_arr: np.ndarray
    :param img_arr: original img arr, shape (h, w, 3)
    :type img_arr: np.ndarray
    :param crf_params: set of parameters for segmentation refinement
    :type crf_params: CRFParams
    :return: refined multiclass foreground segmentation, unrefined multiclass foreground segmentation
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    out = np.zeros_like(clustered_arr)
    for i, val in enumerate(fg_mask):
        current_obj = np.where(clustered_arr == val, fg_clustered[i], 0)
        out += current_obj
    refined = do_crf_from_labels(out, img_arr, np.amax(fg_clustered) + 1, crf_params)
    return refined, out


def foreground_segment(
    net: HighResDV2,
    img_arr: np.ndarray,
    img_tensor: torch.Tensor,
    n_cluster: int,
    crf_params: CRFParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Foreground segmentation using attention density.

    :param net: ViT to extract features from
    :type net: HighResDV2
    :param img_arr: array of img, shape (h, w, 3)
    :type img_arr: np.ndarray
    :param img_tensor: tensor of img, shape (3, h, w)
    :type img_tensor: torch.Tensor
    :param n_cluster: number of k-means clusters
    :type n_cluster: int
    :param crf_params: set of parameters for segmentation refinement
    :type crf_params: CRFParams
    :return: refined merged foreground clusters,
            attn density map, unrefined merged foreground clusters
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    h, w, c = img_arr.shape
    seg, attn, _ = cluster(net, img_arr, img_tensor, n_cluster, True, False)
    seg = seg.reshape((h, w))
    sum_cls = np.sum(attn, axis=0)
    density_map, densities = get_attn_density(seg, sum_cls)
    cutoff = get_attn_cutoff(densities, 2)
    fg_seg = (density_map > cutoff).astype(np.uint8)
    refined, unrefined = do_crf_from_labels(fg_seg, img_arr, 2, crf_params)
    # if crf fails, fall back on (thresholded) attn denisty map
    if np.sum(refined) < SMALL_OBJECT_AREA_CUTOFF:
        refined = fg_seg
    return refined, density_map, unrefined


def multi_object_foreground_segment(
    net: HighResDV2,
    img_arr: np.ndarray,
    img_tensor: torch.Tensor,
    n_cluster: int,
    crf_params: CRFParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Clustering based multi-object localization:

    1) get high res features and attn from ViT
    2) over cluster to K=n_cluster clusters
    3) find foreground clusters using attn density
    4) find semantic threshold using foreground vs background similarities
    5) merge foreground clusters using semantic threshold
    6) refine merged clusters into a (h, w) multiclass segmentation

    :param net: ViT to extract features from
    :type net: HighResDV2
    :param img_arr: array of img, shape (h, w, 3)
    :type img_arr: np.ndarray
    :param img_tensor: tensor of img, shape (3, h, w)
    :type img_tensor: torch.Tensor
    :param n_cluster: number of k-means clusters
    :type n_cluster: int
    :param crf_params: set of parameters for segmentation refinement
    :type crf_params: CRFParams
    :return: refined merged foreground clusters, unrefined merged foreground clusters,
            attn density map, binary foreground segmentation
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    """
    h, w, c = img_arr.shape
    seg, attn, centers = cluster(net, img_arr, img_tensor, n_cluster, True, False)
    seg = seg.reshape((h, w))
    sum_cls = np.sum(attn, axis=0)
    density_map, densities = get_attn_density(seg, sum_cls)
    densities_arr = np.array(densities)

    fg_mask = np.nonzero(densities_arr > np.mean(densities_arr))[0]
    bg_mask = np.nonzero(densities_arr < np.mean(densities_arr))[0]

    fg_clusters = centers[fg_mask]
    bg_clusters = centers[bg_mask]

    fg_bg_sims, fg_fg_sims = get_feature_similarities(fg_clusters, bg_clusters)
    sim_cutoff = get_similarity_cutoff(fg_bg_sims)
    fg_clustered = merge_foreground_clusters(fg_clusters, sim_cutoff)
    refined, unrefined = split_foreground_and_refine(
        fg_mask, fg_clustered, seg, img_arr, crf_params
    )
    if np.sum(refined) < SMALL_OBJECT_AREA_CUTOFF:
        refined = unrefined
    binary = np.where(refined > 0, 1, 0)
    return refined, unrefined, density_map, binary


def get_bbox(arr: np.ndarray, offsets: Tuple[int, int] = (0, 0)) -> List[int]:
    """Get bbox of binary arr by looking at min/max x/y.

    :param arr: binary array shape (h, w)
    :type arr: np.ndarray
    :param offsets: bbox offsets (if img cropped), defaults to (0, 0)
    :type offsets: Tuple[int, int], optional
    :return: bbox in form x0 y0 x1 y1
    :rtype: List[int]
    """
    idxs = np.nonzero(arr)
    y_min, y_max = np.amin(idxs[0]), np.amax(idxs[0])
    x_min, x_max = np.amin(idxs[1]), np.amax(idxs[1])

    ox, oy = offsets
    x0, y0 = int(x_min + ox), int(y_min + oy)
    x1, y1 = int(x_max + ox), int(y_max + oy)
    return [x0, y0, x1, y1]


def get_seg_bboxes(fg_seg: np.ndarray, offsets: Tuple[int, int] = (0, 0)) -> np.ndarray:
    """Get bbox around each isolated component of binary arr.

    :param fg_seg: binary arr shape (h, w)
    :type fg_seg: np.ndarray
    :param offsets: bbox offsets (if img cropped), defaults to (0, 0)
    :type offsets: Tuple[int, int], optional
    :return: bboxes for each isolated component shape (n_bbox, 4)
    :rtype: np.ndarray
    """
    bboxes: List[List[int]] = []
    separated, n_components = label(fg_seg, return_num=True)  # type: ignore
    for i in range(1, n_components + 1):
        current_obj = np.where(separated == i, 1, 0).astype(np.uint8)
        if np.sum(current_obj) > SMALL_OBJECT_AREA_CUTOFF:
            obj_bbox = get_bbox(current_obj, offsets)
            bboxes.append(obj_bbox)
    bboxes_arr = np.array(bboxes)
    return bboxes_arr


def multi_class_bboxes(
    multi_seg: np.ndarray, offsets: Tuple[int, int] = (0, 0)
) -> np.ndarray:
    """For a multiclass arr with background 0, get bboxes for each class.

    :param multi_seg: multiclass arr shape (h, w)
    :type multi_seg: np.ndarray
    :param offsets: bbox offsets (if img cropped), defaults to (0, 0)
    :type offsets: Tuple[int, int], optional
    :return: arr of bboxes, shape (n_bbox, 4)
    :rtype: np.ndarray
    """
    n_classes = int(np.amax(multi_seg))
    bbox_arrs: List[np.ndarray] = []
    for class_val in range(1, n_classes + 1):
        binary = np.where(multi_seg == class_val, 1, 0)
        bbox_arr = get_seg_bboxes(binary, offsets)
        if bbox_arr.shape[0] == 0:
            pass
        else:
            bbox_arrs.append(bbox_arr)
    if len(bbox_arrs) == 0:
        bbox_arrs.append(np.zeros((1, 4)))
    return np.concatenate(bbox_arrs)


def largest_connected_component(arr: np.ndarray) -> np.ndarray:
    """Get largest connected component of binary array via skimage label.

    :param arr: binary arr shape (h, w)
    :type arr: np.ndarray
    :return: binary arr shape (h, w)
    :rtype: np.ndarray
    """
    separated, n_components = label(arr, return_num=True)  # type: ignore
    sizes: List[int] = []
    for i in range(1, n_components + 1):
        current_obj = np.where(separated == i, 1, 0).astype(np.uint8)
        n_pixels = int(np.sum(current_obj))
        sizes.append(n_pixels)
    largest_class = np.argmax(sizes) + 1
    return np.where(separated == largest_class, 1, 0)


# TODO: comment this file
# TODO: delete old notebooks
# TODO: merge onto main
