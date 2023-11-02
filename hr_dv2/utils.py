# TODO: rename somehtng like 'post-processing'
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from skimage.filters.rank import entropy
from skimage.morphology import disk

from typing import Tuple, List


def do_single_pca(arr: np.ndarray, n_components: int = 3) -> np.ndarray:
    # arr in shape (n_samples, n_features)
    pca = PCA(n_components=n_components)
    pca.fit(arr)
    projection = pca.transform(arr)
    return projection


def rescale_pca(pca: np.ndarray) -> np.ndarray:
    # normalize each component of the pca individually
    out = np.zeros_like(pca)
    n_components: int = pca.shape[-1]
    for i in range(n_components):
        c = pca[:, i]
        amax, amin = np.amax(c), np.amin(c)
        out[:, i] = (c - amin) / (amax - amin)
    return out


def normalise_pca(pca: np.ndarray) -> np.ndarray:
    return normalize(pca)


def standardise_pca(pca: np.ndarray) -> np.ndarray:
    # standardize each component of the pca individually
    out = np.zeros_like(pca)
    n_components: int = pca.shape[-1]
    for i in range(n_components):
        c = pca[:, i]
        mean, std = np.mean(c), np.std(c)
        out[:, i] = (c - mean) / std
    return out


def rescale_pca_img(pca_img: np.ndarray) -> np.ndarray:
    # Assume H, W, C
    out = np.zeros_like(pca_img)
    n_components: int = pca_img.shape[-1]
    for i in range(n_components):
        c = pca_img[:, :, i]
        amax, amin = np.amax(c), np.amin(c)
        out[:, :, i] = (c - amin) / (amax - amin)
    return out


def standardise_pca_img(pca_img: np.ndarray) -> np.ndarray:
    # Assume H, W, C
    out = np.zeros_like(pca_img)
    n_components: int = pca_img.shape[-1]
    for i in range(n_components):
        c = pca_img[:, :, i]
        mean, std = np.mean(c), np.std(c)
        out[:, :, i] = (c - mean) / std
    return out


def threshold_pca(
    features: np.ndarray,
    pca: np.ndarray,
    threshold: float,
    greater_than: bool,
    norm: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if norm:
        pca = normalise_pca(pca)

    fg_pca: np.ndarray
    fg_mask: np.ndarray
    if greater_than is True:
        fg_pca = np.where(pca[:, 0] > threshold)
        fg_mask = np.where(pca[:, 0] > threshold, 1, 0)
    else:
        fg_pca = np.where(pca[:, 0] < threshold)
        fg_mask = np.where(pca[:, 0] < threshold, 1, 0)
    fg_features: np.ndarray = features[fg_pca]
    return fg_features, fg_pca, fg_mask


def get_entropy_img(arr: np.ndarray, k: int = 10) -> np.ndarray:
    return entropy(arr, disk(k))


def entropy_per_area(mask: np.ndarray, entropy_img: np.ndarray) -> float:
    n_fg: int = np.sum(mask)
    total_entropy = np.sum(entropy_img * mask)
    return total_entropy / n_fg


def get_best_mask(
    feat_arr: np.ndarray, grey_img_arr: np.ndarray, k: int = 10
) -> Tuple[np.ndarray, int]:
    h, w = grey_img_arr.shape
    entropy_img = get_entropy_img(grey_img_arr, k)

    pca = do_single_pca(feat_arr)
    mask: np.ndarray
    _feat, _pca, mask = threshold_pca(feat_arr, pca, 0, True, True)
    mask = mask.reshape(h, w)
    masks: List[np.ndarray] = [mask, ~mask]

    threshold_1_entropy = entropy_per_area(mask, entropy_img)
    threshold_2_entropy = entropy_per_area(~mask, entropy_img)
    best_mask_idx: int = int(np.argmax([threshold_1_entropy, threshold_2_entropy]))
    best_mask = masks[best_mask_idx]
    return best_mask, best_mask_idx
