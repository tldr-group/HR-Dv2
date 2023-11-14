"""
Functions copied from 'deep-spectral-segmentation' (https://github.com/lukemelas/deep-spectral-segmentation/tree/main)
which uses graph-based methods on deep ViT features for automatic object localization and semantic segmentation by
considering eigenvectors of the Laplacian of a feature affinity matrix that combines ViT features and a colour
affinity matrix of an image. They were resolution limited by the patch size of the ViT but we can use HR-Dv2 to
achieve high resolution feature maps and feed these in.
"""
import numpy as np
import scipy
from scipy.sparse.linalg import eigsh
from pymatting.util.kdtree import knn
from pymatting.util.util import row_sum

from .utils import rescale_pca

from typing import List, Tuple


def knn_affinity(
    image: np.ndarray,
    n_neighbors: List[int] = [20, 10],
    distance_weights: List[float] = [2.0, 0.1],
) -> scipy.sparse.csr_matrix:
    """Computes a KNN-based affinity matrix. Note that this function requires pymatting

    :param image: (normalised) image arr we want the knn colour affinity matrix of
    :type image: np.ndarray
    :param n_neighbors: number of nearest neighbours to consider, defaults to [20, 10]
    :type n_neighbors: List[int], optional
    :param distance_weights: _description_, defaults to [2.0, 0.1]
    :type distance_weights: List[float], optional
    :return: knn affinity matrix
    :rtype: scipy.sparse.csr_matrix
    """

    h, w = image.shape[:2]
    # Colour info
    r, g, b = image.reshape(-1, 3).T
    n = w * h

    x = np.tile(np.linspace(0, 1, w), h)
    y = np.repeat(np.linspace(0, 1, h), w)

    i, j = [], []

    for k, distance_weight in zip(n_neighbors, distance_weights):
        f = np.stack(
            [r, g, b, distance_weight * x, distance_weight * y],
            axis=1,
            out=np.zeros((n, 5), dtype=np.float32),
        )

        distances, neighbors = knn(f, f, k=k)

        i.append(np.repeat(np.arange(n), k))
        j.append(neighbors.flatten())

    ij = np.concatenate(i + j)
    ji = np.concatenate(j + i)
    coo_data = np.ones(2 * sum(n_neighbors) * n)

    # This is our affinity matrix
    W = scipy.sparse.csr_matrix((coo_data, (ij, ji)), (n, n))
    return W


def get_diagonal(
    W: scipy.sparse.csr_matrix, threshold: float = 1e-12
) -> scipy.sparse.csr_matrix:
    """Gets the diagonal sum of a sparse matrix

    :param W: sparse affinity matrix
    :type W: scipy.sparse.csr_matrix
    :param threshold: epsilon to prevent zero divison, defaults to 1e-12
    :type threshold: float, optional
    :return: diagonal sum of sparse matrix
    :rtype: scipy.sparse.csr_matrix
    """

    D = row_sum(W)
    D[D < threshold] = 1.0  # Prevent division by zero.
    D = scipy.sparse.diags(D)
    return D


def get_affinity_eigvectors(
    features: np.ndarray,
    image: np.ndarray,
    image_colour_lambda: float = 8.0,
    K: int = 5,
    normalize: bool = True,
    threshold_at_zero: bool = True,
    lapnorm=True,
) -> Tuple[np.ndarray, np.ndarray]:
    if normalize:
        features = rescale_pca(features)
        # features = (features - mins) / (maxs - mins)
    print("Starting mat mul")
    W_feat = features @ features.T
    if threshold_at_zero:
        W_feat = W_feat * (W_feat > 0)
    W_feat = W_feat / W_feat.max()
    print("Staring affinity")
    if image_colour_lambda > 0:
        W_lr = knn_affinity(image)
        W_color = np.array(W_lr.todense().astype(np.float32))
    else:
        W_color = 0
    print("Starting diagonalising")
    W_comb = W_feat + W_color * image_colour_lambda
    D_comb = np.array(get_diagonal(W_comb).todense())
    print("Finding eigvectors")
    eigenvalues: np.ndarray = np.zeros((1))
    eigenvectors: np.ndarray = np.zeros((1))
    if lapnorm:
        try:
            eigenvalues, eigenvectors = eigsh(
                D_comb - W_comb, k=K, sigma=0, which="LM", M=D_comb
            )
        except:
            eigenvalues, eigenvectors = eigsh(
                D_comb - W_comb, k=K, which="SM", M=D_comb
            )
    else:
        try:
            eigenvalues, eigenvectors = eigsh(D_comb - W_comb, k=K, sigma=0, which="LM")
        except:
            eigenvalues, eigenvectors = eigsh(D_comb - W_comb, k=K, which="SM")
    # Sign ambiguity
    for k in range(eigenvectors.shape[0]):
        if 0.5 < np.mean((eigenvectors[k] > 0).float()).item() < 1.0:  # reverse segment
            eigenvectors[k] = 0 - eigenvectors[k]
    return eigenvalues, eigenvectors
