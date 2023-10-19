import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA

from typing import Tuple


def get_input_transform(resize_dim: int, crop_dim: int) -> transforms.Compose:
    transform = transforms.Compose(
        [
            transforms.Resize(resize_dim),
            transforms.CenterCrop(crop_dim),
            transforms.ToTensor(),
            # transforms.Normalize(mean=0.5, std=0.2),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return transform


to_norm_tensor = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

unnormalize = transforms.Normalize(
    mean=(-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225),
    std=(1 / 0.229, 1 / 0.224, 1 / 0.225),
)

to_img = transforms.ToPILImage()


def load_image(
    path: str, transform: transforms.Compose
) -> Tuple[torch.Tensor, Image.Image]:
    # Load image with PIL, convert to tensor by applying $transform, and invert transform to get display image
    image = Image.open(path).convert("RGB")
    tensor = transform(image)
    transformed_img = to_img(unnormalize(tensor))
    return tensor, transformed_img


def to_numpy(x: torch.Tensor, batched: bool = True) -> np.ndarray:
    if batched:
        x = x.squeeze(0)
    return x.detach().cpu().numpy()


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
    # normalize each component of the pca individually
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


def normalise_pca_img(pca_img: np.ndarray) -> np.ndarray:
    # Assume H, W, C
    out = np.zeros_like(pca_img)
    n_components: int = pca_img.shape[-1]
    for i in range(n_components):
        c = pca_img[:, :, i]
        mean, std = np.mean(c), np.std(c)
        out[:, :, i] = (c - mean) / std
    return out
