import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from typing import Tuple


def get_input_transform(resize_dim: int, crop_dim: int) -> transforms.Compose:
    transform = transforms.Compose(
        [
            transforms.Resize(resize_dim),
            transforms.CenterCrop(crop_dim),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.2),
        ]
    )
    return transform


unnormalize = transforms.Normalize(mean=-0.5 / 0.2, std=1 / 0.2)

to_img = transforms.ToPILImage()


def load_image(
    path: str, transform: transforms.Compose
) -> Tuple[torch.Tensor, Image.Image]:
    # Load image with PIL, convert to tensor by applying $transform, and invert transform to get display image
    image = Image.open(path).convert("RGB")
    tensor = transform(image)
    transformed_img = to_img(unnormalize(tensor))
    return tensor, transformed_img


def normalise_pca(pca: np.ndarray) -> np.ndarray:
    # normalize each component of the pca individually
    out = np.zeros_like(pca)
    n_components: int = pca.shape[-1]
    for i in range(n_components):
        c = pca[:, i]
        amax, amin = np.amax(c), np.amin(c)
        out[:, i] = (c - amin) / (amax - amin)
    return out


def normalize_pca_img(pca_img: np.ndarray) -> np.ndarray:
    # Assume H, W, C
    out = np.zeros_like(pca_img)
    n_components: int = pca_img.shape[-1]
    for i in range(n_components):
        c = pca_img[:, :, i]
        amax, amin = np.amax(c), np.amin(c)
        out[:, :, i] = (c - amin) / (amax - amin)
    return out
