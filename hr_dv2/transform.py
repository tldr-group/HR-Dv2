import torch
import torchvision.transforms as transforms

from functools import partial
from PIL import Image
import numpy as np
from typing import List, Literal, TypeAlias, Tuple, Callable

PartialTrs: TypeAlias = List[partial]


# ==================== MODEL TRANSFORMS ====================
def compute_shift_directions(
    pattern: Literal["Neumann", "Moore"]
) -> List[Tuple[int, int]]:
    # Precompute neighbourhood shift unit vectors
    shifts = [  # shifts in yx format
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 0),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]
    shift_directions: List[Tuple[int, int]] = []
    for i in range(9):
        if pattern == "Neumann" and i % 2 == 1:
            shift_directions.append(shifts[i])
        elif pattern == "Moore" and i != 4:
            shift_directions.append(shifts[i])
    return shift_directions


def get_shift_transforms(
    dists: List[int], pattern: Literal["Neumann", "Moore"]
) -> Tuple[PartialTrs, PartialTrs]:
    transforms: PartialTrs = []
    inv_transforms: PartialTrs = []
    shifts = compute_shift_directions(pattern)

    def roll_arg_rev(shift: Tuple[int, int], x: torch.Tensor) -> torch.Tensor:
        return torch.roll(x, shift, dims=(-2, -1))

    for d in dists:
        for s in shifts:
            shift = (d * s[0], d * s[1])
            inv_shift = (-d * s[0], -d * s[1])
            tr = partial(roll_arg_rev, shift)
            inv_tr = partial(roll_arg_rev, inv_shift)
            transforms.append(tr)
            inv_transforms.append(inv_tr)
    return transforms, inv_transforms


def iden(x: torch.Tensor) -> torch.Tensor:
    return x


iden_partial = partial(iden)


def get_flip_transforms() -> Tuple[PartialTrs, PartialTrs]:
    def flip_arg_rev(dims: Tuple[int, ...], x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, dims)

    horizontal_flip_partial = partial(flip_arg_rev, (-1,))
    vertical_flip_partial = partial(flip_arg_rev, (-2,))
    transforms = [
        iden_partial,
        horizontal_flip_partial,
        iden_partial,
        vertical_flip_partial,
    ]
    inv_tranforms = [
        iden_partial,
        horizontal_flip_partial,
        iden_partial,
        vertical_flip_partial,
    ]
    return transforms, inv_tranforms


# TODO: add rotations and a general formula for composing the transfroms, will probs be
# a combination of itertools premutations and wrapptes around partial functions
# i.e will want to compute all shifts then all flips of shifts
# having many transforms will necessitate a redesign of high res to have the option
# to be sequential


# ==================== PYTORCH INPUT TRANSFORMS ====================
def get_input_transform(resize_dim: int, crop_dim: int) -> transforms.Compose:
    transform = transforms.Compose(
        [
            transforms.Resize(resize_dim),
            transforms.CenterCrop(crop_dim),
            transforms.ToTensor(),
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


def centre_crop(crop_h: int, crop_w: int) -> transforms.Compose:
    transform = transforms.Compose(
        [
            transforms.CenterCrop((crop_h, crop_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return transform


def closest_crop(h: int, w: int, patch_size: int = 14) -> transforms.Compose:
    # Crop to h,w values that are closest to given patch/stride size
    sub_h: int = h % patch_size
    sub_w: int = w % patch_size
    new_h, new_w = h - sub_h, w - sub_w
    transform = transforms.Compose(
        [
            transforms.CenterCrop((new_h, new_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return transform


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


def flatten(
    x: torch.Tensor | np.ndarray, h: int, w: int, c: int, convert: bool = False
) -> np.ndarray:
    if type(x) == torch.Tensor:
        x = to_numpy(x)
    x = x.reshape((c, h * w))
    x = x.T
    return x  # type: ignore
