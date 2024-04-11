import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

from functools import partial
from itertools import product
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
    transforms: PartialTrs = [true_iden_partial]
    inv_transforms: PartialTrs = [true_iden_partial]
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


def get_shift_transforms_conv(
    dists: List[int], pattern: Literal["Neumann", "Moore"]
) -> Tuple[PartialTrs, PartialTrs]:
    # don't think convs is causing a speed increase
    transforms: PartialTrs = []
    inv_transforms: PartialTrs = []
    shifts = compute_shift_directions(pattern)

    def conv_arg_rev(d: int, shift: Tuple[int, int], x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 3:  # batch
            x = x.unsqueeze(0)
        sx, sy = shift
        b, c, h, w = x.shape
        c = int(c)
        mask = torch.zeros((c, 1, 3, 3), device=x.device)
        mask[:, :, 1 + sx, 1 + sy] = 1
        padded: torch.Tensor = F.pad(x, (d, d, d, d), mode="circular")
        shifted = F.conv2d(padded, mask, stride=1, dilation=d, groups=c)
        return shifted

    for d in dists:
        for s in shifts:
            inv_shift = (-s[0], -s[1])
            tr = partial(conv_arg_rev, d, s)
            inv_tr = partial(conv_arg_rev, d, inv_shift)
            transforms.append(tr)
            inv_transforms.append(inv_tr)

    return transforms, inv_transforms


def iden(x: torch.Tensor) -> torch.Tensor:
    if len(x.shape) == 3:  # batch
        x = x.unsqueeze(0)
    return x


def true_iden(x: torch.Tensor) -> torch.Tensor:
    return x


iden_partial = partial(iden)
true_iden_partial = partial(true_iden)


def get_flip_transforms() -> Tuple[PartialTrs, PartialTrs]:
    def flip_arg_rev(dims: Tuple[int, ...], x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 3:  # batch
            x = x.unsqueeze(0)
        return torch.flip(x, dims)

    # np.flip would be faster apparently

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


def get_rotation_transforms() -> Tuple[PartialTrs, PartialTrs]:
    def rot_arg_rev(dims: Tuple[int, ...], k: int, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 3:  # batch
            x = x.unsqueeze(0)
        return torch.rot90(x, k, dims)

    transforms = [iden_partial]
    inv_transforms = [iden_partial]
    for direction in [-1, 1]:
        for angle in [1, 2, 3]:
            fwd = partial(rot_arg_rev, (-2, -1), direction * angle)
            inv = partial(rot_arg_rev, (-2, -1), -direction * angle)
            if direction == 1 and angle == 2:
                pass
            else:
                transforms.append(fwd)
                inv_transforms.append(inv)
    return transforms, inv_transforms


def combine_transforms(
    fwd_1: PartialTrs, fwd_2: PartialTrs, inv_1: PartialTrs, inv_2: PartialTrs
) -> Tuple[PartialTrs, PartialTrs]:
    def combined(tr_1, tr_2, x: torch.Tensor) -> torch.Tensor:
        return tr_2(tr_1(x))

    fwd_pairs = product(fwd_1, fwd_2)
    inv_pairs = product(inv_1, inv_2)  # no guarantee these are in order

    fwd: PartialTrs = []
    inv: PartialTrs = []
    for pair in fwd_pairs:
        f1, f2 = pair
        partial_combined = partial(combined, f1, f2)
        fwd.append(partial_combined)
    for pair in inv_pairs:
        i1, i2 = pair
        partial_combined = partial(combined, i1, i2)
        inv.append(partial_combined)
    return (fwd, inv)


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


def closest_crop(
    h: int, w: int, patch_size: int = 14, to_tensor: bool = True
) -> transforms.Compose:
    # Crop to h,w values that are closest to given patch/stride size
    sub_h: int = h % patch_size
    sub_w: int = w % patch_size
    new_h, new_w = h - sub_h, w - sub_w
    if to_tensor:
        transform = transforms.Compose(
            [
                transforms.CenterCrop((new_h, new_w)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.CenterCrop((new_h, new_w)),
            ]
        )
    return transform


def closest_pad(h: int, w: int, patch_size: int = 14) -> transforms.Compose:
    add_h: int = patch_size - (h % patch_size)
    add_w: int = patch_size - (w % patch_size)
    transform = transforms.Compose(
        [
            transforms.Pad((add_w, add_h, 0, 0)),
        ]
    )
    return transform


to_img = transforms.ToPILImage()
to_tensor = transforms.ToTensor()


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
