import torch

torch.cuda.empty_cache()
import torch.nn as nn
import torch.utils
import torch.utils.data
import torchvision  # type: ignore

from os import getcwd

CWD = getcwd()
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import resize, to_pil_image, pil_to_tensor  # type: ignore
from torchvision.transforms.functional import normalize as vision_norm
from torchvision.transforms import ToTensor, Normalize

from hr_dv2 import HighResDV2, torch_pca
import hr_dv2.transform as tr
from hr_dv2.utils import *
from torchmetrics.classification.jaccard import JaccardIndex

from PIL import Image  # type: ignore
import matplotlib.pyplot as plt


class ResizeLongestSide:
    """
    Resizes images to the longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming both numpy array and batched torch tensors.
    """

    def __init__(self, target_length: int, norm: bool = False) -> None:
        self.target_length = target_length
        self.norm = norm
        self.normalize = Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )
        self.to_tensor = ToTensor()

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(
            image.shape[0], image.shape[1], self.target_length
        )
        return np.array(resize(to_pil_image(image), target_size))

    def apply_image_torch(self, image: torch.Tensor) -> torch.Tensor:
        """
        Expects batched images with shape BxCxHxW and float format. This
        transformation may not exactly match apply_image. apply_image is
        the transformation expected by the model.
        """
        # Expects an image in BCHW format. May not exactly match apply_image.
        target_size = self.get_preprocess_shape(
            image.shape[2], image.shape[3], self.target_length
        )
        return F.interpolate(
            image, target_size, mode="bilinear", align_corners=False, antialias=False
        )

    def __call__(self, x: Image.Image) -> torch.Tensor:
        arr = np.asarray(x)
        target_size = self.get_preprocess_shape(
            arr.shape[0], arr.shape[1], self.target_length
        )
        if self.norm:
            tensor = self.to_tensor(x)
        else:
            tensor = pil_to_tensor(x)
        resized = resize(tensor, target_size, antialias=False)

        if self.norm:
            resized = self.normalize(resized)

        return resized

    @staticmethod
    def get_preprocess_shape(
        oldh: int, oldw: int, long_side_length: int, patch_size: int = 14
    ) -> tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        neww = neww - (neww % patch_size)
        newh = newh - (newh % patch_size)
        return (newh, neww)


def bit_get(val, idx):
    """Gets the bit value.
    Args:
      val: Input value, int or numpy int array.
      idx: Which bit of the input val.
    Returns:
      The "idx"-th bit of input val.
    """
    return (val >> idx) & 1


def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.
    Returns:
      A colormap for visualizing segmentation results.
    """
    colormap = np.zeros((512, 3), dtype=int)
    ind = np.arange(512, dtype=int)

    for shift in reversed(list(range(8))):
        for channel in range(3):
            colormap[:, channel] |= bit_get(ind, channel) << shift
        ind >>= 3

    return colormap


def to_np(x: torch.Tensor, squeeze: bool = False, unnorm: bool = False) -> np.ndarray:
    # assume BCHW
    if unnorm:
        x = tr.unnormalize(x)

    if x.shape[1] == 3:  # rgb
        x = x.permute((0, 2, 3, 1))
    else:  # greyscale
        x = x.squeeze(1)

    if squeeze:
        x = x.squeeze(0)
    return x.detach().cpu().numpy()


def visualise_batch(
    x: list[torch.Tensor],
    y: list[torch.Tensor],
    y_pred: list[torch.Tensor],
    title: str,
    save_dir: str,
    cmap,
) -> None:
    n = len(x)
    fig, axs = plt.subplots(nrows=3, ncols=n)
    fig.set_size_inches(16, 8)
    plt.suptitle(title, fontsize=22)

    for j, title in enumerate(["x", "y", "y_pred"]):
        axs[j, 0].set_ylabel(title, fontsize=18)

    for i in range(n):
        for j, data in enumerate([x[i], y[i], y_pred[i]]):
            if j == 0:
                data = to_np(data, True, True)
            elif j == 1:
                data = to_np(data, True, False)
                data = cmap[data]
            if j > 1:
                data = to_np(data, True, False)
                data = cmap[data]
            axs[j, i].imshow(data, interpolation="nearest")
            axs[j, i].set_axis_off()
    plt.tight_layout()
    plt.savefig(save_dir)
    plt.close(fig)


C = 384


class LinearHead(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.bn = nn.SyncBatchNorm(C)
        self.conv_seg = nn.Conv2d(C, n_classes, 1, 1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        feats = feats.to(torch.float)
        feats = self.bn(feats)
        return self.conv_seg(feats)


def apply_state_dict(raw_cfg: dict, linear_head: nn.Module) -> nn.Module:
    new_state_dict = {}
    for keys, vals in raw_cfg["state_dict"].items():
        new_key = keys.replace("decode_head.", "")
        new_state_dict[new_key] = vals
    linear_head.load_state_dict(new_state_dict)
    linear_head.cuda()
    linear_head.eval()
    return linear_head


cmap = create_pascal_label_colormap()
inp_transform, targ_transform = ResizeLongestSide(518, True), ResizeLongestSide(518)


if __name__ == "__main__":
    jac = JaccardIndex(num_classes=21, task="multiclass", ignore_index=-1).cuda()
    backbone_name = "dinov2_vits14"
    net = HighResDV2(
        backbone_name, 4, dtype=torch.float16
    )  # dino_vits8 #dinov2_vits14_reg
    # net.interpolation_mode = "bilinear"
    net.interpolation_mode = "nearest-exact"
    net.eval()
    net.cuda()

    fwd_shift, inv_shift = tr.get_shift_transforms([1, 2], "Moore")
    fwd_flip, inv_flip = tr.get_flip_transforms()
    fwd, inv = tr.combine_transforms(fwd_shift, fwd_flip, inv_shift, inv_flip)
    net.set_transforms(fwd_shift, inv_shift)

    jbu = torch.hub.load("mhamilton723/FeatUp", "dinov2", use_norm=False)
    jbu.eval()
    jbu.cuda()
    pass

    model = LinearHead(21)
    cfg = torch.load(
        f"{CWD}/notebooks/figures/fig_data/dinov2_vits14_voc2012_linear_head.pth"
    )
    model = apply_state_dict(cfg, model)

    dataset = torchvision.datasets.VOCSegmentation(
        f"{CWD}/experiments/object_localization/datasets/VOC2012/",
        download=False,
        image_set="trainval",
        transform=inp_transform,
        target_transform=targ_transform,
    )

    last10x, last10y, last10y_pred = [], [], []
    for i, batch in enumerate(dataset):
        x, y = batch
        C, H, W = x.shape
        x, y = x.cuda(), y.cuda()
        # x = x.unsqueeze(0)

        feats = jbu.forward(x.unsqueeze(0))
        feats = F.interpolate(feats, (H, W))
        logits = model(feats)

        y = y.to(torch.long)
        y[y > 20] = -1
        y_pred = torch.argmax(logits, dim=1)

        last10x.append(x.unsqueeze(0))
        last10y.append(y.unsqueeze(0))
        last10y_pred.append(y_pred)

        jac.update(y_pred, y)

        if i > 8:
            last10x.pop(0)
            last10y.pop(0)
            last10y_pred.pop(0)

        if i % 50 == 0 and i > 20:
            visualise_batch(
                last10x,
                last10y,
                last10y_pred,
                f"ours {i}",
                f"{CWD}/experiments/semantic_seg/voc_out/jbu/{i}.png",
            )
            print(f"[{i} / {len(dataset)}]: {jac.compute()}")


"""
vanilla dv2 mIoU: 0.8060468435287476
ours dv2 stride 4 st1,2: 0.7016257047653198 (lol)
1950 plant comparison
featUp dv2 mIou: 0.8245130181312561

"""
