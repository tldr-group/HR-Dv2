import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor
from torchmetrics.classification.jaccard import JaccardIndex

import numpy as np
from PIL import Image
import os
import mmcv

from hr_dv2 import HighResDV2, torch_pca
import hr_dv2.transform as tr
from hr_dv2.utils import *

from voc import (
    inp_transform,
    targ_transform,
    LinearHead,
    apply_state_dict,
    CWD,
    visualise_batch,
)

cfg = torch.load(
    f"{CWD}/notebooks/figures/fig_data/dinov2_vits14_ade20k_linear_head.pth"
)
mmcv_cfg = mmcv.Config(cfg["meta"])


class ADE20k(Dataset):
    def __init__(self, root: str, transform, target_transform) -> None:
        super().__init__()
        self.root = root
        self.image_paths: list[str] = []
        self.mask_paths: list[str] = []
        for category in os.listdir(self.root):
            category_path = os.path.join(self.root, category)
            for subcategory in os.listdir(category_path):
                subcategory_path = os.path.join(category_path, subcategory)
                for f in os.listdir(subcategory_path):
                    if ".jpg" in f:
                        img_path = os.path.join(subcategory_path, f)
                        self.image_paths.append(img_path)
                        fname = f.split(".")[0]
                        mask_path = img_path = os.path.join(
                            subcategory_path, f"{fname}_seg.png"
                        )
                        self.mask_paths.append(mask_path)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.image_paths[index]).convert("RGB")
        target = Image.open(self.mask_paths[index])

        target_np = np.array(target).astype(np.float32)
        R, G = target_np[:, :, 0], target_np[:, :, 1]
        target_classes_np = (R / 10) * 256 + G
        print(target_classes_np.shape)

        img = self.transform(img)
        target_classes = torch.tensor(target_classes_np).unsqueeze(0).unsqueeze(0)
        classes = (
            F.interpolate(target_classes, (img.shape[1], img.shape[2]))
            .squeeze(0)
            .squeeze(0)
        )

        return img, classes

    def __len__(self):
        return len(self.image_paths)


if __name__ == "__main__":
    jac = JaccardIndex(num_classes=150, task="multiclass").cuda()
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

    model = LinearHead(150)
    cfg = torch.load(
        f"{CWD}/notebooks/figures/fig_data/dinov2_vits14_ade20k_linear_head.pth"
    )
    model = apply_state_dict(cfg, model)

    dataset = ADE20k(
        f"{CWD}/experiments/semantic_seg/datasets/ADE20K_2021_17_01/images/ADE/validation",
        inp_transform,
        targ_transform,
    )

    last10x, last10y, last10y_pred = [], [], []
    for i, batch in enumerate(dataset):
        x, y = batch
        C, H, W = x.shape
        x, y = x.cuda(), y.cuda()
        y = y.unsqueeze(0)

        feats = jbu.forward(x.unsqueeze(0))
        feats = F.interpolate(feats, (H, W))
        logits = model(feats)

        y = y.to(torch.int32)
        y[y >= 150] = 0  # todo: fix
        y_pred = torch.argmax(logits, dim=1)
        # print(x.shape, y.shape, y_pred.shape)
        # print(torch.amax(y_pred), torch.amin(y_pred))
        # print(torch.amax(y), torch.amin(y))

        last10x.append(x.unsqueeze(0))
        last10y.append(y)
        last10y_pred.append(y_pred)

        try:
            jac.update(y_pred, y)
        except RuntimeError:  # todo: fix
            continue

        if i > 8:
            last10x.pop(0)
            last10y.pop(0)
            last10y_pred.pop(0)

        if i % 10 == 0 and i > 9:
            visualise_batch(
                last10x,
                last10y,
                last10y_pred,
                f"ours {i}",
                f"{CWD}/experiments/semantic_seg/ade_out/jbu/{i}.png",
                np.array(mmcv_cfg["PALETTE"]),
            )
            print(f"[{i} / {len(dataset)}]: {jac.compute()}")
