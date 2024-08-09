import torch

torch.cuda.empty_cache()
import torch.nn as nn
import torch.utils
import torch.utils.data
import torchvision  # type: ignore
from torch.nn import functional as F
from PIL import Image
from matplotlib.colors import ListedColormap

from hr_dv2 import HighResDV2, torch_pca
import hr_dv2.transform as tr
from hr_dv2.utils import *

from voc import (
    LinearHead,
    apply_state_dict,
    ResizeLongestSide,
    visualise_batch,
    JaccardIndex,
)
from voc import cmap as voc_cmap


from pickle import load as loadp

from os import getcwd, path

CWD = getcwd()
ds_path = path.join(CWD, "experiments", "semantic_seg", "datasets")
idx_path = path.join(ds_path, "ADE20K_2021_17_01")
print(ds_path)

RLS = ResizeLongestSide(518, True)
RLS_no_norm = ResizeLongestSide(518, False)

if __name__ == "__main__":
    state = torch.load(
        "notebooks/figures/fig_data/dinov2_vits14_ade20k_linear_head.pth"
    )
    cmap = state["meta"]["PALETTE"]
    print(state["meta"])

    norm_cmap = [[i / 255 for i in j] for j in cmap]
    mpl_cmap = ListedColormap(norm_cmap)

    with open(f"{idx_path}/index_ade20k.pkl", "rb") as f:
        index = loadp(f)
    print(index.keys())
    print(index["objectnames"])
    with open(f"{idx_path}/categoryMapping.txt", "r") as f:
        text = f.read().splitlines()[1:]
        k, v = [], []
        for t in text:
            parsed = t.split("\t\t")
            k.append(int(parsed[-1]))
            v.append(int(parsed[1]))

        k, v = np.array(k), np.array(v)
        mapping_ar = np.zeros(k.max() + 1, dtype=v.dtype)  # k,v from approach #1
        mapping_ar[k] = v

    jac = JaccardIndex(num_classes=150, task="multiclass", ignore_index=-1).cuda()
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

    model = LinearHead(150)
    model = apply_state_dict(state, model)

    last10x, last10y, last10y_pred = [], [], []
    n_val = 0
    for i in range(len(index["filename"])):
        folder_name = index["folder"][i]
        file_name = index["filename"][i]

        if "training" in folder_name:
            continue
        else:
            img = Image.open(f"{ds_path}/{folder_name}/{file_name}")
            seg_fname = file_name.split(".")[0] + "_seg.png"
            seg_img = Image.open(f"{ds_path}/{folder_name}/{seg_fname}")
            seg_arr = np.array(seg_img, dtype=np.int32)

            R, G, B = seg_arr[:, :, 0], seg_arr[:, :, 1], seg_arr[:, :, 2]
            class_mask = R // 10 * 256 + G
            class_mask = mapping_ar[class_mask]

            x = RLS(img)
            n_val += 1
            y = torch.tensor(class_mask, dtype=torch.int32)
            y[y > 150] = -1
            y = y.to(torch.float32)

            y = F.interpolate(y.unsqueeze(0).unsqueeze(0), (x.shape[-2], x.shape[-1]))
            y = y.squeeze(0).squeeze(0).to(torch.int32)
            C, H, W = x.shape
            x, y = x.cuda(), y.cuda()

            # feats = jbu.forward(x.unsqueeze(0))
            feats = net.forward(x)
            feats = F.interpolate(feats, (H, W))
            logits = model(feats)

            y_pred = torch.argmax(logits, dim=1)

            last10x.append(x.unsqueeze(0))
            last10y.append(y.unsqueeze(0))
            last10y_pred.append(y_pred)

            # jac.update(y_pred, y)

            if n_val > 8:
                last10x.pop(0)
                last10y.pop(0)
                last10y_pred.pop(0)

            if n_val % 10 == 0 and n_val > 9:
                visualise_batch(
                    last10x,
                    last10y,
                    last10y_pred,
                    f"ours {i}",
                    f"{CWD}/experiments/semantic_seg/ade_out/jbu/{n_val}.png",
                    voc_cmap,
                )
                print(f"[{i} / {len(index['filename'])}]: {jac.compute()}")

            if n_val > 50:
                break
