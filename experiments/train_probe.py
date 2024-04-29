import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image

from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.jaccard import JaccardIndex

torch.cuda.empty_cache()

from timm import create_model
from hr_dv2 import HighResDV2
import hr_dv2.transform as tr
import numpy as np

from semantic_seg.datasets.coco import (
    Coco,
    inp_tr,
    label_tr,
    create_pascal_label_colormap,
)

import matplotlib.pyplot as plt
from matplotlib import colormaps
from skimage.color import label2rgb

colours = colormaps["tab20"]
cmap = [colours(i) for i in range(20)]

new_cmap = create_pascal_label_colormap()


train = Coco(
    "experiments/semantic_seg/datasets/coco",
    "train",
    inp_tr,
    label_tr,
    coarse_labels=False,
)
val = Coco(
    "experiments/semantic_seg/datasets/coco",
    "val",
    inp_tr,
    label_tr,
    coarse_labels=False,
)

BATCH_SIZE = 128
train_loader = DataLoader(
    train, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=12
)
val_loader = DataLoader(
    val, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=12
)
# NEED TO FEATURISE IMAGE FIRST! maybe use a hrdv2?
N_FEAT_DIM = 384
featurizer = HighResDV2("vit_vits16", 16, pca_dim=-1, dtype=torch.float16)
featurizer.cuda()
featurizer.eval()

net = nn.Conv2d(N_FEAT_DIM, 27, 1, 1)
net.cuda()
optim = torch.optim.Adam(net.parameters(), 0.001)
loss = nn.CrossEntropyLoss(ignore_index=-1)

acc = Accuracy(num_classes=27, task="multiclass", top_k=1, ignore_index=-1).cuda()
jac = JaccardIndex(num_classes=27, task="multiclass", ignore_index=-1).cuda()

TRAIN_BATCHES_PER_EPOCH = 100  # 100
VAL_BATCHES_PER_EPOCH = 10
N_EPOCHS = 200


def visualise_batch(
    x: np.ndarray,
    y: np.ndarray,
    y_pred: np.ndarray,
    n: int,
    save_dir: str,
    epoch: int,
    train: bool,
) -> None:
    title = "train" if train else "val"
    x, y, y_pred = x[:n], y[:n], y_pred[:n]
    x = x.transpose((0, 2, 3, 1))
    fig, axs = plt.subplots(nrows=3, ncols=n)
    fig.set_size_inches(30, 10)

    def colorise(a: np.ndarray) -> np.ndarray:
        h, w = a.shape

        flat = a.flatten()
        col = new_cmap[flat]
        return col.reshape(h, w, 3)

    for i in range(n):
        axs[0, i].imshow(x[i].astype(np.float32))
        pred_colorised = colorise(y_pred[i]).astype(np.uint8)
        axs[1, i].imshow(pred_colorised)
        gt_colorised = colorise(y[i]).astype(np.uint8)
        axs[2, i].imshow(gt_colorised)

    for row in range(3):
        for col in range(n):
            axs[row, col].set_axis_off()
    plt.suptitle(f"{title} predictions")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{epoch}_{title}_grid.png")
    plt.close(fig)


def plot_losses(
    train_losses: list[float], val_losses: list[float], epochs: list[int], save_dir: str
) -> None:
    fig = plt.figure()
    plt.plot(epochs, train_losses, lw=3, label="Train")
    plt.plot(epochs, val_losses, lw=3, label="Val")

    plt.xlim(0, N_EPOCHS)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.savefig(f"{save_dir}/{epochs[-1]}_plot.png")

    plt.close(fig)


def get_spatialized_features(
    x: torch.Tensor,
    img_w: int = 224,
    img_h: int = 224,
    patch_size: int = 16,
    high_res: bool = False,
) -> torch.Tensor:
    if high_res is False:
        feat_dict = featurizer.dinov2.forward_feats_attn(x, None, "none")  # type: ignore
        feats = feat_dict["x_norm_patchtokens"].to(torch.float)
        n_patch_w: int = 1 + (img_w - patch_size) // featurizer.stride[0]
        n_patch_h: int = 1 + (img_h - patch_size) // featurizer.stride[0]

        B, T, C = feats.shape

        f = feats.permute((0, 2, 1))  # swap C and T
        f = f.reshape((B, C, n_patch_h, n_patch_w))  # B, C, H, W
        f = torch.nn.functional.interpolate(f, (img_h, img_w))
    else:
        feat_list = []
        for b in range(x.shape[0]):
            feat = featurizer.forward(x[b])
            feat_list.append(feat)
        f = torch.concat(feat_list).to(torch.float)
    return f


def feed_batch(
    net: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: nn.CrossEntropyLoss,
    optim: torch.optim.Adam,
    train: bool = True,
    high_res: bool = False,
) -> tuple[torch.Tensor, float]:
    optim.zero_grad()
    with torch.no_grad():
        feats = get_spatialized_features(x, high_res=high_res)
    y_pred = net(feats)
    loss = loss_fn(y_pred, y)

    if train:
        loss.backward()
        optim.step()
    else:
        print(acc(y_pred, y))
        print(jac(y_pred, y))
    return y_pred, loss.item()


def do_epoch(
    net: nn.Module,
    n_batches: int,
    loader: DataLoader,
    loss_fn: nn.CrossEntropyLoss,
    optim: torch.optim.Adam,
    epoch: int,
    train: bool = True,
) -> float:
    epoch_loss = 0.0
    for i, data in enumerate(loader):
        if i > n_batches:
            return epoch_loss
        x, y = data["img"], data["label"]
        x, y = x.cuda(), y.cuda()
        x = x.to(torch.half)

        y_pred, loss = feed_batch(net, x, y, loss_fn, optim, train)

        epoch_loss += loss

        x = x.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        y_pred = torch.argmax(y_pred, dim=1).cpu().detach().numpy()

        if i == n_batches:
            visualise_batch(x, y, y_pred, 10, "experiments/probe_out", epoch, train)
            if train:
                torch.save(net, f"experiments/probe_out/{epoch}.pth")

    return epoch_loss


def train_loop():
    train_losses, val_losses, epochs = [], [], []
    for i in range(N_EPOCHS):
        epoch_loss = do_epoch(
            net, TRAIN_BATCHES_PER_EPOCH, train_loader, loss, optim, i, True
        )
        val_loss = do_epoch(
            net, VAL_BATCHES_PER_EPOCH, val_loader, loss, optim, i, False
        )

        epochs.append(i)
        train_losses.append(epoch_loss)
        val_losses.append(val_loss)
        plot_losses(train_losses, val_losses, epochs, "experiments/probe_out")
        print(f"[{i}/{N_EPOCHS}]: {epoch_loss}")


def val_loop(high_res: bool = False):
    classifier = torch.load("experiments/probe_out/best_small_batch_178.pth")
    featurizer.set_model_stride(featurizer.dinov2, 4)
    featurizer.interpolation_mode = "bilinear"

    if high_res:
        fwd_shift, inv_shift = tr.get_shift_transforms(
            [
                1,
                2,
            ],
            "Moore",
        )
        fwd_flip, inv_flip = tr.get_flip_transforms()
        fwd, inv = tr.combine_transforms(fwd_shift, fwd_flip, inv_shift, inv_flip)
        featurizer.set_transforms(fwd, inv)

    for i, data in enumerate(val_loader):
        print(i)
        x, y = data["img"], data["label"]
        x, y = x.cuda(), y.cuda()
        x = x.to(torch.half)

        y_pred, loss_val = feed_batch(
            classifier, x, y, loss, optim, False, high_res=high_res
        )

        acc.update(y_pred, y)
        jac.update(y_pred, y)

        x = x.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        y_pred = torch.argmax(y_pred, dim=1).cpu().detach().numpy()

        try:
            visualise_batch(x, y, y_pred, 10, "experiments/probe_out/val", i, False)
        except IndexError:  # uneven final batch
            pass

    print(acc.compute())
    print(jac.compute())


if __name__ == "__main__":
    val_loop(True)


"""
theirs:
vit16s:
top-1 acc: tensor(0.6517, device='cuda:0')
mIoU: tensor(0.4065, device='cuda:0')

featup:
top-1 acc: tensor(0.6877, device='cuda:0')
mIoU: tensor(0.4341, device='cuda:0')


ours:
vit16s:
top-1 acc: tensor(0.6331, device='cuda:0')
mIoU: tensor(0.3973, device='cuda:0')

vit16s-strided (no bilinear):
top-1 acc: tensor(0.6241, device='cuda:0')
mIoU: tensor(0.3839, device='cuda:0')

"""
