import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image

torch.cuda.empty_cache()

from timm import create_model
from hr_dv2 import HighResDV2

from semantic_seg.datasets.coco import Coco, inp_tr, label_tr

import matplotlib.pyplot as plt

train = Coco("experiments/semantic_seg/datasets/coco", "train", inp_tr, label_tr)
val = Coco("experiments/semantic_seg/datasets/coco", "val", inp_tr, label_tr)

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
optim = torch.optim.Adam(net.parameters(), 0.005)
loss = nn.CrossEntropyLoss(reduction="mean", ignore_index=-1)

TRAIN_BATCHES_PER_EPOCH = 1  # 100
VAL_BATCHES_PER_EPOCH = 5
N_EPOCHS = 200


def visualise_batch(
    x: torch.Tensor, y: torch.Tensor, y_pred: torch.Tensor, n: int
) -> None:
    x, y, y_pred = x[:n], y[:n], y_pred[:n]
    fig, axs = plt.subplots()


def plot_losses(
    train_losses: list[float], val_losses: list[float], epochs: list[int]
) -> None:
    pass


def get_spatialized_features(
    x: torch.Tensor,
    img_w: int = 224,
    img_h: int = 224,
    patch_size: int = 16,
    stride: int = 16,
) -> torch.Tensor:
    feat_dict = featurizer.dinov2.forward_feats_attn(x, None, "none")  # type: ignore
    feats = feat_dict["x_norm_patchtokens"].to(torch.float)

    n_patch_w: int = 1 + (img_w - patch_size) // stride
    n_patch_h: int = 1 + (img_h - patch_size) // stride

    B, T, C = feats.shape

    f = feats.permute((0, 2, 1))  # swap C and T
    f = f.reshape((B, C, n_patch_h, n_patch_w))  # B, C, H, W
    f = torch.nn.functional.interpolate(f, (img_h, img_w))
    return f


def feed_batch(
    net: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: nn.CrossEntropyLoss,
    optim: torch.optim.Adam,
    train: bool = True,
) -> tuple[torch.Tensor, float]:
    optim.zero_grad()
    with torch.no_grad():
        feats = get_spatialized_features(x)
    y_pred = net(feats)
    loss = loss_fn(y_pred, y)

    if train:
        loss.backward()
        optim.step()
    return y_pred loss.item()


def do_epoch(
    net: nn.Module,
    n_batches: int,
    loader: DataLoader,
    loss_fn: nn.CrossEntropyLoss,
    optim: torch.optim.Adam,
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

        x = x.cpu()
        y = y.cpu()

        # add vis in here?
    return epoch_loss


for i in range(N_EPOCHS):
    epoch_loss = do_epoch(net, TRAIN_BATCHES_PER_EPOCH, train_loader, loss, optim, True)
    print(f"[{i}/{N_EPOCHS}]: {epoch_loss}")
