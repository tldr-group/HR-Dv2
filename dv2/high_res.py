# IMPORTS
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from os import getcwd

from typing import List, Literal, TypeAlias, Tuple

Neighbourhood: TypeAlias = Literal["Neumann", "Moore"]
PATH = getcwd()

# TRANSFORMS
IMG_SIZE: int = 518
RESIZE_DIM: int = IMG_SIZE + 2
PATCH_SIZE: int = 14
FEAT_DIM: int = 384  # for vits
PATCH_W: int = IMG_SIZE // PATCH_SIZE
PATCH_H: int = IMG_SIZE // PATCH_SIZE

transform = transforms.Compose(
    [
        transforms.Resize(RESIZE_DIM),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.2),
    ]
)

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


# MODULE
def svd_flip(u, v):
    # columns of u, rows of v
    max_abs_cols = torch.argmax(torch.abs(u), 0)
    i = torch.arange(u.shape[1]).to(u.device)
    signs = torch.sign(u[max_abs_cols, i])
    u *= signs
    v *= signs.view(-1, 1)
    return u, v


class PCA(nn.Module):
    # from user gngdb https://github.com/gngdb/pytorch-pca/blob/main/pca.py
    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components

    @torch.no_grad()
    def fit(self, X):
        n, d = X.size()
        if self.n_components is not None:
            d = min(self.n_components, d)
        self.register_buffer("mean_", X.mean(0, keepdim=True))
        Z = X - self.mean_  # center
        U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
        Vt = Vh
        U, Vt = svd_flip(U, Vt)
        self.register_buffer("components_", Vt[:d])
        return self

    @torch.no_grad()
    def forward(self, X):
        return self.transform(X)

    @torch.no_grad()
    def transform(self, X):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(X - self.mean_, self.components_.t())

    @torch.no_grad()
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    @torch.no_grad()
    def inverse_transform(self, Y):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(Y, self.components_) + self.mean_


# TODO: replace on set of loops by placing all masks into one kernel
# also only PCA *after* aggregating rather than before - as post processing step
class HighResDV2(nn.Module):
    def __init__(
        self,
        dino_name: str,
        shifts: List[int],
        pattern: Neighbourhood = "Neumann",
        n_components: int = 30,
        patch_size: int = 14,
    ) -> None:
        super().__init__()

        self.dinov2: nn.Module = torch.hub.load("facebookresearch/dinov2", dino_name)
        self.dinov2.eval()
        masks_tensors = self.compute_masks(pattern)
        # Make these masks parameters s.t they're moved to GPU
        self.masks = nn.ParameterList(
            [nn.Parameter(i, requires_grad=False) for i in masks_tensors]
        )
        self.shifts = shifts
        self.n_components = n_components
        self.PCA = PCA(n_components)

        self.patch_size: int = patch_size

    def compute_masks(self, pattern: Neighbourhood) -> List[torch.Tensor]:
        # Precompute neighbourhoods/weights for 3x3 conv2d
        out_masks: List[torch.Tensor] = []
        for i in range(9):
            # We need (out_ch, in_ch, h, w) for our weights dimension
            zero_mask = torch.zeros((3, 1, 3, 3))
            if pattern == "Neumann" and i % 2 == 1:
                zero_mask[:, :, i // 3, i % 3] = 1
                out_masks.append(zero_mask)
            elif pattern == "Moore":
                zero_mask[:, :, i // 3, i % 3] = 1
                out_masks.append(zero_mask)
        return out_masks

    def get_shifted_img_batch(self, x: torch.Tensor) -> torch.Tensor:
        # For each pixel shift and each direction in mask, compute conv2d of img with mask and store it
        img_list: List[torch.Tensor] = []
        for m in self.masks:
            for s in self.shifts:
                padded: torch.Tensor = F.pad(
                    x.unsqueeze(0), (s, s, s, s), mode="circular"
                )
                shifted_img = F.conv2d(
                    padded, m, stride=1, dilation=s, groups=3
                )  # groups=3
                img_list.append(shifted_img)
        img_batch = torch.cat(img_list)
        return img_batch

    def pca(self, f: torch.Tensor, norm: bool = False) -> torch.Tensor:
        # TODO: test if pytorhc's batched implementation would work better?
        # Use pytorch's PCA on x to get first k components. Data is not reshaped.
        projection = self.PCA.fit_transform(f)
        if norm:
            projection = F.normalize(projection)
        return projection

    def rescale_unshift_features(
        self, feature_batch: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        _, h, w = x.shape
        c = FEAT_DIM if self.n_components < 1 else self.n_components
        # Summand variable here to be memory efficient
        out_feature_img: torch.Tensor = torch.zeros(1, c, h, w, device=x.device)
        feat_idx: int = 0
        # TODO: make sure this reverse ordering works for moore neighbourhoods
        for m in self.masks[::-1]:
            for s in self.shifts:
                feat_patch_flat = feature_batch[feat_idx]
                # should I PCA before averaging or after?
                if self.n_components < 1:
                    reduced_dim_flat = feat_patch_flat
                else:
                    reduced_dim_flat = self.pca(feat_patch_flat)
                feat_patch = reduced_dim_flat.reshape((PATCH_H, PATCH_W, c))
                feat_img: torch.Tensor = torch.kron(
                    feat_patch, torch.ones((14, 14, 1), device=x.device)
                )
                full_size_tensor = feat_img.permute((2, 0, 1))
                # TODO: can i replace the kron with an interpolate
                # and/or can i get rid of all these resizes
                # feat_img: torch.Tensor = F.interpolate(feat_patch, size=(h, w))
                padded = F.pad(
                    full_size_tensor.unsqueeze(0), (s, s, s, s), mode="circular"
                )
                spatial_mask = m[0, 0, :, :]
                channel_mask = spatial_mask.unsqueeze(0).unsqueeze(0)
                # we want a depthwise convolution here i.e each bit of the input gets
                # its own filter and mapped to its own output channel
                channel_mask = channel_mask.repeat(c, 1, 1, 1)
                unshifted = F.conv2d(
                    padded, channel_mask, stride=1, dilation=s, groups=c
                )

                out_feature_img += unshifted
                feat_idx += 1

        n_imgs: int = feature_batch.shape[0]
        mean = out_feature_img / n_imgs
        return mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x.detach()
        with torch.no_grad():
            img_batch = self.get_shifted_img_batch(x)
            features_dict = self.dinov2.forward_features(img_batch)
            features_batch: torch.Tensor = features_dict["x_norm_patchtokens"]
            upsampled_unshifted = self.rescale_unshift_features(features_batch, x)
            return upsampled_unshifted


"""
plane_tensor, plane_img = load_image(f"{PATH}/dv2/plane.jpg", transform)
plane_tensor = plane_tensor.to("cuda")
print(plane_tensor.shape)

shifts = [2, 4, 7]  # [i for i in range(1, 8)]
net = HighResDV2("dinov2_vits14", shifts)
net.cuda()

features = net(plane_tensor)

print(features.shape)
"""
