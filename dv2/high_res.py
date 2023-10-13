# IMPORTS
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Literal, TypeAlias, Tuple

Neighbourhood: TypeAlias = Literal["Neumann", "Moore"]


class HighResDV2(nn.Module):
    def __init__(
        self,
        dino_name: str,
        shifts: List[int],
        pattern: Neighbourhood = "Neumann",
        dtype: torch.dtype = torch.float32,
        pca_each: bool = False,
    ) -> None:
        super().__init__()

        self.dinov2: nn.Module = torch.hub.load("facebookresearch/dinov2", dino_name)
        self.dinov2.eval()

        self.shift_directions = self.compute_shift_directions(pattern)
        self.shift_distances = shifts

        feat, patch = self.get_model_params(dino_name)
        self.patch_size: int = patch
        self.feat_dim: int = feat

        self.dtype = dtype

        self.pca_each = pca_each
        self.pca_k: int = 70

    def get_model_params(self, dino_name: str) -> Tuple[int, int]:
        # Match a name like dinov2_vits14 / dinov2_vitg16_lc etc. to feature dim and patch size.
        split_name = dino_name.split("_")
        model = split_name[1]
        arch, patch_size = model[3], int(model[4:])
        feat_dim_lookup = {"s": 384, "b": 768, "l": 1024, "g": 1536}
        feat_dim: int = feat_dim_lookup[arch]
        return feat_dim, patch_size

    def compute_shift_directions(self, pattern: Neighbourhood) -> List[Tuple[int, int]]:
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

    @torch.no_grad()
    def low_res_features(self, x: torch.Tensor) -> torch.Tensor:
        feat_dict = self.dinov2.forward_features(x)
        feat_tensor: torch.Tensor = feat_dict["x_norm_patchtokens"]
        if self.dtype != torch.float32:
            feat_tensor = feat_tensor.type(self.dtype)
        return feat_tensor

    @torch.no_grad()
    def get_shifted_img_batch(self, x: torch.Tensor) -> torch.Tensor:
        # For each shift direction and distance, torch.roll image for pixel shift w/ circular bcs
        img_list: List[torch.Tensor] = []
        for shift_dir in self.shift_directions:
            for shift_dist in self.shift_distances:
                shift = (shift_dir[0] * shift_dist, shift_dir[1] * shift_dist)
                shifted_img: torch.Tensor = torch.roll(x, shift, dims=(-2, -1))
                img_list.append(shifted_img)
        img_batch = torch.stack(img_list)
        if self.dtype != torch.float32:
            img_batch = img_batch.type(self.dtype)
        return img_batch

    @torch.no_grad()
    def pca(self, f: torch.Tensor, k: int) -> torch.Tensor:
        # Approximate (batched) tensor f with its k principal components computed over all b in f
        U, S, V = torch.pca_lowrank(f, q=k)
        projection = torch.matmul(f, V)
        return projection

    @torch.no_grad()
    def rescale_unshift_features(
        self, feature_batch: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        _, h, w = x.shape
        c = self.feat_dim if self.pca_each is False else self.pca_k
        n_patch_h, n_patch_w = h // self.patch_size, w // self.patch_size

        # Summand variable here to be memory efficient
        out_feature_img: torch.Tensor = torch.zeros(
            1, c, h, w, device=x.device, dtype=self.dtype
        )

        feat_idx: int = 0
        for shift_dir in self.shift_directions:
            for shift_dist in self.shift_distances:
                feat_patch_flat = feature_batch[feat_idx]
                if self.pca_each:
                    feat_patch_flat = self.pca(feat_patch_flat, self.pca_k)
                # reshape or view - view possibly more memory efficient
                feat_patch = feat_patch_flat.view((n_patch_h, n_patch_w, c))
                permuted = feat_patch.permute((2, 0, 1)).unsqueeze(0)

                full_size_tensor = F.interpolate(
                    permuted, scale_factor=self.patch_size, mode="nearest-exact"
                )

                rev_shift = (
                    -1 * shift_dir[0] * shift_dist,
                    -1 * shift_dir[1] * shift_dist,
                )
                unshifted = torch.roll(full_size_tensor, rev_shift, dims=(-2, -1))
                out_feature_img += unshifted
                feat_idx += 1

        n_imgs: int = feature_batch.shape[0]
        mean = out_feature_img / n_imgs
        return mean

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        img_batch = self.get_shifted_img_batch(x)
        features_batch = self.low_res_features(img_batch)
        upsampled_unshifted = self.rescale_unshift_features(features_batch, x)
        return upsampled_unshifted
