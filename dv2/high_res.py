# IMPORTS
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from types import MethodType
import math

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
        track_grad: bool = False,
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
        if dtype != torch.float32:
            self = self.to(dtype)
        self.track_grad = track_grad

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
        if self.dtype != torch.float32:
            x = x.to(self.dtype)
        feat_dict = self.dinov2.forward_features(x)
        feat_tensor: torch.Tensor = feat_dict["x_norm_patchtokens"]
        if self.dtype != torch.float32:
            feat_tensor = feat_tensor.to(self.dtype)
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
            img_batch = img_batch.to(self.dtype)
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
        n_samples = math.floor(math.sqrt(feature_batch.shape[1]))
        n_patch_h, n_patch_w = n_samples, n_samples

        # Summand variable here to be memory efficient
        out_feature_img: torch.Tensor = torch.zeros(
            1, c, h, w, device=x.device, dtype=self.dtype, requires_grad=self.track_grad
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
                    permuted,
                    (h, w),
                    mode="nearest-exact",  # scale_factor=self.patch_size
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
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x.requires_grad = self.track_grad
        if self.dtype != torch.float32:  # cast (i.e to f16)
            x = x.type(self.dtype)
        img_batch = self.get_shifted_img_batch(x)

        features_batch = self.low_res_features(img_batch)
        low_res_features = self.low_res_features(x.unsqueeze(0))

        upsampled_unshifted = self.rescale_unshift_features(features_batch, x)
        return upsampled_unshifted, low_res_features

    @staticmethod
    def _fix_pos_enc(patch_size: int, stride_hw: Tuple[int, int]):
        """
        Creates a method for position encoding interpolation.
        :param patch_size: patch size of the model.
        :param stride_hw: A tuple containing the new height and width stride respectively.
        :return: the interpolation method
        """

        def interpolate_pos_encoding(
            self, x: torch.Tensor, w: int, h: int
        ) -> torch.Tensor:
            previous_dtype = x.dtype
            npatch = x.shape[1] - 1
            N = self.pos_embed.shape[1] - 1
            if npatch == N and w == h:
                return self.pos_embed
            pos_embed = self.pos_embed.float()
            class_pos_embed = pos_embed[:, 0]
            patch_pos_embed = pos_embed[:, 1:]
            dim = x.shape[-1]
            # compute number of tokens taking stride into account
            w0 = 1 + (w - patch_size) // stride_hw[1]
            h0 = 1 + (h - patch_size) // stride_hw[0]
            # assert (
            #    w0 * h0 == npatch
            # ), f"""got wrong grid size for {h}x{w} with patch_size {patch_size} and
            #                               stride {stride_hw} got {h0}x{w0}={h0 * w0} expecting {npatch}"""
            # we add a small number to avoid floating point error in the interpolation
            # see discussion at https://github.com/facebookresearch/dino/issues/8
            w0, h0 = w0 + 0.1, h0 + 0.1
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(
                    1, int(math.sqrt(N)), int(math.sqrt(N)), dim
                ).permute(0, 3, 1, 2),
                scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                mode="bicubic",
                align_corners=False,
                recompute_scale_factor=False,
            )
            assert (
                int(w0) == patch_pos_embed.shape[-2]
                and int(h0) == patch_pos_embed.shape[-1]
            )
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(
                previous_dtype
            )

        return interpolate_pos_encoding

    def make_high_res(self, dino_model: nn.Module, new_stride: int) -> None:
        new_stride_pair = torch.nn.modules.utils._pair(new_stride)
        dino_model.patch_embed.proj.stride = new_stride_pair  # type: ignore
        dino_model.interpolate_pos_encoding = MethodType(
            HighResDV2._fix_pos_enc(self.patch_size, new_stride_pair), dino_model
        )
