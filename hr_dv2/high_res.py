# ==================== IMPORTS ====================
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
import torch.nn.functional as F

from .patch import Patch

from types import MethodType
from .transform import iden_partial

from functools import partial
import math
from typing import List, Tuple, Callable, TypeAlias, Literal

Interpolation: TypeAlias = Literal[
    "nearest", "linear", "bilinear", "bicubic", "trilinear", "area", "nearest-exact"
]


# ==================== MODULE ====================


class HighResDV2(nn.Module):
    def __init__(
        self,
        dino_name: str,
        stride: int,
        sequential: bool = False,
        dtype: torch.dtype = torch.float32,
        track_grad: bool = False,
    ) -> None:
        super().__init__()

        self.dinov2: nn.Module = torch.hub.load("facebookresearch/dinov2", dino_name)
        self.dinov2.eval()

        # Get params of Dv2 model and store references to original settings & methods
        feat, patch = self.get_model_params(dino_name)
        self.original_patch_size: int = patch
        self.original_stride = _pair(patch)
        # may need to deepcopy this instead of just referencing
        self.original_pos_enc = self.dinov2.interpolate_pos_encoding
        self.feat_dim: int = feat
        self.n_heads: int = 6

        self.stride = _pair(stride)
        self.set_model_stride(self.dinov2, stride)

        self.transforms: List[partial] = []
        self.inverse_transforms: List[partial] = []
        self.sequential = sequential
        self.interpolation_mode: Interpolation = "nearest-exact"

        # If we want to save memory, change to float16
        self.dtype = dtype
        if dtype != torch.float32:
            self = self.to(dtype)
        self.track_grad = track_grad  # off by default to save memory

        self.patch_last_block(self.dinov2)

    def get_model_params(self, dino_name: str) -> Tuple[int, int]:
        """Match a name like dinov2_vits14 / dinov2_vitg16_lc etc. to feature dim and patch size.

        :param dino_name: string of dino model name on torch hub
        :type dino_name: str
        :return: tuple of original patch size and hidden feature dimension
        :rtype: Tuple[int, int]
        """
        split_name = dino_name.split("_")
        model = split_name[1]
        arch, patch_size = model[3], int(model[4:])
        feat_dim_lookup = {"s": 384, "b": 768, "l": 1024, "g": 1536}
        feat_dim: int = feat_dim_lookup[arch]
        return feat_dim, patch_size

    def set_model_stride(
        self, dino_model: nn.Module, stride_l: int, verbose: bool = False
    ) -> None:
        """Create new positional encoding interpolation method for $dino_model with
        supplied $stride, and set the stride of the patch embedding projection conv2D
        to $stride.

        :param dino_model: Dv2 model
        :type dino_model: DinoVisionTransformer
        :param new_stride: desired stride, usually stride < original_stride for higher res
        :type new_stride: int
        :return: None
        :rtype: None
        """

        new_stride_pair = torch.nn.modules.utils._pair(stride_l)
        if new_stride_pair == self.stride:
            return  # early return as nothing to be done
        self.stride = new_stride_pair
        dino_model.patch_embed.proj.stride = new_stride_pair  # type: ignore
        if verbose:
            print(f"Setting stride to ({stride_l},{stride_l})")

        if new_stride_pair == self.original_stride:
            # if resetting to original, return original method
            dino_model.interpolate_pos_encoding = self.original_pos_enc  # type: ignore
        else:
            dino_model.interpolate_pos_encoding = MethodType(  # type: ignore
                Patch._fix_pos_enc(self.original_patch_size, new_stride_pair),
                dino_model,
            )  # typed ignored as they can't type check reassigned methods (generally is poor practice)

    def patch_last_block(self, dino_model: nn.Module) -> None:
        """Patch the final block of the dino model to add attention return code.

        :param dino_model: DINO or DINOv2 model
        :type dino_model: nn.Module
        """
        final_block = dino_model.blocks[-1]  # type: ignore
        attn_block = final_block.attn  # type: ignore
        attn_block.forward = MethodType(Patch._fix_mem_eff_attn(), attn_block)
        final_block.forward = MethodType(Patch._fix_block_forward(), final_block)  # type: ignore
        dino_model.get_last_self_attention = MethodType(  # type: ignore
            Patch._add_forward_attn(), dino_model
        )

    def get_n_patches(self, img_h: int, img_w: int) -> Tuple[int, int]:
        stride_l = self.stride[0]
        n_patch_h: int = 1 + (img_h - self.original_patch_size) // stride_l
        n_patch_w: int = 1 + (img_w - self.original_patch_size) // stride_l
        return (n_patch_h, n_patch_w)

    def set_transforms(
        self, transforms: List[partial], inv_transforms: List[partial]
    ) -> None:
        assert len(transforms) == len(
            inv_transforms
        ), "Each transform must have an inverse!"
        self.transforms = transforms
        self.inverse_transforms = inv_transforms

    @torch.no_grad()
    def get_transformed_input_batch(
        self, x: torch.Tensor, transforms: List[partial]
    ) -> torch.Tensor:
        """Loop through a list of (invertible) transforms, apply them to input $x, store
        in a list then batch and return.

        :param x: input unbatched standardized image tensor
        :type x: torch.Tensor
        :param transforms: list of partial functions representing transformations on image
        :type transforms: List[partial]
        :return: batch of transformed images
        :rtype: torch.Tensor
        """
        img_list: List[torch.Tensor] = []
        if len(transforms) == 0:  # if we want to test VE by itself
            print("Warning: no transforms supplied, using identity transform")
            self.transforms.append(iden_partial)
            self.inverse_transforms.append(iden_partial)

        for transform in transforms:
            transformed_img: torch.Tensor = transform(x)
            img_list.append(transformed_img)

        if len(img_list[0].shape) == 3:  # if c, h, w then stack list
            img_batch = torch.stack(img_list)
        else:  # if b, c, h, w then cat
            img_batch = torch.cat(img_list)

        if self.dtype != torch.float32:
            img_batch = img_batch.to(self.dtype)
        return img_batch

    @torch.no_grad()
    def get_dv2_features(self, x: torch.Tensor, stride_l: int = -1) -> torch.Tensor:
        """Feed batched img tensor $x into DINOv2, optionally set stride and return features.

        :param x: batched img tensor, shape [B, C, H, W]
        :type x: torch.Tensor
        :param stride: desired stride/resolution for VE, defaults to -1
        :type stride: int, optional
        :return: features extracted by DINOv2
        :rtype: torch.Tensor
        """
        if self.dtype != torch.float32:
            x = x.to(self.dtype)

        if stride_l > 0:  # if we don't want to change stride. Assumes square stride
            self.set_model_stride(self.dinov2, stride_l)
        feat_dict = self.dinov2.forward_features(x)  # type: ignore
        feat_tensor: torch.Tensor = feat_dict["x_norm_patchtokens"]
        if self.dtype != torch.float32:
            feat_tensor = feat_tensor.to(self.dtype)
        return feat_tensor

    @torch.no_grad()
    def get_dv2_attn(
        self,
        x: torch.Tensor,
        stride_l: int = -1,
        which: Literal["cls", "reg", "both"] = "cls",
    ) -> torch.Tensor:
        """Feed batched img tensor $x into DINOv2, optionally set stride and return features.

        :param x: batched img tensor, shape [B, C, H, W]
        :type x: torch.Tensor
        :param stride: desired stride/resolution for VE, defaults to -1
        :type stride: int, optional
        :return: features extracted by DINOv2
        :rtype: torch.Tensor
        """
        if self.dtype != torch.float32:
            x = x.to(self.dtype)

        if stride_l > 0:  # if we don't want to change stride. Assumes square stride
            self.set_model_stride(self.dinov2, stride_l)
        attention: torch.Tensor = self.dinov2.get_last_self_attention(x)  # type: ignore

        n_register_tokens = 4  # change this for ViTs
        b, n_heads, n_tokens, _ = attention.shape
        s0: int = 0
        s1: int = 1
        if which == "reg":
            s0 = 1
            s1 = 1 + n_register_tokens
        elif which == "both":
            s1 = 1 + n_register_tokens
        delta = s1 - s0
        # attention tokens are packed in after the first token; the spatial tokens follow
        attention = attention[:, :, s0:s1, 1 + n_register_tokens :].reshape(
            b, n_heads * delta, -1
        )
        attention = torch.permute(attention, (0, 2, 1))
        if self.dtype != torch.float32:
            attention = attention.to(self.dtype)
        return attention

    @torch.no_grad()
    def invert_transforms(
        self, feature_batch: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """For each flat Dv2 features of our transformed imgs in $feature_batch, loop through,
        make them spatial again by reshaping, permuting and resizing, then perform the
        corresponding inverse transform and add to our summand variable. Finally we divide by
        N_imgs to create average.

        :param feature_batch: batch of N_transform features from Dv2 with shape (n_patches, n_features)
        :type feature_batch: torch.Tensor
        :param x: original input img of size (channels, img_h, img_w), useful for resizing later
        :type x: torch.Tensor
        :return: feature image of size (n_features, img_h, img_w)
        :rtype: torch.Tensor
        """
        _, img_h, img_w = x.shape
        c = feature_batch.shape[-1]  # self.feat_dim
        # print(feature_batch.shape)
        stride_l = self.stride[0]
        n_patch_w: int = 1 + (img_w - self.original_patch_size) // stride_l
        n_patch_h: int = 1 + (img_h - self.original_patch_size) // stride_l

        # Summand variable here to be memory efficient
        out_feature_img: torch.Tensor = torch.zeros(
            1,
            c,
            img_h,
            img_w,
            device=x.device,
            dtype=self.dtype,
            requires_grad=self.track_grad,
        )

        for i, inv_transform in enumerate(self.inverse_transforms):
            feat_patch_flat = feature_batch[i]
            # interp expects batched spatial tensors so reshape and unsqueeze
            feat_patch = feat_patch_flat.view((n_patch_h, n_patch_w, c))
            permuted = feat_patch.permute((2, 0, 1)).unsqueeze(0)

            full_size = F.interpolate(
                permuted,
                (img_h, img_w),
                mode=self.interpolation_mode,
            )
            inverted: torch.Tensor = inv_transform(full_size)
            out_feature_img += inverted

        n_imgs: int = feature_batch.shape[0]
        mean = out_feature_img / n_imgs
        return mean

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, attn: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Feed input img $x through network and get low and high res features.

        :param x: unbatched image tensor
        :type x: torch.Tensor
        :return: tuple of low-res Dv2 features and our upsample high-res Dv2 features
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        x.requires_grad = self.track_grad
        if self.dtype != torch.float32:  # cast (i.e to f16)
            x = x.type(self.dtype)
        img_batch = self.get_transformed_input_batch(x, self.transforms)

        # this is unweildy, might need a change
        temp_stride = self.stride

        low_res_features = self.get_dv2_features(
            x.unsqueeze(0), self.original_stride[0]
        )
        if attn is False:
            features_batch = self.get_dv2_features(img_batch, temp_stride[0])
        else:
            features_batch = self.get_dv2_attn(img_batch, temp_stride[0])
        upsampled_features = self.invert_transforms(features_batch, x)
        return upsampled_features, low_res_features

    @torch.no_grad()
    def forward_sequential(
        self, x: torch.Tensor, attn: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform transform -> featurise -> upscale -> inverse -> average forward pass
        sequentially, performing more calls to DINOv2 but reducing the memory overhead.

        :param x: unbatched image tensor
        :type x: torch.Tensor
        :return: tuple of low-res Dv2 features and our upsample high-res Dv2 features
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        x.requires_grad = self.track_grad
        if self.dtype != torch.float32:  # cast (i.e to f16)
            x = x.type(self.dtype)
        img_batch = self.get_transformed_input_batch(x, self.transforms)
        temp_stride = self.stride

        low_res_features = self.get_dv2_features(
            x.unsqueeze(0), self.original_stride[0]
        )

        _, img_h, img_w = x.shape
        c = self.n_heads if attn else self.feat_dim
        stride_l = temp_stride[0]
        n_patch_w: int = 1 + (img_w - self.original_patch_size) // stride_l
        n_patch_h: int = 1 + (img_h - self.original_patch_size) // stride_l

        out_feature_img: torch.Tensor = torch.zeros(
            1,
            c,
            img_h,
            img_w,
            device=x.device,
            dtype=self.dtype,
            requires_grad=self.track_grad,
        )

        N_transforms = len(self.transforms)
        for i in range(N_transforms):
            transformed_img = img_batch[i].unsqueeze(0)
            if attn is False:
                features = self.get_dv2_features(transformed_img, temp_stride[0])
            else:
                features = self.get_dv2_attn(transformed_img, temp_stride[0])
            features = features.squeeze(0)
            feat_patch = features.view((n_patch_h, n_patch_w, c))
            permuted = feat_patch.permute((2, 0, 1)).unsqueeze(0)

            full_size = F.interpolate(
                permuted,
                (img_h, img_w),
                mode=self.interpolation_mode,
            )
            inv_transform = self.inverse_transforms[i]
            inverted: torch.Tensor = inv_transform(full_size)
            out_feature_img += inverted

        mean = out_feature_img / N_transforms
        return mean, low_res_features

    @torch.no_grad()
    def pca(self, f: torch.Tensor, k: int) -> torch.Tensor:
        """Approximate (batched) tensor $f with its $k principal components computed over
        all batches in f.

        :param f: tensor of features. Can be flat or spatial, but flat preferred
        :type f: torch.Tensor
        :param k: number of principal components to use
        :type k: int
        :return: $k component PCA of $f
        :rtype: torch.Tensor
        """
        U, S, V = torch.pca_lowrank(f, q=k)
        projection = torch.matmul(f, V)
        return projection
