# ==================== IMPORTS ====================
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
from timm import create_model

from .patch import Patch

from types import MethodType
from .transform import iden_partial

from functools import partial
import math
from typing import List, Tuple, Callable, TypeAlias, Literal

Interpolation: TypeAlias = Literal[
    "nearest", "linear", "bilinear", "bicubic", "trilinear", "area", "nearest-exact"
]
AttentionOptions: TypeAlias = Literal["q", "k", "v", "o", "none"]


# ==================== MODULE ====================


class HighResDV2(nn.Module):
    def __init__(
        self,
        dino_name: str,
        stride: int,
        pca_dim: int = -1,
        dtype: torch.dtype = torch.float32,
        track_grad: bool = False,
    ) -> None:
        super().__init__()

        if "dinov2" in dino_name:
            hub_path = "facebookresearch/dinov2"
            self.dinov2: nn.Module = torch.hub.load(hub_path, dino_name)
        elif "dino" in dino_name:
            hub_path = "facebookresearch/dino:main"
            self.dinov2: nn.Module = torch.hub.load(hub_path, dino_name)
        elif "deit" in dino_name:
            self.dinov2: nn.Module = create_model(  # was 224
                "deit_small_patch16_224", pretrained=True
            )
        else:
            self.dinov2: nn.Module = create_model(  # was 224
                "vit_small_patch16_224", pretrained=True
            )

        # self.dinov2: nn.Module = torch.hub.load("facebookresearch/dinov2", dino_name)

        self.dinov2.eval()

        if "dinov2" not in dino_name:
            self.dinov2.num_heads = 6  # type: ignore
            self.dinov2.num_register_tokens = 0  # type: ignore

        # Get params of Dv2 model and store references to original settings & methods
        feat, patch = self.get_model_params(dino_name)
        self.original_patch_size: int = patch
        self.original_stride = _pair(patch)
        # may need to deepcopy this instead of just referencing
        # self.original_pos_enc = self.dinov2.interpolate_pos_encoding
        self.feat_dim: int = feat
        self.n_heads: int = 6
        self.n_register_tokens = 4

        self.stride = _pair(stride)
        # we need to set the stride to the original once before we set it to desired stride
        # i don't know why
        self.set_model_stride(self.dinov2, patch)
        self.set_model_stride(self.dinov2, stride)

        self.transforms: List[partial] = []
        self.inverse_transforms: List[partial] = []
        self.interpolation_mode: Interpolation = "nearest-exact"
        self.pca_dim = pca_dim
        self.do_pca = pca_dim > 3

        # If we want to save memory, change to float16
        self.dtype = dtype
        if dtype != torch.float32:
            self = self.to(dtype)
        self.track_grad = track_grad  # off by default to save memory

        self.patch_last_block(self.dinov2, dino_name)

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

        # if new_stride_pair == self.original_stride:
        # if resetting to original, return original method
        #    dino_model.interpolate_pos_encoding = self.original_pos_enc  # type: ignore
        # else:
        dino_model.interpolate_pos_encoding = MethodType(  # type: ignore
            Patch._fix_pos_enc(self.original_patch_size, new_stride_pair),
            dino_model,
        )  # typed ignored as they can't type check reassigned methods (generally is poor practice)

    def patch_last_block(self, dino_model: nn.Module, dino_name: str) -> None:
        """Patch the final block of the dino model to add attention return code.

        :param dino_model: DINO or DINOv2 model
        :type dino_model: nn.Module
        """
        final_block = dino_model.blocks[-1]  # type: ignore
        attn_block = final_block.attn  # type: ignore
        # hilariously this also works for dino i.e we can patch dino's attn block forward to
        # use the memeory efficienty attn like in dinov2
        attn_block.forward = MethodType(Patch._fix_mem_eff_attn(), attn_block)
        if "dinov2" in dino_name:
            final_block.forward = MethodType(Patch._fix_block_forward_dv2(), final_block)  # type: ignore
            dino_model.forward_feats_attn = MethodType(  # type: ignore
                Patch._add_new_forward_features_dv2(), dino_model
            )
        elif "dino" in dino_name:
            for i, blk in enumerate(dino_model.blocks):
                blk.forward = MethodType(Patch._fix_block_forward_dino(), blk)
                attn_block = blk.attn
                attn_block.forward = MethodType(Patch._fix_mem_eff_attn(), attn_block)
            final_block.forward = MethodType(Patch._fix_block_forward_dino(), final_block)  # type: ignore
            dino_model.forward_feats_attn = MethodType(  # type: ignore
                Patch._add_new_forward_features_dino(), dino_model
            )
        else:
            for i, blk in enumerate(dino_model.blocks):
                blk.forward = MethodType(Patch._fix_block_forward_dino(), blk)
                attn_block = blk.attn
                attn_block.forward = MethodType(Patch._fix_mem_eff_attn(), attn_block)
            final_block.forward = MethodType(Patch._fix_block_forward_dino(), final_block)  # type: ignore
            dino_model.forward_feats_attn = MethodType(  # type: ignore
                Patch._add_new_forward_features_vit(), dino_model
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
            # print("Warning: no transforms supplied, using identity transform")
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
    def invert_transforms(
        self, feature_batch: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """For each flat Dv2 features of our transformed imgs in $feature_batch, loop through,
        make them spatial again by reshaping, permuting and resizing, then perform the
        corresponding inverse transform and add to our summand variable. Finally we divide by
        N_imgs to create average.

        # TODO: parameterise this with the inverse transform s.t can just feed single batch (of
        say the attn map) and the iden partial transform and get upsampled

        :param feature_batch: batch of N_transform features from Dv2 with shape (n_patches, n_features)
        :type feature_batch: torch.Tensor
        :param x: original input img of size (channels, img_h, img_w), useful for resizing later
        :type x: torch.Tensor
        :return: feature image of size (n_features, img_h, img_w)
        :rtype: torch.Tensor
        """
        _, img_h, img_w = x.shape
        c = feature_batch.shape[-1]
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
        self,
        x: torch.Tensor,
        attn_choice: AttentionOptions = "none",
    ) -> torch.Tensor:
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
        out_dict = self.dinov2.forward_feats_attn(img_batch, None, attn_choice)  # type: ignore
        if attn_choice != "none":
            feats, attn = out_dict["x_norm_patchtokens"], out_dict["x_patchattn"]
            features_batch = torch.concat((feats, attn), dim=-1)
        else:
            features_batch = out_dict["x_norm_patchtokens"]

        if self.dtype != torch.float32:  # cast (i.e to f16)
            features_batch = features_batch.type(self.dtype)

        upsampled_features = self.invert_transforms(features_batch, x)
        return upsampled_features

    @torch.no_grad()
    def forward_sequential(
        self, x: torch.Tensor, attn_choice: AttentionOptions = "none"
    ) -> torch.Tensor:
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

        _, img_h, img_w = x.shape

        if attn_choice != "none":
            c = self.n_heads + self.feat_dim
        else:
            c = self.feat_dim

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
            out_dict = self.dinov2.forward_feats_attn(transformed_img, None, attn_choice)  # type: ignore
            if attn_choice != "none":
                feats, attn = out_dict["x_norm_patchtokens"], out_dict["x_patchattn"]
                features = torch.concat((feats, attn), dim=-1)
            else:
                features = out_dict["x_norm_patchtokens"]

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
        return mean


# from FeatUp: https://github.com/mhamilton723/FeatUp/blob/main/featup/util.py
class TorchPCA(object):

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        self.mean_ = X.mean(dim=0)
        unbiased = X - self.mean_.unsqueeze(0)
        U, S, V = torch.pca_lowrank(
            unbiased, q=self.n_components, center=False, niter=4
        )
        self.components_ = V.T
        self.singular_values_ = S
        return self

    def transform(self, X):
        t0 = X - self.mean_.unsqueeze(0)
        projected = t0 @ self.components_.T
        return projected


def torch_pca(
    feature_img: torch.Tensor,
    dim: int = 128,
    fit_pca=None,
    max_samples: int = 20000,
):
    device = feature_img[0].device
    C, H, W = feature_img.shape
    N = H * W
    flat = feature_img.reshape(C, N).permute(1, 0)

    if max_samples is not None and N > max_samples:
        indices = torch.randperm(N)[:max_samples]
        sample = flat[indices].to(torch.float32)
    else:
        sample = flat.to(torch.float32)

    if fit_pca is None:
        fit_pca = TorchPCA(dim)
        fit_pca.fit(sample)
    transformed = fit_pca.transform(flat)
    return transformed
