"""Patch the various methods of classes and subclasses in the Vision Transformer"""
import torch
import torch.nn.functional as F
import math
import os
import warnings

from typing import Tuple, Callable

XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
    else:
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False


class Patch:
    @staticmethod
    def _fix_pos_enc(patch_size: int, stride_hw: Tuple[int, int]) -> Callable:
        """Creates a method for position encoding interpolation, used to overwrite
        the original method in the DINO/DINOv2 vision transformer.
        Taken from https://github.com/ShirAmir/dino-vit-features/blob/main/extractor.py,
        added some bits from the Dv2 code in.

        :param patch_size: patch size of the model.
        :type patch_size: int
        :param stride_hw: A tuple containing the new height and width stride respectively.
        :type Tuple[int, int]
        :return: the interpolation method
        :rtype: Callable
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
            w0: float = 1 + (w - patch_size) // stride_hw[1]
            h0: float = 1 + (h - patch_size) // stride_hw[0]
            assert (
                w0 * h0 == npatch
            ), f"""got wrong grid size for {h}x{w} with patch_size {patch_size} and
            #                               stride {stride_hw} got {h0}x{w0}={h0 * w0} expecting {npatch}"""
            # we add a small number to avoid floating point error in the interpolation
            # see discussion at https://github.com/facebookresearch/dino/issues/8
            w0, h0 = w0 + 0.1, h0 + 0.1
            patch_pos_embed = F.interpolate(
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

    @staticmethod
    def _fix__attn() -> Callable:
        def forward(self, x: torch.Tensor, return_attn: bool = False) -> torch.Tensor:
            B, N, C = x.shape
            qkv = (
                self.qkv(x)
                .reshape(B, N, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )

            q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
            attn = q @ k.transpose(-2, -1)

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            if return_attn:
                return attn

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

        return forward

    @staticmethod
    def _fix_mem_eff_attn() -> Callable:
        def forward(
            self, x: torch.Tensor, attn_bias=None, return_attn: bool = False
        ) -> torch.Tensor:
            if not XFORMERS_AVAILABLE:
                if attn_bias is not None:
                    raise AssertionError(
                        "xFormers is required for using nested tensors"
                    )
                return super().forward(x)
                assert (
                    attn_bias is None
                ), "xFormers is required for nested tensors usage"
                return super().forward(x, return_attn)
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

            q, k, v = unbind(qkv, 2)

            x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
            if return_attn:
                attn = x.permute(0, 2, 1, 3) @ v.permute(0, 2, 3, 1)
                return attn
            x = x.reshape([B, N, C])

            x = self.proj(x)
            x = self.proj_drop(x)
            return x

        return forward

    @staticmethod
    def _fix_block_forward() -> Callable:
        def forward(self, x: torch.Tensor, return_attn: bool = False) -> torch.Tensor:
            def attn_residual_func(x: torch.Tensor) -> torch.Tensor:
                return self.ls1(self.attn(self.norm1(x)))

            def ffn_residual_func(x: torch.Tensor) -> torch.Tensor:
                return self.ls2(self.mlp(self.norm2(x)))

            if return_attn:
                return self.attn(self.norm1(x), return_attn=True)

            if self.training and self.sample_drop_ratio > 0.1:
                # the overhead is compensated only for a drop path rate larger than 0.1
                x = drop_add_residual_stochastic_depth(
                    x,
                    residual_func=attn_residual_func,
                    sample_drop_ratio=self.sample_drop_ratio,
                )
                x = drop_add_residual_stochastic_depth(
                    x,
                    residual_func=ffn_residual_func,
                    sample_drop_ratio=self.sample_drop_ratio,
                )
            elif self.training and self.sample_drop_ratio > 0.0:
                x = x + self.drop_path1(attn_residual_func(x))
                x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
            else:
                x = x + attn_residual_func(x)
                x = x + ffn_residual_func(x)
            return x

        return forward

    @staticmethod
    def _fix_nested_block_forward() -> Callable:
        def forward(self, x_or_x_list):
            if isinstance(x_or_x_list, torch.Tensor):
                return super().forward(x_or_x_list)
            elif isinstance(x_or_x_list, list):
                if not XFORMERS_AVAILABLE:
                    raise AssertionError(
                        "xFormers is required for using nested tensors"
                    )
                return self.forward_nested(x_or_x_list)
            else:
                raise AssertionError

        return forward

    @staticmethod
    def _add_forward_attn() -> Callable:
        def get_last_self_attention(self, x, masks=None):
            if isinstance(x, list):
                return self.forward_features_list(x, masks)

            x = self.prepare_tokens_with_masks(x, masks)

            # Run through model, at the last block just return the attention.
            for i, blk in enumerate(self.blocks):
                if i < len(self.blocks) - 1:
                    x = blk(x)
                else:
                    return blk(x, return_attn=True)

        return get_last_self_attention


def drop_add_residual_stochastic_depth(
    x: torch.Tensor,
    residual_func: Callable[[torch.Tensor], torch.Tensor],
    sample_drop_ratio: float = 0.0,
) -> torch.Tensor:
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    residual = residual_func(x_subset)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(
        x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor
    )
    return x_plus_residual.view_as(x)
