"""Patch the various methods of classes and subclasses in the Vision Transformer to add ]
new features like overlapping patches and attention visualisation"""

import torch
import torch.nn.functional as F
import math
import os

from typing import Tuple, Callable, TypeAlias, Literal

AttentionOptions: TypeAlias = Literal["q", "k", "v", "o", "none"]

# here to avoid syntax erros - checked already in DinoV2 code
XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
    else:
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False


def get_qkvo_per_head(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    x_a: torch.Tensor,
    which: AttentionOptions,
    drop_fn: Callable,
) -> torch.Tensor:
    """Return either the mean q, k, v per head for tokens or the attn for the CLS
    token per head. Note that using mem eff attn means we need to explicitly
    recompute the attn tensor. This is expensive and scales poorly with image
    resolution.

    https://github.com/facebookresearch/dinov2/pull/306
    https://github.com/facebookresearch/dinov2/issues/90
    https://github.com/facebookresearch/xformers/issues/730#issuecomment-1518740489

    Nt=N_tokens (patches, [CLS] and [REGs]), Nh=N_heads and C is output embed dim of
    model, for vit_tiny C=384, Nh=6

    :param q: query tensor, shape [B, Nt, Nh, C //Nh]
    :type q: torch.Tensor
    :param k: key tensor, shape [B, Nt, Nh, C //Nh]
    :type k: torch.Tensor
    :param v: value tensor, shape [B, Nt, Nh, C //Nh]
    :type v: torch.Tensor
    :param x_a: output of mem eff attn tensor
    :type x_a: torch.Tensor
    :param which: choice of which of qkvo to extract
    :type which: AttentionOptions
    :raises Exception: if "none" attention option passed
    :return: desired primitive of shape [B, Nt, Nh]
    :rtype: torch.Tensor
    """
    prims, mapping = [q, k, v], "qkv"
    match which:
        case "q" | "k" | "v":
            prim = prims[mapping.index(which)]
            per_head = prim.sum(dim=-1)
        case "o":
            B, T, nH, _ = q.shape
            # we only care about (and compute) attn of [CLS] token
            x_a_cls = x_a[:, 0, :, :]
            x_a_cls = x_a_cls[:, None, :, :]
            cls_attn = x_a_cls.permute(0, 2, 1, 3) @ v.permute(0, 2, 3, 1)
            cls_attn = cls_attn.squeeze(2).permute(0, 2, 1)
            per_head = cls_attn.reshape([B, T, nH])
        case _:
            raise Exception("not valid attention option")
    return per_head


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

    """The next 5 methods are taken from user 'legel' on GitHub https://github.com/facebookresearch/dinov2/pull/306,
    who's written a pull request for Dv2 to output attention maps. The maintainers suggest using a hook, which is
    another option. Instead of editing the Dv2 source, I've used the same MethodType assignment procedure as above."""

    @staticmethod
    def _fix__attn() -> Callable:
        """Replaces normal 'forward()' method of the attention layer (block.attn) in the Dv2 model with
        an optional early return with attention.

        :return: the new forward method
        :rtype: Callable
        """

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
        """Replaces normal 'forward()' method of the memory efficient attention layer (block.attn)
        in the Dv2 model with an optional early return with attention. Used if xformers used.

        :return: the new forward method
        :rtype: Callable
        """

        def forward(
            self,
            x: torch.Tensor,
            attn_bias=None,
            attn_choice: AttentionOptions = "none",
        ) -> torch.Tensor:
            if not XFORMERS_AVAILABLE:
                if attn_bias is not None:
                    raise AssertionError(
                        "xFormers is required for using nested tensors"
                    )
                return super().forward(x, return_attn)  # type: ignore
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

            q, k, v = unbind(qkv, 2)
            x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
            to_append: torch.Tensor
            if attn_choice != "none":
                to_append = get_qkvo_per_head(q, k, v, x, attn_choice, self.attn_drop)

            x = x.reshape([B, N, C])

            x = self.proj(x)
            x = self.proj_drop(x)

            if attn_choice != "none":
                x = torch.concat((x, to_append), dim=-1)
            return x

        return forward

    @staticmethod
    def _fix_block_forward_dino() -> Callable:
        """
        Replaces normal 'forward()' method of the block module to ensure the 'return_attn'
        flag is used in the attn forward layers.

        :return: the new forward method
        :rtype: Callable
        """

        def forward(
            self, x: torch.Tensor, attn_choice: AttentionOptions = "none"
        ) -> torch.Tensor:
            def attn_residual_func(x: torch.Tensor) -> torch.Tensor:
                # return self.ls1(self.attn(self.norm1(x)))
                return self.attn(self.norm1(x))

            def ffn_residual_func(x: torch.Tensor) -> torch.Tensor:
                # return self.ls2(self.mlp(self.norm2(x)))
                return self.mlp(self.norm2(x))

            if attn_choice != "none":
                nH = self.attn.num_heads
                ax = self.attn(self.norm1(x), attn_choice=attn_choice)
                xa, a = ax[:, :, :-nH], ax[:, :, -nH:]
                x = x + xa  # self.ls1(xa)
                x = x + ffn_residual_func(x)
                x = torch.concat((x, a), dim=-1)
                return x
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
    def _fix_block_forward_dv2() -> Callable:
        """
        Replaces normal 'forward()' method of the block module to ensure the 'return_attn'
        flag is used in the attn forward layers.

        :return: the new forward method
        :rtype: Callable
        """

        def forward(
            self, x: torch.Tensor, attn_choice: AttentionOptions = "none"
        ) -> torch.Tensor:
            def attn_residual_func(x: torch.Tensor) -> torch.Tensor:
                return self.ls1(self.attn(self.norm1(x)))

            def ffn_residual_func(x: torch.Tensor) -> torch.Tensor:
                return self.ls2(self.mlp(self.norm2(x)))

            if attn_choice != "none":
                nH = self.attn.num_heads
                ax = self.attn(self.norm1(x), attn_choice=attn_choice)
                xa, a = ax[:, :, :-nH], ax[:, :, -nH:]
                x = x + self.ls1(xa)
                x = x + ffn_residual_func(x)
                x = torch.concat((x, a), dim=-1)
                return x
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
        """Not used - assumes user isn't using a list"""

        def forward(self, x_or_x_list):
            if isinstance(x_or_x_list, torch.Tensor):
                return super().forward(x_or_x_list)  # type: ignore
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
    def _add_new_forward_features_dino() -> Callable:
        def forward_feats_attn(
            self, x, masks=None, attn_choice: AttentionOptions = "none"
        ):
            if isinstance(x, list):
                return self.forward_features_list(x, masks)

            x = self.prepare_tokens(x)  # prepare_tokens_with_masks(x, masks) for dv2

            for i, blk in enumerate(self.blocks):
                if i < len(self.blocks) - 1:
                    x = blk(x)
                else:
                    x = blk(x, attn_choice=attn_choice)

            if attn_choice != "none":
                x_feats = x[:, :, : -self.num_heads]
                # in our new function, the attn options are the last 6 channels of the features
                x_attn = x[:, :, -self.num_heads :]
            else:
                x_feats = x

            x_norm = self.norm(x_feats)
            out_dict = {
                "x_norm_clstoken": x_norm[:, 0],
                "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
                "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
                "x_prenorm": x,
                "masks": masks,
            }

            if attn_choice != "none":
                out_dict["x_patchattn"] = x_attn[:, self.num_register_tokens + 1 :]

            return out_dict

        return forward_feats_attn

    @staticmethod
    def _add_new_forward_features_dv2() -> Callable:
        def forward_feats_attn(
            self, x, masks=None, attn_choice: AttentionOptions = "none"
        ):
            if isinstance(x, list):
                return self.forward_features_list(x, masks)

            x = self.prepare_tokens_with_masks(x, masks)

            for i, blk in enumerate(self.blocks):
                if i < len(self.blocks) - 1:
                    x = blk(x)
                else:
                    x = blk(x, attn_choice=attn_choice)

            if attn_choice != "none":
                x_feats = x[:, :, : -self.num_heads]
                # in our new function, the attn options are the last 6 channels of the features
                x_attn = x[:, :, -self.num_heads :]
            else:
                x_feats = x

            x_norm = self.norm(x_feats)
            out_dict = {
                "x_norm_clstoken": x_norm[:, 0],
                "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
                "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
                "x_prenorm": x,
                "masks": masks,
            }

            if attn_choice != "none":
                out_dict["x_patchattn"] = x_attn[:, self.num_register_tokens + 1 :]

            return out_dict

        return forward_feats_attn

    @staticmethod
    def _add_new_forward_features_vit() -> Callable:
        def forward_feats_attn(
            self, x, masks=None, attn_choice: AttentionOptions = "none"
        ):
            B, nc, w, h = x.shape

            x = self.patch_embed(x)
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            # x = self.patch_drop(x)
            x = x + self.interpolate_pos_encoding(x, w, h)

            # x = self.prepare_tokens_with_masks(x, masks)

            for i, blk in enumerate(self.blocks):
                if i < len(self.blocks) - 1:
                    x = blk(x)
                else:
                    x = blk(x, attn_choice=attn_choice)

            if attn_choice != "none":
                x_feats = x[:, :, : -self.num_heads]
                # in our new function, the attn options are the last 6 channels of the features
                x_attn = x[:, :, -self.num_heads :]
            else:
                x_feats = x

            x_norm = self.norm(x_feats)
            out_dict = {
                "x_norm_clstoken": x_norm[:, 0],
                "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
                "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
                "x_prenorm": x,
                "masks": masks,
            }

            if attn_choice != "none":
                out_dict["x_patchattn"] = x_attn[:, self.num_register_tokens + 1 :]

            return out_dict

        return forward_feats_attn


# here to avoid syntax errors
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
