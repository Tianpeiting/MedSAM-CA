# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .common import LayerNorm2d
import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Tuple, Type
from .common import LayerNorm2d


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        cnn_dims: List[int] = [64, 128, 256, 512],  # 设置默认值 # CNN 特征的通道数列表
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        initial_bias: float = 0.7,  # 新增参数，用于设置初始偏置因子
        # initial_bias: float = 0.6,  # 新增参数，用于设置初始偏置因子
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.cnn_dims = cnn_dims or [64, 128, 256, 512]  # 如果未传递，则使用默认值
        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        # 上采样模块，支持跳跃连接
        self.output_upscaling = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(
                    transformer_dim, transformer_dim // 2, kernel_size=2, stride=2
                ),
                LayerNorm2d(transformer_dim // 2),
                activation(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(
                    transformer_dim // 2, transformer_dim // 4, kernel_size=2, stride=2
                ),
                LayerNorm2d(transformer_dim // 4),
                activation(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(
                    transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
                ),
                LayerNorm2d(transformer_dim // 8),
                activation(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(
                    transformer_dim // 8, transformer_dim // 8, kernel_size=2, stride=2
                ),
                LayerNorm2d(transformer_dim // 8),
                activation(),
            ),
        ])

        # 注意力融合模块（每个上采样阶段一个）
        self.attention_fusion = nn.ModuleList([
            MultiHeadAttentionFusion(
                cnn_channels=cnn_dim, 
                decoder_channels=transformer_dim // (2 ** i), 
                num_heads=4, 
                initial_bias=initial_bias
            )
            # AttentionFusionWithBias(cnn_dim, transformer_dim // (2 ** i), initial_bias=initial_bias)
            for i, cnn_dim in enumerate(reversed(cnn_dims))  # [512, 256, 128, 64]
        ])

        # # 注意力融合模块（每个上采样阶段一个）
        # self.attention_fusion = nn.ModuleList([
        #     AdditiveFusion(
        #         cnn_channels=cnn_dim,
        #         decoder_channels=transformer_dim // (2 ** i)
        #     )
        #     for i, cnn_dim in enumerate(reversed(cnn_dims))  # [512, 256, 128, 64]
        # ])

        for i, cnn_dim in enumerate(reversed(self.cnn_dims)):  # 从 scale4 到 scale1
            expected_transformer_dim = self.transformer_dim // (2 ** i)

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        cnn_features: List[torch.Tensor],  # CNN 提取的多尺度特征
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            cnn_features=cnn_features,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        cnn_features: List[torch.Tensor],  # CNN 多尺度特征
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0
        )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        if image_embeddings.shape[0] != tokens.shape[0]:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            src = image_embeddings
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        # 跳跃连接：将 CNN 特征直接传递到解码器
        # 跳跃连接 + 注意力融合
        for i, (upsample, fusion) in enumerate(zip(self.output_upscaling, self.attention_fusion)):
            # 对齐 CNN 特征的分辨率
            cnn_aligned = F.interpolate(cnn_features[3 - i], size=src.shape[-2:], mode='bilinear', align_corners=False)

            # 注意力融合
            src = fusion(cnn_aligned, src)
            # 上采样
            src = upsample(src)  # 将特征分辨率放大一倍

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = src.shape
        masks = (hyper_in @ src.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


class MultiHeadAttentionFusion(nn.Module):
    def __init__(self, cnn_channels, decoder_channels, num_heads=4, initial_bias=0.7):
        super(MultiHeadAttentionFusion, self).__init__()
        self.num_heads = num_heads

        # 调整 CNN 特征的通道数
        self.cnn_channel_adjust = nn.Conv2d(cnn_channels, decoder_channels, kernel_size=1)

        # 注意力头
        self.attention_heads = nn.ModuleList([
            nn.Conv2d(decoder_channels + decoder_channels, 1, kernel_size=1)
            for _ in range(num_heads)
        ])
        self.sigmoid = nn.Sigmoid()
        self.bias_factor = nn.Parameter(torch.tensor(initial_bias))

    def forward(self, cnn_features, decoder_features):
        # 调整 CNN 特征通道数
        cnn_features = self.cnn_channel_adjust(cnn_features)

        # 拼接特征
        combined_features = torch.cat([cnn_features, decoder_features], dim=1)

        # 计算注意力权重
        attention_weights = [self.sigmoid(head(combined_features)) for head in self.attention_heads]
        attention_weights = torch.stack(attention_weights, dim=0).mean(dim=0)

        # 融合特征
        fused_features = attention_weights * decoder_features + (1 - attention_weights) * cnn_features
        return fused_features

# class AdditiveFusion(nn.Module):
#     def __init__(self, cnn_channels, decoder_channels):
#         super(AdditiveFusion, self).__init__()
#         self.cnn_channel_adjust = nn.Conv2d(cnn_channels, decoder_channels, kernel_size=1)

#     def forward(self, cnn_features, decoder_features):
#         cnn_features = F.interpolate(
#             cnn_features,
#             size=decoder_features.shape[-2:],
#             mode="bilinear",
#             align_corners=False
#         )
#         cnn_features = self.cnn_channel_adjust(cnn_features)
#         return decoder_features + cnn_features

# class AttentionFusionWithBias(nn.Module):
#     def __init__(self, cnn_channels, decoder_channels, initial_bias=0.7):
#         """
#         注意力融合 + 偏置因子模块
#         Args:
#             cnn_channels (int): CNN 特征的通道数
#             decoder_channels (int): 解码器特征的通道数
#             initial_bias (float): 初始化的偏置因子，越接近 1，则越偏向解码器特征
#         """
#         super(AttentionFusionWithBias, self).__init__()
#         # 用于生成注意力权重的 1x1 卷积
#         self.attention_conv = nn.Conv2d(cnn_channels + decoder_channels, 1, kernel_size=1)
#         self.sigmoid = nn.Sigmoid()
#         # self.feature_align = FeatureAlignment(cnn_channels, decoder_channels)

#         # 可学习的偏置因子，初始值为 initial_bias
#         self.bias_factor = nn.Parameter(torch.tensor(initial_bias))

#     def forward(self, cnn_features, decoder_features):
#         """
#         Args:
#             cnn_features: 来自 CNN 的特征 (B, C_cnn, H, W)
#             decoder_features: 来自解码器的特征 (B, C_decoder, H, W)
#         Returns:
#             融合后的特征 (B, C_decoder, H, W)
#         """
#         # 1. 拼接 CNN 特征和解码器特征
#         # cnn_features = self.feature_align(cnn_features)
#         combined_features = torch.cat([cnn_features, decoder_features], dim=1)  # (B, C_cnn + C_decoder, H, W)

#         # 2. 计算注意力权重
#         attention_weights = self.sigmoid(self.attention_conv(combined_features))  # (B, 1, H, W)

#         # 3. 引入偏置因子调整注意力权重
#         adjusted_weights = attention_weights * (1 - self.bias_factor) + self.bias_factor

#         # 4. 加权融合 CNN 特征和解码器特征
#         fused_features = adjusted_weights * decoder_features + (1 - adjusted_weights) * cnn_features
#         return fused_features

# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x



