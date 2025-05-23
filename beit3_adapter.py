# Modified from https://github.com/microsoft/torchscale/blob/main/torchscale/model/BEiT3.py
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from torch.nn.init import normal_
from timm.models.layers import trunc_normal_, LayerNorm2d

from torchscale.model.BEiT3 import BEiT3
from torchscale.architecture.config import EncoderConfig
from torchscale.component.multiway_network import set_split_position

from adapter_modules import SpatialPriorModule, deform_inputs
from adapter_modules import InteractionBlockWithClsAndMultiWay as InteractionBlock
from adapter_modules import AdapterMultiscaleDeformableAttention

def block_decorator(f):
    def wrapper(x, H, W):
        x, l_aux = f(x)
        return x
    return wrapper

class BEiT3Adapter(BEiT3):
    def __init__(self, beit3_args, conv_inplane=64, n_points=4,
                 deform_num_heads=6, init_values=0., interaction_indexes=None, with_cffn=True,
                 cffn_ratio=0.25, deform_ratio=1.0, add_vit_feature=True, pretrained=None,
                 use_extra_extractor=True, with_cp=None, asymetric_input=False, beit_resolution=None,
                 intepolate_pos=False, num_segments=None, **kwargs):
        
        super().__init__(beit3_args)

        if num_segments is not None:
            self.segment_embed = nn.Embedding(num_segments, beit3_args.encoder_embed_dim)
        
        self.asymetric_input = asymetric_input
        self.beit_resolution = beit_resolution
        self.intepolate_pos = intepolate_pos
        
        with_cp = with_cp if with_cp is not None else beit3_args.checkpoint_activations
        
        self.norm_layer = LayerNorm
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        self.embed_dim = beit3_args.encoder_embed_dim
        embed_dim = beit3_args.encoder_embed_dim
        
        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=embed_dim, with_cp=False)
        self.interactions = nn.Sequential(*[
            InteractionBlock(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
                             init_values=init_values, drop_path=beit3_args.drop_path_rate,
                             norm_layer=self.norm_layer, with_cffn=with_cffn,
                             cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                             extra_extractor=((True if i == len(interaction_indexes) - 1
                                               else False) and use_extra_extractor),
                             with_cp=with_cp)
            for i in range(len(interaction_indexes))
        ])
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.norm1 = LayerNorm2d(embed_dim)
        self.norm2 = LayerNorm2d(embed_dim)
        self.norm3 = LayerNorm2d(embed_dim)
        self.norm4 = LayerNorm2d(embed_dim)
        
        self.init_weights()
        
    def init_weights(self):
        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _init_deform_weights(self, m):
        if isinstance(m, AdapterMultiscaleDeformableAttention):
            m._reset_parameters()
            
    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4
    
    def forward_encoder(
        self,
        src_tokens,
        encoder_padding_mask=None,
        attn_mask=None,
        return_all_hiddens=False,
        token_embeddings=None,
        multiway_split_position=None,
        features_only=False,
        incremental_state=None,
        positions=None,
        **kwargs
    ):
        assert src_tokens is not None or token_embeddings is not None

        if encoder_padding_mask is None:
            if src_tokens is not None:
                encoder_padding_mask = torch.zeros_like(
                    src_tokens, device=src_tokens.device
                ).bool()
            else:
                encoder_padding_mask = torch.zeros(
                    [token_embeddings.size(0), token_embeddings.size(1)],
                    device=token_embeddings.device,
                ).bool()

        if multiway_split_position is not None:
            assert self.encoder.args.multiway
            self.encoder.apply(set_split_position(multiway_split_position))

        x, encoder_embedding = self.encoder.forward_embedding(src_tokens, token_embeddings, positions)
        x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        encoder_states = []

        if return_all_hiddens:
            encoder_states.append(x)
        
        rel_pos_bias = None
        if self.encoder.relative_position is not None:
            rel_pos_bias = self.encoder.relative_position(
                batch_size=x.size(0), qlen=x.size(1), klen=x.size(1)
            )

        # incremental_state is not None during inference if we use the bidirectional encoder as a generator as in s2s-ft (https://arxiv.org/abs/2110.13640)
        l_aux = []
        for idx, layer in enumerate(self.encoder.layers):
            x, l_aux_i = layer(
                x,
                encoder_padding_mask=encoder_padding_mask if incremental_state is None else None,
                attn_mask=attn_mask,
                rel_pos=rel_pos_bias,
                multiway_split_position=multiway_split_position,
                incremental_state=incremental_state[idx] if incremental_state is not None else None,
            )
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)
            l_aux.append(l_aux_i)

        if self.encoder.layer_norm is not None:
            x = self.encoder.layer_norm(x)
            
        if not features_only and self.encoder.output_projection is not None:
            x = self.encoder.output_projection(x)

        return {
            "encoder_out": x,
            "encoder_embedding": encoder_embedding,
            "encoder_padding_mask": encoder_padding_mask,
            "encoder_states": encoder_states,
            "l_aux": l_aux,
        }
    
    def forward_beit3(
        self,
        textual_tokens=None,
        visual_tokens=None,
        text_padding_position=None,
        attn_mask=None,
        vision_masked_position=None,
        incremental_state=None,
        positions=None,
    ):
        assert textual_tokens is not None or visual_tokens is not None

        if textual_tokens is None:
            x = self.vision_embed(visual_tokens, vision_masked_position)
            encoder_padding_mask = None
            multiway_split_position = -1
        elif visual_tokens is None:
            x = self.text_embed(textual_tokens)
            encoder_padding_mask = text_padding_position
            multiway_split_position = 0
        else:
            x1 = self.vision_embed(visual_tokens, vision_masked_position)
            multiway_split_position = x1.size(1)
            x2 = self.text_embed(textual_tokens)
            x = torch.cat([x1, x2], dim=1)

            if text_padding_position is not None:
                encoder_padding_mask = torch.cat(
                    [
                        torch.zeros(x1.shape[:-1]).to(x1.device).bool(),
                        text_padding_position,
                    ],
                    dim=1,
                )
            else:
                encoder_padding_mask = None

        encoder_out = self.forward_encoder(
            src_tokens=None,
            encoder_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
            token_embeddings=x,
            multiway_split_position=multiway_split_position,
            incremental_state=incremental_state,
            positions=positions,
        )
        encoder_out["multiway_split_position"] = multiway_split_position

        return encoder_out
    
    def forward(
        self,
        textual_tokens=None,
        visual_tokens=None,
        text_padding_position=None,
        attn_mask=None,
        vision_masked_position=None,
        incremental_state=None,
        positions=None,
        segment_ids=None,
        use_vit_adapter=False,
        return_all_hiddens=False,
    ):
        if not use_vit_adapter:
            return self.forward_beit3(
                textual_tokens=textual_tokens,
                visual_tokens=visual_tokens,
                text_padding_position=text_padding_position,
                attn_mask=attn_mask,
                vision_masked_position=vision_masked_position,
                incremental_state=incremental_state,
                positions=positions,
            )
        
        bsz, _, H, W = visual_tokens.shape
        H, W = H//16, W//16
        
        deform_inputs1, deform_inputs2 = deform_inputs(visual_tokens, ss=self.beit_resolution)
        
        # SPM forward
        c1, c2, c3, c4 = self.spm(visual_tokens)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)
        
        # original beit patch embedding
        if self.asymetric_input:
            beit_input = F.interpolate(
                visual_tokens, size=self.beit_resolution, mode="bilinear"
            )
            beit_H, beit_W = self.beit_resolution[0]//16, self.beit_resolution[1]//16
        else:
            beit_input = visual_tokens
            beit_H, beit_W = H, W

        if self.intepolate_pos:
            v_cls = self.vision_embed.cls_token.expand(bsz, -1, -1)
            v_feat = self.vision_embed.proj(beit_input)
            _, _, vH, vW = v_feat.shape
            multiway_split_position = (vH * vW) + 1
            x1 = torch.cat(
                (v_cls, v_feat.flatten(2).transpose(1, 2)), dim=1,
            )
            # print(x1.shape)
            
            v_extra_pos = self.encoder.embed_positions.A.ori_weight[:3]
            v_pos = self.encoder.embed_positions.A.ori_weight[3:].reshape(40, 40, -1).permute(2, 0, 1).unsqueeze(0)
            v_pos = F.interpolate(
                v_pos,
                size=(vH, vW),
                mode='bicubic',
                antialias=False,
                align_corners=False,
            ).squeeze(0).flatten(1).transpose(0, 1)
            self.encoder.embed_positions.A.weight = torch.nn.Parameter(torch.cat((v_extra_pos, v_pos), dim=0))

            if textual_tokens is not None:
                x2 = self.text_embed(textual_tokens)
                if segment_ids is not None:
                    x2 = x2 + self.segment_embed(segment_ids)

                x = torch.cat([x1, x2], dim=1)
                if text_padding_position is not None:
                    encoder_padding_mask = torch.cat(
                        [
                            torch.zeros(x1.shape[:-1]).to(x1.device).bool(),
                            text_padding_position,
                        ],
                        dim=1,
                    )
                else:
                    encoder_padding_mask = None
            else:
                x = x1
                encoder_padding_mask = None
                multiway_split_position = -1
        
        else:
            if textual_tokens is None:
                x = self.vision_embed(beit_input, vision_masked_position)
                encoder_padding_mask = None
                multiway_split_position = -1
            else:
                x1 = self.vision_embed(beit_input, vision_masked_position)
                multiway_split_position = x1.size(1)
                x2 = self.text_embed(textual_tokens)
                if segment_ids is not None:
                    x2 = x2 + self.segment_embed(segment_ids)

                x = torch.cat([x1, x2], dim=1)
                if text_padding_position is not None:
                    encoder_padding_mask = torch.cat(
                        [
                            torch.zeros(x1.shape[:-1]).to(x1.device).bool(),
                            text_padding_position,
                        ],
                        dim=1,
                    )
                else:
                    encoder_padding_mask = None
        
        # beit3 output to encoder input
        src_tokens=None
        token_embeddings=x
        
        # original encoder embedding
        assert src_tokens is not None or token_embeddings is not None

        if encoder_padding_mask is None:
            if src_tokens is not None:
                encoder_padding_mask = torch.zeros_like(
                    src_tokens, device=src_tokens.device
                ).bool()
            else:
                encoder_padding_mask = torch.zeros(
                    [token_embeddings.size(0), token_embeddings.size(1)],
                    device=token_embeddings.device,
                ).bool()

        if multiway_split_position is not None:
            assert self.encoder.args.multiway
            self.encoder.apply(set_split_position(multiway_split_position))

        x, encoder_embedding = self.encoder.forward_embedding(src_tokens, token_embeddings, positions)
        x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))
        bs, n, dim = x.shape
        
        rel_pos_bias = None
        if self.encoder.relative_position is not None:
            rel_pos_bias = self.encoder.relative_position(
                batch_size=x.size(0), qlen=x.size(1), klen=x.size(1)
            )
        
        encoder_states = []
        if return_all_hiddens:
            encoder_states.append(x)

        # Interaction
        l_aux = None
        outs = list()
        cls, x = x[:, :1, ], x[:, 1:, ]
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            wrapped_blocks = [
                block_decorator(partial(
                    self.encoder.layers[i_layer],
                    encoder_padding_mask=encoder_padding_mask if incremental_state is None else None,
                    attn_mask=attn_mask, rel_pos=rel_pos_bias, 
                    multiway_split_position=multiway_split_position,
                    incremental_state=incremental_state[i_layer] if incremental_state is not None else None,
                ))
                for i_layer in range(indexes[0], indexes[-1]+1)
            ]
            x, c, cls, hiddens = layer(x, c, cls, multiway_split_position, wrapped_blocks,
                                              deform_inputs1, deform_inputs2, H, W, return_hiddens=return_all_hiddens)
            encoder_states.extend(hiddens)
            
            if multiway_split_position == -1:
                outs.append(x.transpose(1, 2).view(bs, dim, beit_H, beit_W).contiguous())
            else:
                x_visual, x_text = x[:, :multiway_split_position-1], x[:, multiway_split_position-1:]
                outs.append(x_visual.transpose(1, 2).view(bs, dim, beit_H, beit_W).contiguous())

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            x1 = F.interpolate(x1, size=c1.shape[-2:], mode='bilinear', align_corners=False)
            x2 = F.interpolate(x2, size=c2.shape[-2:], mode='bilinear', align_corners=False)
            x3 = F.interpolate(x3, size=c3.shape[-2:], mode='bilinear', align_corners=False)
            x4 = F.interpolate(x4, size=c4.shape[-2:], mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)

        x = torch.cat((cls, x), dim=1)
        if self.encoder.layer_norm is not None:
            x = self.encoder.layer_norm(x)
        visual_feature = x[:, 1:multiway_split_position, :]
        text_feature = x[:, multiway_split_position:, :]
        
        outputs = {
            "fpn_features": (f1, f2, f3, f4),
            "visual_feature": visual_feature,
            "text_feature": text_feature,
            "encoder_out": x,
            "encoder_embedding": encoder_embedding,
            "encoder_padding_mask": encoder_padding_mask,
            "encoder_states": encoder_states,
            "multiway_split_position": multiway_split_position,
        }
        
        return outputs