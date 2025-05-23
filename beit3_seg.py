from typing import Optional, Dict, List, Tuple
import numpy as np

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from dataclasses import dataclass
from transformers.file_utils import ModelOutput, requires_backends
from transformers.activations import ACT2FN
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerPixelDecoder, Mask2FormerModel, Mask2FormerPreTrainedModel,
    Mask2FormerTransformerModule, Mask2FormerLoss, Mask2FormerSinePositionEmbedding,
    Mask2FormerMaskedAttentionDecoder, Mask2FormerMaskedAttentionDecoderOutput, Mask2FormerAttention,
    Mask2FormerMaskPredictor, Mask2FormerHungarianMatcher)

from beit3_adapter import BEiT3Adapter

### define class outputs
@dataclass
class BEiT3SegMaskedAttentionDecoderOutput(Mask2FormerMaskedAttentionDecoderOutput):
    text_last_hidden_state: torch.FloatTensor = None
    text_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    text_intermediate_hidden_states: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class BEiT3SegPixelLevelModuleOutput(ModelOutput):
    fpn_features: Tuple[torch.FloatTensor] = None
    text_feature: torch.FloatTensor = None
    encoder_last_hidden_state: torch.FloatTensor = None
    encoder_visual_last_hidden_state: torch.FloatTensor = None
    encoder_text_last_hidden_state: torch.FloatTensor = None

    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None # beit3 new
    decoder_last_hidden_state: torch.FloatTensor = None
    decoder_hidden_states: Tuple[torch.FloatTensor] = None
        
@dataclass
class BEiT3SegModelOutput(ModelOutput):
    encoder_visual_last_hidden_state: torch.FloatTensor = None # beit3 new
    encoder_text_last_hidden_state: torch.FloatTensor = None # beit3 new
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None # beit3 new
    fpn_features: Tuple[torch.FloatTensor] = None # beit3 new

    transformer_decoder_text_last_hidden_state: torch.FloatTensor = None
    transformer_decoder_text_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    transformer_decoder_text_intermediate_states: Tuple[torch.FloatTensor] = None
        
    pixel_decoder_last_hidden_state: torch.FloatTensor = None
    transformer_decoder_last_hidden_state: torch.FloatTensor = None
    pixel_decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    transformer_decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    transformer_decoder_intermediate_states: Tuple[torch.FloatTensor] = None
    masks_queries_logits: Tuple[torch.FloatTensor] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class BEiT3SegForUniversalSegmentationOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    class_queries_logits: torch.FloatTensor = None
    masks_queries_logits: torch.FloatTensor = None
    auxiliary_logits: Optional[List[Dict[str, torch.FloatTensor]]] = None
        
    encoder_visual_last_hidden_state: torch.FloatTensor = None # beit3 new
    encoder_text_last_hidden_state: torch.FloatTensor = None # beit3 new
    fpn_features: Tuple[torch.FloatTensor] = None # beit3 new
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None # beit3 new
        
    pixel_decoder_last_hidden_state: torch.FloatTensor = None
    transformer_decoder_last_hidden_state: torch.FloatTensor = None
    pixel_decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    transformer_decoder_hidden_states: Optional[torch.FloatTensor] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

    loss_dict: Optional[Dict[str, torch.FloatTensor]] = None

# Adapted from https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/criterion.py
class BEiT3SegLoss(Mask2FormerLoss):
    def __init__(self, config, weight_dict: Dict[str, float]):
        """
        The Mask2Former Loss. The loss is computed very similar to DETR. The process happens in two steps: 1) we
        compute hungarian assignment between ground truth masks and the outputs of the model 2) we supervise each pair
        of matched ground-truth / prediction (supervise class and mask)

        Args:
            config (`Mask2FormerConfig`):
                The configuration for Mask2Former model also containing loss calculation specific parameters.
            weight_dict (`Dict[str, float]`):
                A dictionary of weights to be applied to the different losses.
        """
        super().__init__(config, weight_dict)
        self.match_once_only = config.match_once_only
        self.drop_first_ce_loss = config.drop_first_ce_loss
        self.use_objectness_loss = config.use_objectness_loss

        if self.use_objectness_loss:
            self.obj_loss_weight = torch.ones(2)
            self.obj_loss_weight[-1] = self.eos_coef
            self.empty_weight = self.empty_weight[:-1]

    def forward(
        self,
        masks_queries_logits: torch.Tensor,
        class_queries_logits: torch.Tensor,
        mask_labels: List[torch.Tensor],
        class_labels: List[torch.Tensor],
        auxiliary_predictions: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:

        if self.use_objectness_loss:
            class_queries_logits_cls, class_queries_logits_obj = class_queries_logits
            # all obj_labels = 0 (first class)
            obj_labels = [torch.zeros_like(class_label) for class_label in class_labels]
            indices = self.matcher(masks_queries_logits, class_queries_logits_obj, mask_labels, obj_labels)
            num_masks = self.get_num_masks(obj_labels, device=obj_labels[0].device)

            # print('class_queries_logits_cls', class_queries_logits_cls.shape)
            # print('class_queries_logits_obj', class_queries_logits_obj.shape)

            losses: Dict[str, Tensor] = {
                **self.loss_masks(masks_queries_logits, mask_labels, indices, num_masks),
                **self.loss_labels(
                    class_queries_logits_obj, obj_labels, indices,
                    empty_weight=self.obj_loss_weight.to(class_queries_logits_obj.device),
                    fill_value=1, loss_name="loss_objectness",
                    ),
                **self.loss_labels(
                    class_queries_logits_cls, class_labels, indices,
                    empty_weight=self.empty_weight.to(class_queries_logits_cls.device),
                    fill_value=-100, loss_name="loss_cross_entropy",
                    ),
            }
        
        else:
            # retrieve the matching between the outputs of the last layer and the labels
            # print('run match')
            indices = self.matcher(masks_queries_logits, class_queries_logits, mask_labels, class_labels)
            # compute the average number of target masks for normalization purposes
            num_masks = self.get_num_masks(class_labels, device=class_labels[0].device)
            # get all the losses
            losses: Dict[str, Tensor] = {
                **self.loss_masks(masks_queries_logits, mask_labels, indices, num_masks),
                **self.loss_labels(class_queries_logits, class_labels, indices),
            }

        # in case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if auxiliary_predictions is not None:
            for idx, aux_outputs in enumerate(auxiliary_predictions):
                masks_queries_logits = aux_outputs["masks_queries_logits"]
                class_queries_logits = aux_outputs["class_queries_logits"]
                if self.match_once_only:
                    raise NotImplementedError('not implement match_once_only')
                    loss_dict = {
                        **self.loss_masks(masks_queries_logits, mask_labels, indices, num_masks),
                        **self.loss_labels(class_queries_logits, class_labels, indices),
                    }
                else:
                    loss_dict = self.forward(masks_queries_logits, class_queries_logits, mask_labels, class_labels)

                if idx == 0 and self.drop_first_ce_loss:
                    if 'loss_cross_entropy' in loss_dict:
                        del loss_dict['loss_cross_entropy']
                    if 'loss_objectness' in loss_dict:
                        del loss_dict['loss_objectness']
                
                loss_dict = {f"{key}_{idx}": value for key, value in loss_dict.items()}
                losses.update(loss_dict)

        return losses

    def loss_labels(
        self, class_queries_logits: Tensor, class_labels: List[Tensor], indices: Tuple[np.array],
        empty_weight=None, fill_value=None, loss_name="loss_cross_entropy",
    ):
        if empty_weight is None:
            empty_weight = self.empty_weight
        if fill_value is None:
            fill_value = self.num_labels

        pred_logits = class_queries_logits
        batch_size, num_queries, _ = pred_logits.shape
        criterion = nn.CrossEntropyLoss(empty_weight)
        idx = self._get_predictions_permutation_indices(indices)  # shape of (batch_size, num_queries)
        target_classes_o = torch.cat(
            [target[j] for target, (_, j) in zip(class_labels, indices)]
        )  # shape of (batch_size, num_queries)
        target_classes = torch.full(
            (batch_size, num_queries), fill_value=fill_value, dtype=torch.int64, device=pred_logits.device
        )
        target_classes[idx] = target_classes_o
        # Permute target_classes (batch_size, num_queries, num_labels) -> (batch_size, num_labels, num_queries)
        pred_logits_transposed = pred_logits.transpose(1, 2)
        loss_ce = criterion(pred_logits_transposed, target_classes)
        losses = {loss_name: loss_ce}
        return losses

class BEiT3SegPixelLevelModule(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder = BEiT3Adapter(**config.backbone_config)
        self.decoder = Mask2FormerPixelDecoder(config, feature_channels=[self.encoder.embed_dim] * 4)

    def forward(
        self, pixel_values,
        input_ids=None, text_padding_position=None,
        position_ids: Optional[Tensor] = None,
        segment_ids: Optional[Tensor] = None,
        output_hidden_states=False,
        ):
        backbone_features = self.encoder(
            visual_tokens=pixel_values,
            textual_tokens=input_ids,
            text_padding_position=text_padding_position,
            positions=position_ids,
            segment_ids=segment_ids,
            use_vit_adapter=True,
            return_all_hiddens=True,
        )
        
        fpn_features = backbone_features['fpn_features']
        text_feature = backbone_features['text_feature']
        encoder_out = backbone_features['encoder_out']
        multiway_split_position = backbone_features['multiway_split_position']
        if multiway_split_position == -1:
            encoder_visual_last_hidden_state = encoder_out
            encoder_text_last_hidden_state = None
        else:
            encoder_visual_last_hidden_state = encoder_out[:, :multiway_split_position]
            encoder_text_last_hidden_state = encoder_out[:, multiway_split_position:]

        encoder_encoder_states = backbone_features['encoder_states']
        if multiway_split_position == -1:
            encoder_encoder_states = [[state[:, :1], state[:, 1:]] for state in encoder_encoder_states]
        else:
            encoder_encoder_states = [
                [state[:, :1], state[:, 1:multiway_split_position], state[:, multiway_split_position:]]
            for state in encoder_encoder_states]
        
        decoder_output = self.decoder(fpn_features, output_hidden_states=output_hidden_states)

        return BEiT3SegPixelLevelModuleOutput(
            fpn_features=fpn_features,
            text_feature=text_feature,

            encoder_last_hidden_state=encoder_out,
            encoder_visual_last_hidden_state=encoder_visual_last_hidden_state,
            encoder_text_last_hidden_state=encoder_text_last_hidden_state,
            encoder_hidden_states=encoder_encoder_states,

            decoder_last_hidden_state=decoder_output.mask_features,
            decoder_hidden_states=decoder_output.multi_scale_features,
        )

class BEiT3SegMaskedAttentionDecoderLayer(nn.Module):
    """
    The Mask2FormerMaskedAttentionDecoderLayer is made up of self-attention, cross (masked) attention as well as FFN
    blocks. The cross attention block used as part of `Mask2FormerMaskedAttentionDecoderLayer` is actually a `masked
    attention` block that restricts the attention to localized features centered around predicted segments which leads
    to faster convergence and improved performance. The order of self and cross (i.e. masked) attention blocks have
    also been swapped in Mask2FormerMaskedAttentionDecoder compared to a standard DetrDecoder as an optimization
    improvement.

    Args:
        config (`Mask2FormerConfig`):
            The configuration used to initialize the Mask2FormerMaskedAttentionDecoder.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = self.config.hidden_dim
        self.pre_norm = self.config.pre_norm
        self.self_attn = nn.MultiheadAttention(
            self.embed_dim,
            self.config.num_attention_heads,
            self.config.dropout,
        )

        self.dropout = self.config.dropout
        self.activation_fn = ACT2FN[self.config.activation_function]
        self.activation_dropout = self.config.dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.cross_attn = nn.MultiheadAttention(self.embed_dim, self.config.num_attention_heads, self.config.dropout)
        self.cross_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, self.config.dim_feedforward)
        self.fc2 = nn.Linear(self.config.dim_feedforward, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

        self.use_text_cross_attn = self.config.use_text_cross_attn
        if self.use_text_cross_attn:
            self.text_cross_attn = nn.MultiheadAttention(self.embed_dim, self.config.num_attention_heads, self.config.dropout)
            self.text_cross_attn_layer_norm = nn.LayerNorm(self.embed_dim)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        hidden_states: torch.Tensor,
        level_index: int = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        text_position_embeddings: Optional[torch.Tensor] = None,
        query_position_embeddings: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_text_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ):
        # Masked(Cross)-Attention Block
        cross_attn_weights = None
        self_attn_weights = None

        residual = hidden_states
            
        hidden_states, cross_attn_weights = self.cross_attn(
            query=self.with_pos_embed(hidden_states, query_position_embeddings),
            key=self.with_pos_embed(encoder_hidden_states[level_index], position_embeddings[level_index]),
            value=encoder_hidden_states[level_index],
            attn_mask=encoder_attention_mask,
            key_padding_mask=None,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.cross_attn_layer_norm(hidden_states)

        # Text Cross-Attention Block
        if self.use_text_cross_attn:
            residual = hidden_states
            hidden_states, _ = self.text_cross_attn(
                query=self.with_pos_embed(hidden_states, query_position_embeddings),
                key=self.with_pos_embed(encoder_text_hidden_states, text_position_embeddings),
                value=encoder_text_hidden_states,
                attn_mask=None,
                key_padding_mask=None,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.text_cross_attn_layer_norm(hidden_states)

        # Self Attention Block
        residual = hidden_states
        hidden_states, self_attn_weights = self.self_attn(
            query=self.with_pos_embed(hidden_states, query_position_embeddings),
            key=self.with_pos_embed(hidden_states, query_position_embeddings),
            value=hidden_states,
            attn_mask=None,
            key_padding_mask=None,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states, None, )

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        level_index: int = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        text_position_embeddings: Optional[torch.Tensor] = None,
        query_position_embeddings: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_text_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(seq_len, batch, embed_dim)`.
            attention_mask (`torch.FloatTensor`):
                Attention mask of shape `(1, seq_len, tgt_len, src_len)`.
            position_embeddings (`torch.FloatTensor`, *optional*):
                Position embeddings that are added to the keys in the masked-attention layer.
            query_position_embeddings (`torch.FloatTensor`, *optional*):
                Position embeddings that are added to the queries and keys in the self-attention layer.
            encoder_hidden_states (`torch.FloatTensor`):
                Cross attention input to the layer of shape `(seq_len, batch, embed_dim)`.
            encoder_attention_mask (`torch.FloatTensor`):
                Encoder attention mask of size`(1, seq_len, tgt_len, src_len)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """

        if self.pre_norm:
            raise NotImplementedError('not implement pre-norm')
        else:
            outputs = self.forward_post(
                hidden_states=hidden_states,
                level_index=level_index,
                position_embeddings=position_embeddings,
                text_position_embeddings=text_position_embeddings,
                query_position_embeddings=query_position_embeddings,
                encoder_hidden_states=encoder_hidden_states,
                encoder_text_hidden_states=encoder_text_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )

        return outputs


class BEiT3SegMaskPredictor(Mask2FormerMaskPredictor):
    def forward(self, outputs: torch.Tensor, pixel_embeddings: torch.Tensor, attention_mask_target_size: int = None):
        mask_embeddings = self.mask_embedder(outputs.transpose(0, 1))

        outputs_mask = torch.einsum('bqc, bchw -> bqhw', mask_embeddings, pixel_embeddings)

        attention_mask = nn.functional.interpolate(
            outputs_mask, size=attention_mask_target_size, mode="bilinear", align_corners=False
        )

        attention_mask = attention_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        attention_mask = (attention_mask.flatten(0, 1) < 0.5).bool()
        attention_mask = attention_mask.detach()

        return outputs_mask, attention_mask

class BEiT3SegMaskedAttentionDecoder(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.mask_feature_size = config.mask_feature_size
        self.dropout = config.dropout
        self.layerdrop = config.dropout
        self.num_feature_levels = 3  # level embedding (3 scales)
        self.decoder_layers = config.decoder_layers - 1

        self.layers = nn.ModuleList(
            [BEiT3SegMaskedAttentionDecoderLayer(self.config) for _ in range(self.decoder_layers)]
        )
        self.layernorm = nn.LayerNorm(config.hidden_dim)

        self.mask_predictor = BEiT3SegMaskPredictor(
            hidden_size=config.hidden_dim,
            num_heads=config.num_attention_heads,
            mask_feature_size=self.mask_feature_size,
        )

        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds: torch.Tensor = None,
        multi_stage_positional_embeddings: torch.Tensor = None,
        text_positional_embeddings: torch.Tensor = None,
        pixel_embeddings: torch.Tensor = None,
        encoder_hidden_states: torch.Tensor = None,
        encoder_text_hidden_states: torch.Tensor = None,
        query_position_embeddings: torch.Tensor = None,
        feature_size_list: List = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(num_queries, batch_size, hidden_size)`):
                The query embeddings that are passed into the decoder.
            multi_stage_positional_embeddings (`torch.FloatTensor` of shape `(height*width, batch_size, num_channels)`):
                Position embeddings that are added to the keys in each cross(masked)-attention layer.
            pixel_embeddings (`torch.FloatTensor`):
                Tensor of shape `(batch_size, num_channels, height, width)`, 1/4 scale features from the last Pixel
                Decoder.
            query_position_embeddings (`torch.FloatTensor` of shape `(num_queries, batch_size, hidden_size)`):
                , *optional*): Position embeddings that are added to the queries and keys in each self-attention layer.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the
                cross(masked)-attention of the decoder.
            feature_size_list (`List[torch.Size]` ):
                This is a list containing shapes (height & width) of multi-scale features from the Pixel Decoder.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is not None:
            hidden_states = inputs_embeds

        # intermediate hidden states with layernorm applied - required for predicting class logits
        intermediate = ()
        text_intermediate = ()

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_text_hidden_states = () if output_hidden_states else None
        attentions = () if output_attentions else None

        # intermediate mask predictions from transformer decoder layers
        intermediate_mask_predictions = ()

        intermediate_hidden_states = self.layernorm(inputs_embeds)
        intermediate += (intermediate_hidden_states,)

        predicted_mask, attention_mask = self.mask_predictor(
            intermediate_hidden_states, pixel_embeddings, feature_size_list[0]
        )
        intermediate_mask_predictions += (predicted_mask,)

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                # all_text_hidden_states += (text_hidden_states,)

            dropout_probability = torch.rand([])

            if self.training and (dropout_probability < self.layerdrop):
                continue

            if self.gradient_checkpointing and self.training:
                raise NotImplementedError('no grad checkpointing')
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    None,
                    None,
                    output_attentions,
                )

            else:
                level_index = idx % self.num_feature_levels

                attention_mask[torch.where(attention_mask.sum(-1) == attention_mask.shape[-1])] = False

                layer_outputs = decoder_layer(
                    hidden_states,
                    level_index=level_index,
                    position_embeddings=multi_stage_positional_embeddings,
                    text_position_embeddings=text_positional_embeddings,
                    query_position_embeddings=query_position_embeddings,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_text_hidden_states=encoder_text_hidden_states,
                    encoder_attention_mask=attention_mask,
                    output_attentions=output_attentions,
                )

                intermediate_hidden_states = self.layernorm(layer_outputs[0])

                predicted_mask, attention_mask = self.mask_predictor(
                    intermediate_hidden_states,
                    pixel_embeddings,
                    feature_size_list[(idx + 1) % self.num_feature_levels],
                )

                intermediate_mask_predictions += (predicted_mask,)

                # add intermediate hidden states with layer norm applied which will be used for predicting class logits
                intermediate += (intermediate_hidden_states,)

            hidden_states = layer_outputs[0]
            text_hidden_states = layer_outputs[1]

            if output_attentions:
                attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
            all_text_hidden_states += (text_hidden_states,)

        hidden_states = hidden_states.transpose(1, 0)
        # text_hidden_states = text_hidden_states.transpose(1, 0)
        if not return_dict:
            outputs = [hidden_states, all_hidden_states, attentions, intermediate, intermediate_mask_predictions]
            return tuple(v for v in outputs if v is not None)

        return BEiT3SegMaskedAttentionDecoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            # text_last_hidden_state=text_hidden_states,
            text_hidden_states=all_text_hidden_states,
            attentions=attentions,
            intermediate_hidden_states=intermediate,
            text_intermediate_hidden_states=text_intermediate,
            masks_queries_logits=intermediate_mask_predictions,
        )

class BEiT3SegTransformerModule(nn.Module):
    def __init__(self, in_features, config):
        super().__init__()
        hidden_dim = config.hidden_dim
        self.num_feature_levels = 3
        self.position_embedder = Mask2FormerSinePositionEmbedding(num_pos_feats=hidden_dim // 2, normalize=True)
        self.queries_embedder = nn.Embedding(config.num_queries, hidden_dim)
        self.queries_features = nn.Embedding(config.num_queries, hidden_dim)
        self.input_projections = []

        # for text
        self.use_text_features = config.use_text_features
        if self.use_text_features:
            self.text_position_embedding = nn.Embedding(1000, hidden_dim)
            self.text_queries_features = nn.Embedding(config.num_queries, hidden_dim)
            # self.psuedo_class_embedder = nn.Embedding(config.num_labels + 1, hidden_dim)
            self.text_input_projections = nn.Linear(config.backbone_dim, hidden_dim)

        for _ in range(self.num_feature_levels):
            if in_features != hidden_dim or config.enforce_input_projection:
                self.input_projections.append(nn.Conv2d(in_features, hidden_dim, kernel_size=1))
            else:
                self.input_projections.append(nn.Sequential())

        self.decoder = BEiT3SegMaskedAttentionDecoder(config=config)
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)

        self.intepolate_pos = False

    def forward(
        self,
        multi_scale_features: List[Tensor],
        mask_features: Tensor,
        text_features: Tensor,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> BEiT3SegMaskedAttentionDecoderOutput:
        multi_stage_features = []
        multi_stage_positional_embeddings = []
        size_list = []

        for i in range(self.num_feature_levels):
            size_list.append(multi_scale_features[i].shape[-2:])
            multi_stage_positional_embeddings.append(self.position_embedder(multi_scale_features[i], None).flatten(2))
            multi_stage_features.append(
                self.input_projections[i](multi_scale_features[i]).flatten(2)
                + self.level_embed.weight[i][None, :, None]
            )

            # Flatten (batch_size, num_channels, height, width) -> (height*width, batch_size, num_channels)
            multi_stage_positional_embeddings[-1] = multi_stage_positional_embeddings[-1].permute(2, 0, 1)
            multi_stage_features[-1] = multi_stage_features[-1].permute(2, 0, 1)

        # for text
        if text_features is not None and self.use_text_features:
            batch_size, text_seq_len, _ = text_features.shape
            # bsz, text_len, c -> text_len, bsz, c
            text_embeddings = self.text_input_projections(text_features.transpose(0, 1))
            text_pos_embedding = None
        else:
            text_embeddings = None
            text_pos_embedding = None

        _, batch_size, _ = multi_stage_features[0].shape

        # [num_queries, batch_size, num_channels]
        query_embeddings = self.queries_embedder.weight.unsqueeze(1).repeat(1, batch_size, 1)
        query_features = self.queries_features.weight.unsqueeze(1).repeat(1, batch_size, 1)
        # text_query_features = self.text_queries_features.weight.unsqueeze(1).repeat(1, batch_size, 1)

        decoder_output = self.decoder(
            inputs_embeds=query_features,
            # text_inputs_embeds=text_query_features,
            multi_stage_positional_embeddings=multi_stage_positional_embeddings,
            text_positional_embeddings=text_pos_embedding,
            pixel_embeddings=mask_features,
            encoder_hidden_states=multi_stage_features,
            encoder_text_hidden_states=text_embeddings,
            query_position_embeddings=query_embeddings,
            feature_size_list=size_list,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )

        return decoder_output

    
class BEiT3SegModel(Mask2FormerPreTrainedModel):
    main_input_name = "pixel_values"

    def __init__(self, config):
        super().__init__(config)
        self.pixel_level_module = BEiT3SegPixelLevelModule(config)
        self.transformer_module = BEiT3SegTransformerModule(in_features=config.feature_size, config=config)
        self.post_init()

    def forward(
        self,
        pixel_values: Tensor,
        input_ids: Optional[Tensor] = None,
        cat_input_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        segment_ids: Optional[Tensor] = None,
        text_padding_position=None,
        pixel_mask: Optional[Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> BEiT3SegModelOutput:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, _, height, width = pixel_values.shape

        if pixel_mask is None:
            pixel_mask = torch.ones((batch_size, height, width), device=pixel_values.device)

        pixel_level_module_output = self.pixel_level_module(
            pixel_values=pixel_values,
            input_ids=input_ids, text_padding_position=text_padding_position,
            position_ids=position_ids, segment_ids=segment_ids,
            output_hidden_states=output_hidden_states,
        )

        if input_ids is None:
            pixel_level_module_output.text_feature = None

        if cat_input_ids is not None:
            pixel_level_module_output.text_feature = pixel_level_module_output.text_feature[torch.arange(batch_size).unsqueeze(-1), cat_input_ids]

        transformer_module_output = self.transformer_module(
            multi_scale_features=pixel_level_module_output.decoder_hidden_states,
            mask_features=pixel_level_module_output.decoder_last_hidden_state,
            text_features=pixel_level_module_output.text_feature,
            output_hidden_states=True,
            output_attentions=output_attentions,
        )

        fpn_features = None
        pixel_decoder_hidden_states = None
        transformer_decoder_hidden_states = None
        transformer_decoder_intermediate_states = None
        transformer_decoder_text_hidden_states = None
        transformer_decoder_text_intermediate_states = None

        if output_hidden_states:
            fpn_features = pixel_level_module_output.fpn_features
            pixel_decoder_hidden_states = pixel_level_module_output.decoder_hidden_states
            transformer_decoder_hidden_states = transformer_module_output.hidden_states
            transformer_decoder_intermediate_states = transformer_module_output.intermediate_hidden_states

            transformer_decoder_text_hidden_states = transformer_module_output.text_hidden_states
            transformer_decoder_text_intermediate_states = transformer_module_output.text_intermediate_hidden_states

        output = BEiT3SegModelOutput(
            encoder_visual_last_hidden_state=pixel_level_module_output.encoder_visual_last_hidden_state,
            encoder_text_last_hidden_state=pixel_level_module_output.encoder_text_last_hidden_state,
            encoder_hidden_states=pixel_level_module_output.encoder_hidden_states,
            pixel_decoder_last_hidden_state=pixel_level_module_output.decoder_last_hidden_state,
            transformer_decoder_last_hidden_state=transformer_module_output.last_hidden_state,
            fpn_features=fpn_features,
            pixel_decoder_hidden_states=pixel_decoder_hidden_states,
            transformer_decoder_hidden_states=transformer_decoder_hidden_states,
            transformer_decoder_intermediate_states=transformer_decoder_intermediate_states,
            attentions=transformer_module_output.attentions,
            masks_queries_logits=transformer_module_output.masks_queries_logits,

            transformer_decoder_text_last_hidden_state=transformer_module_output.text_last_hidden_state,
            transformer_decoder_text_hidden_states=transformer_decoder_text_hidden_states,
            transformer_decoder_text_intermediate_states=transformer_decoder_text_intermediate_states,
        )

        if not return_dict:
            output = tuple(v for v in output.values() if v is not None)

        return output
    
class BEiT3SegForUniversalSegmentation(Mask2FormerPreTrainedModel):
    main_input_name = "pixel_values"

    def __init__(self, config):
        super().__init__(config)
        self.model = BEiT3SegModel(config)

        self.weight_dict: Dict[str, float] = {
            "loss_objectness": config.objectness_weight if config.use_objectness_loss else 0.0,
            "loss_cross_entropy": config.class_weight,
            "loss_mask": config.mask_weight,
            "loss_dice": config.dice_weight,
        }

        self.use_text_contrastive_loss = config.use_text_contrastive_loss
        self.use_objectness_loss = config.use_objectness_loss

        if self.use_objectness_loss and self.use_text_contrastive_loss:
            self.class_predictor = nn.Sequential(
                nn.Linear(config.hidden_dim, 2),
            )
            self.query_head = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
            )
            self.text_target_head = nn.Sequential(
                nn.Linear(config.backbone_dim, config.hidden_dim),
            )
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07), requires_grad=True)

        elif self.use_text_contrastive_loss:
            self.class_predictor = nn.Sequential(
                nn.Linear(config.hidden_dim, 1),
            )
            self.query_head = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
            )
            self.text_target_head = nn.Sequential(
                nn.Linear(config.backbone_dim, config.hidden_dim),
            )
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07), requires_grad=True)

        else:
            self.class_predictor = nn.Sequential(
                nn.Linear(config.hidden_dim, config.num_labels + 1)
            )
        
        self.criterion = BEiT3SegLoss(config=config, weight_dict=self.weight_dict)
        self.post_init()

        self.model.pixel_level_module.encoder.encoder.embed_positions.A.ori_weight = None


    def set_intepolate_pos(self, interpolate_pos):
        self.model.pixel_level_module.encoder.intepolate_pos = interpolate_pos

        if interpolate_pos == True:
            if self.model.pixel_level_module.encoder.encoder.embed_positions.A.ori_weight is None:
                self.model.pixel_level_module.encoder.encoder.embed_positions.A.ori_weight = self.model.pixel_level_module.encoder.encoder.embed_positions.A.weight
        else:
            if self.model.pixel_level_module.encoder.encoder.embed_positions.A.ori_weight is not None:
                self.model.pixel_level_module.encoder.encoder.embed_positions.A.weight = self.model.pixel_level_module.encoder.encoder.embed_positions.A.ori_weight
                self.model.pixel_level_module.encoder.encoder.embed_positions.A.ori_weight = None



    def get_loss_dict(
        self,
        masks_queries_logits: Tensor,
        class_queries_logits: Tensor,
        mask_labels: Tensor,
        class_labels: Tensor,
        auxiliary_predictions: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        loss_dict: Dict[str, Tensor] = self.criterion(
            masks_queries_logits=masks_queries_logits,
            class_queries_logits=class_queries_logits,
            mask_labels=mask_labels,
            class_labels=class_labels,
            auxiliary_predictions=auxiliary_predictions,
        )

        # weight each loss by `self.weight_dict[<LOSS_NAME>]` including auxiliary losses
        for key, weight in self.weight_dict.items():
            for loss_key, loss in loss_dict.items():
                if key in loss_key:
                    loss *= weight

        return loss_dict

    def get_loss(self, loss_dict: Dict[str, Tensor]) -> Tensor:
        return sum(loss_dict.values())

    def get_auxiliary_logits(self, classes: torch.Tensor, output_masks: torch.Tensor):
        auxiliary_logits: List[Dict(str, Tensor)] = []

        for aux_binary_masks, aux_classes in zip(output_masks[:-1], classes[:-1]):
            auxiliary_logits.append({"masks_queries_logits": aux_binary_masks, "class_queries_logits": aux_classes})

        return auxiliary_logits

    def forward(
        self,
        pixel_values: Tensor,
        input_ids: Optional[Tensor] = None,
        text_padding_position=None,
        cat_input_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        segment_ids: Optional[Tensor] = None,
        mask_labels: Optional[List[Tensor]] = None,
        class_labels: Optional[List[Tensor]] = None,
        pixel_mask: Optional[Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_auxiliary_logits: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_loss_dict: Optional[bool] = None,
    ) -> BEiT3SegForUniversalSegmentationOutput:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            cat_input_ids=cat_input_ids,
            position_ids=position_ids,
            segment_ids=segment_ids,
            text_padding_position=text_padding_position,
            pixel_mask=pixel_mask,
            output_hidden_states=output_hidden_states or self.config.use_auxiliary_loss,
            output_attentions=output_attentions,
            return_dict=True,
        )

        loss, loss_dict, auxiliary_logits = None, None, None
        class_queries_logits = ()

        if self.use_objectness_loss and self.use_text_contrastive_loss:
            bsz, _, _, _ = pixel_values.shape
            logit_scale = self.logit_scale.exp()

            target_embedding = outputs.encoder_text_last_hidden_state
            if cat_input_ids is not None:
                target_embedding = target_embedding[torch.arange(bsz).unsqueeze(-1), cat_input_ids]
            target_embedding = self.text_target_head(target_embedding)
            target_embedding = F.normalize(target_embedding, dim=-1)

            for decoder_output in outputs.transformer_decoder_intermediate_states:
                class_embedding = self.query_head(decoder_output.transpose(0, 1))
                class_embedding = F.normalize(class_embedding, dim=-1)

                class_prediction = logit_scale * class_embedding @ target_embedding.transpose(1, 2)

                mask_prob = self.class_predictor(decoder_output.transpose(0, 1))
                class_prediction = (class_prediction, mask_prob)

                class_queries_logits += (class_prediction,)

        elif self.use_text_contrastive_loss:
            # contrastive loss
            bsz, _, _, _ = pixel_values.shape
            logit_scale = self.logit_scale.exp()

            target_embedding = outputs.encoder_text_last_hidden_state
            if cat_input_ids is not None:
                target_embedding = target_embedding[torch.arange(bsz).unsqueeze(-1), cat_input_ids]
            target_embedding = self.text_target_head(target_embedding)
            target_embedding = F.normalize(target_embedding, dim=-1)

            for decoder_output in outputs.transformer_decoder_intermediate_states:
                class_embedding = self.query_head(decoder_output.transpose(0, 1))
                class_embedding = F.normalize(class_embedding, dim=-1)

                class_prediction = logit_scale * class_embedding @ target_embedding.transpose(1, 2)

                mask_prob = self.class_predictor(decoder_output.transpose(0, 1))
                class_prediction = torch.cat([class_prediction, mask_prob], dim=-1)

                class_queries_logits += (class_prediction,)
        else:
            # cls loss
            for decoder_output in outputs.transformer_decoder_intermediate_states:
                class_prediction = self.class_predictor(decoder_output.transpose(0, 1))
                class_queries_logits += (class_prediction,)

        masks_queries_logits = outputs.masks_queries_logits

        auxiliary_logits = self.get_auxiliary_logits(class_queries_logits, masks_queries_logits)

        if mask_labels is not None and class_labels is not None:
            loss_dict = self.get_loss_dict(
                masks_queries_logits=masks_queries_logits[-1],
                class_queries_logits=class_queries_logits[-1],
                mask_labels=mask_labels,
                class_labels=class_labels,
                auxiliary_predictions=auxiliary_logits,
            )
            loss = self.get_loss(loss_dict)

        fpn_features = None
        pixel_decoder_hidden_states = None
        transformer_decoder_hidden_states = None

        if output_hidden_states:
            fpn_features = outputs.fpn_features
            pixel_decoder_hidden_states = outputs.pixel_decoder_hidden_states
            transformer_decoder_hidden_states = outputs.transformer_decoder_hidden_states

        output_auxiliary_logits = (
            self.config.output_auxiliary_logits if output_auxiliary_logits is None else output_auxiliary_logits
        )
        if not output_auxiliary_logits:
            auxiliary_logits = None

        output = BEiT3SegForUniversalSegmentationOutput(
            loss=loss,
            class_queries_logits=class_queries_logits[-1],
            masks_queries_logits=masks_queries_logits[-1],
            auxiliary_logits=auxiliary_logits,
            
            encoder_visual_last_hidden_state=outputs.encoder_visual_last_hidden_state,
            encoder_text_last_hidden_state=outputs.encoder_text_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            pixel_decoder_last_hidden_state=outputs.pixel_decoder_last_hidden_state,
            transformer_decoder_last_hidden_state=outputs.transformer_decoder_last_hidden_state,
            fpn_features=fpn_features,
            pixel_decoder_hidden_states=pixel_decoder_hidden_states,
            transformer_decoder_hidden_states=transformer_decoder_hidden_states,
            attentions=outputs.attentions,

            loss_dict=loss_dict if return_loss_dict else None,
        )

        if not return_dict:
            output = tuple(v for v in output.values() if v is not None)
            if loss is not None:
                output = ((loss)) + output
        return output