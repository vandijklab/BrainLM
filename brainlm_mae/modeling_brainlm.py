import math
from copy import deepcopy

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Union, Tuple
from transformers.models.vit_mae.modeling_vit_mae import (
    ViTMAEForPreTraining,
    ViTMAEEncoder,
    ViTMAEModel,
    ViTMAEEmbeddings,
    ViTMAEForPreTrainingOutput,
    ViTMAEModelOutput,
    ViTMAEDecoder,
    ViTMAEDecoderOutput,
)
from transformers.models.nystromformer.modeling_nystromformer import NystromformerLayer
from transformers.modeling_outputs import BaseModelOutput


class PositionalEncoding(nn.Module):
    """
    Positional Encoding module from PyTorch tutorial: [link]

    Positional Encoding Formula:
    - PE(pos, 2i) = sin(pos / ( 10000^{2i/d_model} ))  # Even dimensions = sin frequency
    - PE(pos, 2i+1) = cos(pos / ( 10000^{2i/d_model} ))  # Odd dimensions = cosine frequency

    10000 is a user-defined variable, chosen as 10000 by authors of original Transformer paper
    - Scaling by 1/10000 makes 1 cycle very long => guarantees unique positional encodings
    - If you plot sin(x / 10000), takes > 60k to complete 1 cycle of sin
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # div_term creates this part of expression: ( 10000^{2i/d_model} )
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        position = torch.arange(max_len).unsqueeze(1)  # 0 through 5000
        pe = torch.zeros(max_len, d_model)  # [max_seq_len, hidden_size]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        pos_encoding = self.pe[: x.size(2)]  # shape [seq_len, 1, embedding_dim]
        pos_encoding = (
            pos_encoding.unsqueeze(0).unsqueeze(0).repeat(x.shape[0], x.shape[1], 1, 1)
        )
        x = x + pos_encoding
        return self.dropout(x)


class BrainLMEmbeddings(ViTMAEEmbeddings):
    """
    Construct the CLS token, gene index and cell index embeddings.
    """

    def __init__(self, config):
        super().__init__(config)
        # CellLM doesn't use patches, so set attributes dealing with patches to None
        self.patch_embeddings = None
        self.position_embeddings = None
        self.num_brain_voxels = config.num_brain_voxels
        self.num_timepoints_per_voxel = config.num_timepoints_per_voxel
        # self.num_last_timepoints_masked = config.num_last_timepoints_masked
        self.mask_ratio = config.mask_ratio
        self.timepoint_patching_size = config.timepoint_patching_size

        self.signal_embedding_projection = nn.Linear(
            self.timepoint_patching_size, config.hidden_size, bias=True
        )
        self.xyz_embedding_projection = nn.Linear(3, config.hidden_size, bias=True)
        self.pos_embedding = PositionalEncoding(d_model=config.hidden_size)

    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=self.config.initializer_range)

    def forward(self, signal_vectors, xyz_vectors, noise):
        """
        :param signal_vectors: torch tensor of shape [batch, num_voxels, num_timepoints_per_voxel]
        :param xyz_vectors: torch tensor of shape [batch, num_voxels, 3]
        :param noise: torch tensor of noise for reproducibility, e.g. torch.rand(batch_size, seq_length, device=sequence.device)
        :return:
            embeddings: [batch, num_voxels * num_unmasked_patch_tokens + 1 CLS token, hidden_size]
            mask: [batch, num_voxels, num_patch_tokens]
            ids_restore: [batch, num_voxels, num_patch_tokens]
        """
        batch, num_voxels, num_timepoints_per_node = signal_vectors.shape
        num_patch_tokens = num_timepoints_per_node // self.timepoint_patching_size

        # --- Embedding Projections ---#
        reshaped_signal_vectors = torch.reshape(
            signal_vectors, (batch, num_voxels, -1, self.timepoint_patching_size)
        )  # --> [batch, num_voxels, num_patch_tokens, timepoint_patching_size]
        signal_projection = self.signal_embedding_projection(reshaped_signal_vectors)

        # Project xyz coordinates into spatial embedding
        xyz_projection = self.xyz_embedding_projection(
            xyz_vectors
        )  # --> [batch, num_voxels, hidden_size]
        xyz_projection = xyz_projection.unsqueeze(2).repeat(1, 1, num_patch_tokens, 1)
        x = (
            signal_projection + xyz_projection
        )  # --> [batch, num_voxels, num_patch_tokens, hidden_size]

        # Add positional encoding for time signal
        x = self.pos_embedding(x)

        # Flatten num_brain_voxels and window_len dimensions
        x = torch.flatten(x, start_dim=1, end_dim=2)  # --> [batch, seq, hidden_size]

        # Random masking
        embeddings, mask, ids_restore = self.random_masking(x, noise=noise)

        # Append cls token
        cls_tokens = self.cls_token.expand(embeddings.shape[0], -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)  # CLS token idx 0
        return embeddings, mask, ids_restore
    
    def random_masking(self, sequence, noise=None):
        """
        Perform per-sample random masking by per-sample shuffling. Per-sample shuffling is done by argsort random
        noise.

        Args:
            sequence (`torch.LongTensor` of shape `(batch_size, sequence_length, dim)`)
            noise (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) which is
                mainly used for testing purposes to control randomness and maintain the reproducibility
        """
        batch_size, seq_length, dim = sequence.shape
        len_keep = int(seq_length * (1 - self.config.mask_ratio))

        if noise is None:
            noise = torch.rand(batch_size, seq_length, device=sequence.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        sequence_unmasked = torch.gather(sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([batch_size, seq_length], device=sequence.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return sequence_unmasked, mask, ids_restore

    def random_masking_4D(self, sequence):
        """
        Perform per-sample random masking by per-sample and per-voxel shuffling.
        Per-sample, per-voxel shuffling is done by argsort random
        noise.

        Args:
            sequence (`torch.LongTensor` of shape `(batch, num_voxels, num_patch_tokens, dim)`)
            noise (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) which is
                mainly used for testing purposes to control randomness and maintain the reproducibility
        """
        batch_size, num_voxels, num_patch_tokens, dim = sequence.shape
        len_keep = int(num_patch_tokens * (1 - self.config.mask_ratio))

        noise = torch.rand(
            batch_size, num_voxels, num_patch_tokens, device=sequence.device
        )  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=2
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=2)

        # keep the first subset
        ids_keep = ids_shuffle[:, :, :len_keep]
        sequence_unmasked = torch.gather(
            sequence, dim=2, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, dim)
        )

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones(
            [batch_size, num_voxels, num_patch_tokens], device=sequence.device
        )
        mask[:, :, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=2, index=ids_restore)

        return sequence_unmasked, mask, ids_restore

    def last_timepoints_masking(self, sequence):
        """
        Perform per-sample masking of last N timepoints in each sequence in the batch.

        Args:
            sequence (`torch.LongTensor` of shape `(batch_size, num_brain_voxels, window_len, dim)`)
        """
        batch_size, num_brain_voxels, num_patch_tokens, dim = sequence.shape
        num_tokens_to_mask = (
            self.num_last_timepoints_masked // self.timepoint_patching_size
        )
        len_keep = num_patch_tokens - num_tokens_to_mask

        # keep the first subset
        sequence_unmasked = sequence[:, :, :len_keep]

        # generate the binary mask: 0 is keep, 1 is mask
        mask = torch.ones(
            [batch_size, num_brain_voxels, num_patch_tokens], device=sequence.device
        )
        mask[:, :, :len_keep] = 0

        return sequence_unmasked, mask


class BrainLMEncoder(ViTMAEEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList(
            [NystromformerLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    # layer_head_mask,  Nystromformer doesn't accept head_mask
                )
            else:
                # layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)
                # Nystromformer attention layer does not accept head_mask parameter
                layer_outputs = layer_module(
                    hidden_states, output_attentions=output_attentions
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class BrainLMModel(ViTMAEModel):
    def __init__(self, config):
        super().__init__(config)
        # Overwrite self.embeddings class of superclass ViTMAEModel, rest of attributes are same
        self.embeddings = BrainLMEmbeddings(config)  # Embedding logic for fMRI data
        self.encoder = BrainLMEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        signal_vectors: torch.Tensor = None,
        xyz_vectors: torch.Tensor = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        noise: Optional[bool] = None,
    ) -> Union[Tuple, ViTMAEModelOutput]:

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # BrainLM embedding, rather than VitMAE Image Embedding
        embedding_output, mask, ids_restore = self.embeddings(
            signal_vectors, xyz_vectors, noise
        )

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        if not return_dict:
            return (sequence_output, mask) + encoder_outputs[1:]

        return ViTMAEModelOutput(
            last_hidden_state=sequence_output,
            mask=mask,
            ids_restore=ids_restore,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class BrainLMDecoder(ViTMAEDecoder):
    def __init__(self, config, num_patches):
        super().__init__(config, num_patches)
        self.decoder_pos_embed = None  # Not using positional embedding
        self.num_brain_voxels = config.num_brain_voxels
        self.mask_ratio = config.mask_ratio
        self.timepoint_patching_size = config.timepoint_patching_size
        self.use_tanh_decoder = config.use_tanh_decoder

        # Decoder has its own xyz projection
        self.decoder_xyz_projection = nn.Linear(3, config.hidden_size, bias=True)
        self.pos_embedding = PositionalEncoding(d_model=config.hidden_size)

        # Decoder Linear Attention Transformer Layers
        decoder_config = deepcopy(config)
        decoder_config.hidden_size = config.decoder_hidden_size
        decoder_config.num_hidden_layers = config.decoder_num_hidden_layers
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.intermediate_size = config.decoder_intermediate_size
        self.decoder_layers = nn.ModuleList(
            [
                NystromformerLayer(decoder_config)
                for _ in range(config.decoder_num_hidden_layers)
            ]
        )

        # Final Projection Head
        self.decoder_pred1 = nn.Linear(
            in_features=config.decoder_hidden_size,
            out_features=config.decoder_hidden_size // 2,
            bias=True,
        )
        self.decoder_pred_nonlinearity = nn.LeakyReLU(0.1)
        self.decoder_pred2 = nn.Linear(
            in_features=config.decoder_hidden_size // 2,
            out_features=self.timepoint_patching_size,  # Predict each token into the number of timepoints patched together
            bias=True,
        )

        if self.use_tanh_decoder:
            self.decoder_pred_nonlinearity2 = nn.Tanh()

        self.initialize_weights(num_patches)

    def initialize_weights(self, _):
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=self.config.initializer_range)

    def forward(
        self,
        hidden_states,
        xyz_vectors,
        ids_restore,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # embed tokens
        x = self.decoder_embed(hidden_states)

        # Unflatten sequence
        batch_size, flatten_seq_len, hidden_dim = x[:, 1:, :].shape
        num_mask_tokens = ids_restore.shape[1] - flatten_seq_len

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(batch_size, num_mask_tokens, 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, hidden_dim)
        )  # unshuffle
        # --> x_ is shape torch.Size([batch_size, num_voxels, num_tokens, hidden_size])

        num_patch_tokens = x_.shape[1] // self.num_brain_voxels
        x_ = torch.reshape(
            x_, shape=(batch_size, self.num_brain_voxels, num_patch_tokens, hidden_dim)
        )  # --> [batch_size, num_voxels, unmasked_timepoints_per_voxel, hidden_size]

        # Add spatial encoding and temporal encoding for decoder
        xyz_projection = self.decoder_xyz_projection(xyz_vectors)
        xyz_projection = xyz_projection.unsqueeze(2).repeat(1, 1, num_patch_tokens, 1)
        x_ = x_ + xyz_projection

        # Add positional encoding for time signal
        x_ = self.pos_embedding(x_)

        # Flatten again
        x_ = torch.flatten(x_, start_dim=1, end_dim=2)  # --> [batch, seq_len, dim]

        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        hidden_states = x  # No positional embedding

        # apply Transformer layers (blocks)
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.decoder_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    # None,  Nystromformer layer does not accept argument head_mask
                )
            else:
                # layer_outputs = layer_module(hidden_states, head_mask=None, output_attentions=output_attentions)
                # Nystromformer layer does not accept argument head_mask
                layer_outputs = layer_module(
                    hidden_states, output_attentions=output_attentions
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.decoder_norm(hidden_states)

        # predictor projection
        logits = self.decoder_pred1(hidden_states)
        logits = self.decoder_pred_nonlinearity(logits)
        logits = self.decoder_pred2(logits)
        if self.use_tanh_decoder:
            logits = self.decoder_pred_nonlinearity2(logits)

        # remove cls token
        logits = logits[:, 1:, :]  # --> [batch_size, flattened_seq]

        # Unflatten brain voxels and timepoints
        batch_size, flatten_seq_len, pred_timepoints = logits.shape
        logits = torch.reshape(
            logits,
            shape=(
                batch_size,
                self.num_brain_voxels,
                flatten_seq_len // self.num_brain_voxels,
                self.timepoint_patching_size,
            ),
        )  # --> [batch, num_voxels, num_patch_tokens, timepoint_patching_size]

        if not return_dict:
            return tuple(
                v
                for v in [logits, all_hidden_states, all_self_attentions]
                if v is not None
            )
        return ViTMAEDecoderOutput(
            logits=logits,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class BrainLMForPretraining(ViTMAEForPreTraining):
    """
    Model definition is for pretraining on single-cell datasets. Will calculate loss on forward
    pass through model.
    """

    def __init__(self, config):
        super().__init__(config)
        self.vit = BrainLMModel(config)
        self.decoder = BrainLMDecoder(
            config, num_patches=self.vit.embeddings.num_patches
        )

        # Initialize weights and apply final processing
        self.post_init()

    def init_weights(self):
        """
        If needed prunes and maybe initializes weights. If using a custom `PreTrainedModel`, you need to implement any
        initialization logic in `_init_weights`.
        """
        # Prune heads if needed
        if self.config.pruned_heads:
            self.prune_heads(self.config.pruned_heads)

        # Initialize weights
        self.apply(self._initialize_weights)

        # Tie weights should be skipped when not initializing all weights
        # since from_pretrained(...) calls tie weights anyways
        self.tie_weights()

    def _init_weights(self, module):  #
        if isinstance(module, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(module.weight)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.kaiming_uniform_(module.weight)

    def forward_loss(self, signal_values, pred_values, mask):
        """
        Args:
            signal_values: tensor of shape [batch_size, num_brain_voxels, num_tokens, timepoint_patch_preds]
            pred_values: tensor of shape [batch_size, num_brain_voxels, num_tokens, timepoint_patch_preds]
            mask: binary mask of shape [batch_size, num_brain_voxels, num_tokens], 1 means masked out
        Returns:
            `torch.FloatTensor`: Loss value.
        """
        assert signal_values.shape == pred_values.shape
        mask = mask.unsqueeze(-1).repeat(1, 1, 1, pred_values.shape[-1])

        if self.config.loss_fn == "mse":
            loss = (
                ((pred_values - signal_values) ** 2) * mask
            ).sum() / mask.sum()  # MSE
        elif self.config.loss_fn == "mae":
            loss = abs((pred_values - signal_values) * mask).sum() / mask.sum()  # MAE
        else:
            raise NotImplementedError("Unknown loss function specified.")

        return loss

    def forward(
        self,
        signal_vectors: torch.Tensor = None,
        xyz_vectors: torch.Tensor = None,
        labels: torch.Tensor = None,  # not used
        input_ids: torch.Tensor = None,  # not used, identical to torch.cat([expression_vectors, sampled_gene_indices]). Argument is here because input_ids will go to compute_metrics(), and we need expression vectors and sampled gene indices there
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        noise: Optional[bool] = None,
    ) -> Union[Tuple, ViTMAEForPreTrainingOutput]:

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Encoder will perform BrainLM fmri embedding rather than VitMAE Image Embedding
        outputs = self.vit(
            signal_vectors=signal_vectors,
            xyz_vectors=xyz_vectors,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            noise=noise,
        )

        latent = outputs.last_hidden_state
        ids_restore = outputs.ids_restore
        mask = outputs.mask

        decoder_outputs = self.decoder(latent, xyz_vectors, ids_restore)
        logits = (
            decoder_outputs.logits
        )  # shape (batch_size, num_voxels, num_future_timepoint_preds)

        # batch_size, num_voxels, num_tokens, patch_size = logits.shape
        input_signal_vectors_reshaped = torch.reshape(signal_vectors, logits.shape)
        mask = mask.reshape(logits.shape[:-1])
        loss = self.forward_loss(input_signal_vectors_reshaped, logits, mask)

        if loss.item() > 5.:
            print(f"Loss {loss.item():.5f} is a high value, check batch")

        if not return_dict:
            output = (logits, mask) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ViTMAEForPreTrainingOutput(
            loss=loss,
            logits=(logits, latent),
            mask=mask,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
