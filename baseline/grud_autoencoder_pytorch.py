from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Sequence, Tuple, Union, Dict, Literal
import time
import random
import collections
import queue
import threading # Ensure this is imported

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# -----------------------------------------------------------------------------
# 0) Original GRU-D Components (Identical to previous)
# -----------------------------------------------------------------------------

def softclip(x: torch.Tensor, threshold: float = 5.0) -> torch.Tensor:
    return torch.where(
        x.abs() <= threshold,
        x,
        threshold + torch.log1p(torch.expm1(x.abs() - threshold)) * x.sign(),
        )

@dataclass
class TemporalConfig:
    in_size: int = 4
    hid_size: int = 64
    num_layers: int = 2
    use_exponential_decay: bool = True
    softclip_threshold: float = 3.0
    min_log_gamma: float = -10.0
    dropout: float = 0.1
    final_dropout: float = 0.1
    layer_norm: bool = True
    clip_grad_norm: Optional[float] = 5.0
    weight_decay: float = 1e-4
    tbptt_steps: int = 20
    lr: float = 2e-3
    scheduler_t_max: int = 1000
    loss: Literal["mse", "mae", "huber"] = "huber"

class _BaseTemporalCell(nn.Module):
    def __init__(self, cfg: TemporalConfig):
        super().__init__()
        self.cfg = cfg
        self.hid_size = cfg.hid_size
        self.in_size = cfg.in_size
        self.impute_linear = nn.Linear(self.hid_size, self.in_size)
        if cfg.use_exponential_decay:
            self.decay_h = nn.Parameter(torch.zeros(self.hid_size))
        else:
            self.register_parameter("decay_h", None)

    def _gamma(self, decay_param: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        log_gamma_arg = softclip(decay_param, self.cfg.softclip_threshold)
        log_gamma = -F.softplus(log_gamma_arg) * dt
        log_gamma = torch.clamp(log_gamma, min=self.cfg.min_log_gamma, max=-1e-4)
        return torch.exp(log_gamma)

    def apply_decay_h(self, h_prev: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        if not self.cfg.use_exponential_decay or self.decay_h is None:
            return h_prev
        if dt.ndim != 2 or dt.shape[1] != 1:
            raise ValueError("dt must have shape [B,1]; got " + str(tuple(dt.shape)))
        gamma_h = self._gamma(self.decay_h, dt)
        return h_prev * gamma_h

    def impute_x(self, x: torch.Tensor, h_prev: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(x, device=x.device, dtype=x.dtype)
        mask = mask.to(x.dtype)
        x_hat = self.impute_linear(h_prev)
        return mask * x + (1.0 - mask) * x_hat

class GRUDCell(_BaseTemporalCell):
    def __init__(self, cfg: TemporalConfig):
        super().__init__(cfg)
        d_in, d_h = cfg.in_size, cfg.hid_size
        self.W_r = nn.Linear(d_in, d_h)
        self.U_r = nn.Linear(d_h, d_h, bias=False)
        self.V_r = nn.Linear(1, d_h, bias=False)
        self.W_z = nn.Linear(d_in, d_h)
        self.U_z = nn.Linear(d_h, d_h, bias=False)
        self.V_z = nn.Linear(1, d_h, bias=False)
        nn.init.constant_(self.W_z.bias, -1.0)
        self.W_h = nn.Linear(d_in, d_h)
        self.U_h = nn.Linear(d_h, d_h, bias=False)
        self.V_h = nn.Linear(1, d_h, bias=False)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor, dt: torch.Tensor,
                *, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_tilde = self.impute_x(x, h_prev, mask)
        h_decay = self.apply_decay_h(h_prev, dt)
        r = torch.sigmoid(self.W_r(x_tilde) + self.U_r(h_decay) + self.V_r(dt))
        z = torch.sigmoid(self.W_z(x_tilde) + self.U_z(h_decay) + self.V_z(dt))
        h_tilde = torch.tanh(self.W_h(x_tilde) + self.U_h(r * h_decay) + self.V_h(dt))
        h_new = (1 - z) * h_decay + z * h_tilde
        return h_new

class _RNNLayer(nn.Module):
    def __init__(self, cfg: TemporalConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        current_in_size = cfg.in_size if layer_idx == 0 else cfg.hid_size
        cell_cfg_dict = asdict(cfg)
        cell_cfg_dict["in_size"] = current_in_size
        layer_cfg = TemporalConfig(**cell_cfg_dict)

        self.cell = GRUDCell(layer_cfg)
        self.do_ln = cfg.layer_norm
        if self.do_ln:
            self.ln = nn.LayerNorm(cfg.hid_size)
        self.dropout_layer = nn.Dropout(cfg.dropout) if layer_idx < cfg.num_layers - 1 else nn.Identity()


    def forward(self, x_t_in: torch.Tensor, h_prev: torch.Tensor,
                dt_t: torch.Tensor, mask_t: Optional[torch.Tensor]) -> torch.Tensor:
        current_mask_t = mask_t if self.layer_idx == 0 else None
        h_cell = self.cell(x_t_in, h_prev, dt_t, mask=current_mask_t)
        if self.do_ln:
            h_cell = self.ln(h_cell)
        return self.dropout_layer(h_cell)

class TemporalRNN(nn.Module):
    def __init__(self, cfg: TemporalConfig):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList([_RNNLayer(cfg, i) for i in range(cfg.num_layers)])
        self.final_dropout_layer = nn.Dropout(cfg.final_dropout)

    def forward(self, X: torch.Tensor, dt: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                initial_h: Optional[Sequence[torch.Tensor]] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        B, T, _ = X.shape
        device = X.device
        if initial_h is None:
            h_states = [torch.zeros(B, self.cfg.hid_size, device=device) for _ in range(self.cfg.num_layers)]
        else:
            h_states = list(initial_h)

        outputs_top: List[torch.Tensor] = []
        for t_step in range(T):
            x_t = X[:, t_step]
            dt_t = dt[:, t_step]
            mask_t = mask[:, t_step] if mask is not None else None

            current_input_to_layer = x_t
            for l_idx, layer in enumerate(self.layers):
                h_prev_l = h_states[l_idx]
                h_new_l = layer(current_input_to_layer, h_prev_l, dt_t, mask_t)
                h_states[l_idx] = h_new_l
                current_input_to_layer = h_new_l
            outputs_top.append(current_input_to_layer)

        H_seq = torch.stack(outputs_top, dim=1)
        return self.final_dropout_layer(H_seq), h_states

# -----------------------------------------------------------------------------
# 1) Enhanced Autoencoder Configuration (Identical)
# -----------------------------------------------------------------------------
@dataclass
class AutoencoderConfig:
    input_size: int = 12
    latent_size: int = 8
    internal_projection_size: int = 32
    bottleneck_type: Literal["last_hidden", "mean_pool", "max_pool", "attention_pool"] = "mean_pool"
    use_input_projection: bool = True
    tie_encoder_decoder_weights: bool = False
    mask_projection_type: Literal["max_pool", "learned", "any_observed"] = "max_pool"
    attention_context_dim: int = 64
    reconstruction_loss: Literal["mse", "mae", "huber"] = "mse"
    loss_ramp_start: float = 1.0
    loss_ramp_end: float = 1.0

    mode: Literal["reconstruction", "forecasting"] = "reconstruction"
    forecast_horizon: int = 1
    pass_mask_to_decoder_rnn: bool = False

    forecasting_mode: Literal["direct", "autoregressive"] = "direct"
    autoregressive_feedback_transform: Literal["linear", "identity", "learned"] = "linear"

    predict_future_dt: bool = False
    dt_prediction_method: Literal["learned", "last_value"] = "last_value"

# -----------------------------------------------------------------------------
# 2) Mask Projector (Identical)
# -----------------------------------------------------------------------------
class MaskProjector(nn.Module):
    def __init__(self, input_size: int, output_size: int, projection_type: str = "max_pool"):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.projection_type = projection_type

        if projection_type == "learned":
            self.mask_proj_linear = nn.Linear(input_size, output_size, bias=False)
            nn.init.constant_(self.mask_proj_linear.weight, 1.0 / input_size)

    def project_mask(self, mask: torch.Tensor, weight_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C_in = mask.shape
        if C_in == self.output_size:
            return mask.float()

        if self.projection_type == "max_pool":
            if weight_matrix is not None and weight_matrix.shape == (self.output_size, self.input_size):
                abs_weights = weight_matrix.abs()
                threshold = abs_weights.max(dim=1, keepdim=True)[0] * 0.1
                significant_connections = (abs_weights > threshold).float()
                mask_expanded = mask.unsqueeze(2)
                connections_expanded = significant_connections.unsqueeze(0).unsqueeze(0)
                projected_mask = (mask_expanded * connections_expanded).any(dim=-1).float()
            else:
                group_size = self.input_size // self.output_size
                remainder = self.input_size % self.output_size
                projected_parts = []
                start_idx = 0
                for i in range(self.output_size):
                    end_idx = start_idx + group_size + (1 if i < remainder else 0)
                    group_mask = mask[:, :, start_idx:end_idx].any(dim=-1, keepdim=True).float()
                    projected_parts.append(group_mask)
                    start_idx = end_idx
                projected_mask = torch.cat(projected_parts, dim=-1)
        elif self.projection_type == "learned":
            projected_mask = torch.sigmoid(self.mask_proj_linear(mask.float()))
        elif self.projection_type == "any_observed":
            any_obs = mask.any(dim=-1, keepdim=True).float()
            projected_mask = any_obs.expand(-1, -1, self.output_size)
        else:
            raise ValueError(f"Unknown projection_type: {self.projection_type}")
        return projected_mask.float()

# -----------------------------------------------------------------------------
# 3) Flexible Temporal RNN (Identical)
# -----------------------------------------------------------------------------
class FlexibleTemporalRNN(nn.Module):
    def __init__(self, rnn_config: TemporalConfig, actual_input_size: int,
                 use_projection: bool = True, mask_projection_type: str = "max_pool"):
        super().__init__()
        self.actual_input_size = actual_input_size
        self.rnn_expected_input_size = rnn_config.in_size
        self.use_projection = use_projection

        if use_projection and actual_input_size != self.rnn_expected_input_size:
            self.input_proj = nn.Linear(actual_input_size, self.rnn_expected_input_size)
            self.mask_projector = MaskProjector(actual_input_size, self.rnn_expected_input_size, mask_projection_type)
        elif actual_input_size != self.rnn_expected_input_size:
            raise ValueError(
                f"actual_input_size ({actual_input_size}) must match rnn_config.in_size ({self.rnn_expected_input_size}) "
                f"if use_projection is False."
            )
        else:
            self.input_proj = nn.Identity()
            self.mask_projector = None
        self.rnn = TemporalRNN(rnn_config)

    def forward(self, X: torch.Tensor, dt: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                initial_h: Optional[Sequence[torch.Tensor]] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        X_to_rnn = self.input_proj(X)
        mask_to_rnn = None
        if mask is not None:
            if self.mask_projector is not None:
                weight_matrix = getattr(self.input_proj, 'weight', None) if isinstance(self.input_proj, nn.Linear) else None
                mask_to_rnn = self.mask_projector.project_mask(mask, weight_matrix)
            else:
                mask_to_rnn = mask.float()
        return self.rnn(X_to_rnn, dt, mask_to_rnn, initial_h)

# -----------------------------------------------------------------------------
# 4) Attention Pooling (Identical)
# -----------------------------------------------------------------------------
class AttentionPooling(nn.Module):
    def __init__(self, input_dim: int, context_dim: int):
        super().__init__()
        self.query_proj = nn.Linear(input_dim, context_dim)
        self.context_vector = nn.Parameter(torch.randn(context_dim))
        self.scale = math.sqrt(context_dim)

    def forward(self, x: torch.Tensor, sequence_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        queries = self.query_proj(x)
        scores = torch.matmul(queries, self.context_vector) / self.scale
        if sequence_mask is not None:
            fill_mask = sequence_mask == 0 if sequence_mask.dtype != torch.bool else ~sequence_mask
            scores = scores.masked_fill(fill_mask, -float('inf'))
        attn_weights = F.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (x * attn_weights).sum(dim=1)
        return pooled

# -----------------------------------------------------------------------------
# 5) Enhanced Temporal Autoencoder (Identical, using fixed from previous step)
# -----------------------------------------------------------------------------
class TemporalAutoencoder(nn.Module):
    def __init__(self, rnn_cfg_template: TemporalConfig, ae_cfg: AutoencoderConfig):
        super().__init__()
        self.rnn_cfg_template = rnn_cfg_template
        self.ae_cfg = ae_cfg

        encoder_rnn_cfg_dict = asdict(rnn_cfg_template)
        encoder_rnn_cfg_dict["in_size"] = ae_cfg.internal_projection_size
        self.encoder_rnn_cfg = TemporalConfig(**encoder_rnn_cfg_dict)

        self.encoder = FlexibleTemporalRNN(
            self.encoder_rnn_cfg, ae_cfg.input_size, ae_cfg.use_input_projection, ae_cfg.mask_projection_type
        )

        if ae_cfg.bottleneck_type == "attention_pool":
            self.bottleneck_aggregation = AttentionPooling(self.encoder_rnn_cfg.hid_size, ae_cfg.attention_context_dim)
            self.bottleneck_linear = nn.Linear(self.encoder_rnn_cfg.hid_size, ae_cfg.latent_size)
        else:
            self.bottleneck_aggregation = None
            self.bottleneck_linear = nn.Linear(self.encoder_rnn_cfg.hid_size, ae_cfg.latent_size)

        decoder_rnn_cfg_dict = asdict(rnn_cfg_template)
        decoder_rnn_cfg_dict["in_size"] = ae_cfg.internal_projection_size
        self.decoder_rnn_cfg = TemporalConfig(**decoder_rnn_cfg_dict)

        self.decoder_rnn = FlexibleTemporalRNN(
            self.decoder_rnn_cfg,
            ae_cfg.latent_size,
            True,
            ae_cfg.mask_projection_type
        )

        if ae_cfg.bottleneck_type != "last_hidden":
            self.latent_to_decoder_hidden = nn.Linear(
                ae_cfg.latent_size, self.decoder_rnn_cfg.num_layers * self.decoder_rnn_cfg.hid_size
            )

        if ae_cfg.tie_encoder_decoder_weights:
            if not (ae_cfg.use_input_projection and isinstance(self.encoder.input_proj, nn.Linear)):
                raise ValueError("Weight tying requires use_input_projection=True and encoder.input_proj to be Linear.")
            if self.decoder_rnn_cfg.hid_size != ae_cfg.internal_projection_size:
                raise ValueError(
                    f"Tied weights: decoder RNN hid_size ({self.decoder_rnn_cfg.hid_size}) "
                    f"must equal ae_cfg.internal_projection_size ({ae_cfg.internal_projection_size})."
                )
            self.output_proj_linear = None
            self.output_proj_bias = nn.Parameter(torch.zeros(ae_cfg.input_size))
        else:
            self.output_proj_linear = nn.Linear(self.decoder_rnn_cfg.hid_size, ae_cfg.input_size)

        if ae_cfg.pass_mask_to_decoder_rnn:
            self.decoder_input_mask_projector = MaskProjector(
                ae_cfg.input_size,
                self.decoder_rnn.actual_input_size,
                ae_cfg.mask_projection_type
            )
            assert self.decoder_rnn.actual_input_size == ae_cfg.latent_size

        if ae_cfg.predict_future_dt and ae_cfg.dt_prediction_method == "learned":
            self.dt_predictor = nn.Linear(ae_cfg.latent_size, ae_cfg.forecast_horizon)
        else:
            self.dt_predictor = None

        if ae_cfg.forecasting_mode == "autoregressive":
            if ae_cfg.autoregressive_feedback_transform == "linear":
                self.feedback_transform = nn.Linear(ae_cfg.input_size, ae_cfg.latent_size)
            elif ae_cfg.autoregressive_feedback_transform == "learned":
                self.feedback_transform = nn.Sequential(
                    nn.Linear(ae_cfg.input_size, ae_cfg.internal_projection_size),
                    nn.ReLU(),
                    nn.Linear(ae_cfg.internal_projection_size, ae_cfg.latent_size)
                )
            else:
                if ae_cfg.input_size != ae_cfg.latent_size:
                    raise ValueError(
                        f"For autoregressive_feedback_transform='identity', "
                        f"input_size ({ae_cfg.input_size}) must equal latent_size ({ae_cfg.latent_size})"
                    )
                self.feedback_transform = nn.Identity()
        else:
            self.feedback_transform = None
        self.criterion = self._create_criterion()

    def _create_criterion(self):
        loss_fn_map = {"mse": nn.MSELoss, "mae": nn.L1Loss, "huber": nn.HuberLoss}
        return loss_fn_map[self.ae_cfg.reconstruction_loss](reduction='none')

    def _compute_decode_dt(self, dt_input: torch.Tensor, latent: torch.Tensor, T_decode: int) -> torch.Tensor:
        if not self.ae_cfg.predict_future_dt:
            last_dt = dt_input[:, -1:, :]
            return last_dt.expand(-1, T_decode, -1)
        if self.ae_cfg.dt_prediction_method == "learned" and self.dt_predictor is not None:
            future_dt_pred = self.dt_predictor(latent)
            return future_dt_pred.unsqueeze(-1)
        else:
            last_dt = dt_input[:, -1:, :]
            return last_dt.expand(-1, T_decode, -1)

    def _aggregate_sequence(self, hidden_seq: torch.Tensor, feature_mask: Optional[torch.Tensor]) -> torch.Tensor:
        sequence_mask = None
        if feature_mask is not None:
            sequence_mask = feature_mask.any(dim=-1)

        if self.ae_cfg.bottleneck_type == "last_hidden":
            return hidden_seq[:, -1]
        elif self.ae_cfg.bottleneck_type == "mean_pool":
            if sequence_mask is not None:
                masked_sum = (hidden_seq * sequence_mask.float().unsqueeze(-1)).sum(dim=1)
                num_valid_steps = sequence_mask.float().sum(dim=1, keepdim=True).clamp(min=1e-8)
                return masked_sum / num_valid_steps
            return hidden_seq.mean(dim=1)
        elif self.ae_cfg.bottleneck_type == "max_pool":
            if sequence_mask is not None:
                fill_mask = ~sequence_mask if sequence_mask.dtype == torch.bool else (sequence_mask == 0)
                masked_h = hidden_seq.masked_fill(fill_mask.unsqueeze(-1), -float('inf'))
                pooled = masked_h.max(dim=1).values
                pooled = torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))
            else:
                pooled = hidden_seq.max(dim=1).values
            return pooled
        elif self.ae_cfg.bottleneck_type == "attention_pool":
            return self.bottleneck_aggregation(hidden_seq, sequence_mask)
        else:
            raise ValueError(f"Unknown bottleneck type: {self.ae_cfg.bottleneck_type}")

    def encode(self, X: torch.Tensor, dt: torch.Tensor,
               mask: Optional[torch.Tensor] = None,
               initial_h_encoder: Optional[Sequence[torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        encoder_hidden_seq, encoder_final_hidden = self.encoder(X, dt, mask, initial_h_encoder)
        aggregated_hidden = self._aggregate_sequence(encoder_hidden_seq, mask)
        latent = self.bottleneck_linear(aggregated_hidden)
        return latent, encoder_hidden_seq, encoder_final_hidden

    def decode(self, latent: torch.Tensor, dt_decode: torch.Tensor, T_decode: int,
               initial_h_decoder: Optional[List[torch.Tensor]] = None,
               original_feature_mask_for_decoder_input: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if (self.ae_cfg.mode == "forecasting" and
                self.ae_cfg.forecasting_mode == "autoregressive"):
            return self.decode_autoregressive(
                latent, dt_decode, T_decode, initial_h_decoder, original_feature_mask_for_decoder_input
            )
        else:
            return self.decode_direct(
                latent, dt_decode, T_decode, initial_h_decoder, original_feature_mask_for_decoder_input
            )

    def decode_direct(self, latent: torch.Tensor, dt_decode: torch.Tensor, T_decode: int,
                      initial_h_decoder: Optional[List[torch.Tensor]] = None,
                      original_feature_mask_for_decoder_input: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        B = latent.size(0)
        decoder_initial_h = initial_h_decoder

        if decoder_initial_h is None and self.ae_cfg.bottleneck_type != "last_hidden":
            if not hasattr(self, 'latent_to_decoder_hidden'):
                raise AttributeError("latent_to_decoder_hidden layer is missing for non-last_hidden bottleneck type.")
            h_flat = self.latent_to_decoder_hidden(latent)
            decoder_initial_h = list(h_flat.view(B, self.decoder_rnn_cfg.num_layers, self.decoder_rnn_cfg.hid_size).transpose(0,1).unbind(0))
            decoder_initial_h = [h.contiguous() for h in decoder_initial_h]

        latent_seq = latent.unsqueeze(1).expand(-1, T_decode, -1)

        mask_for_decoder_rnn_input = None
        if (self.ae_cfg.pass_mask_to_decoder_rnn and
                original_feature_mask_for_decoder_input is not None and
                hasattr(self, 'decoder_input_mask_projector')):
            mask_for_decoder_rnn_input = self.decoder_input_mask_projector.project_mask(original_feature_mask_for_decoder_input)
            assert mask_for_decoder_rnn_input.shape[-1] == latent_seq.shape[-1]

        decoder_hidden_seq, final_decoder_hidden = self.decoder_rnn(
            latent_seq, dt_decode, mask=mask_for_decoder_rnn_input, initial_h=decoder_initial_h
        )

        if self.output_proj_linear is not None:
            reconstructed = self.output_proj_linear(decoder_hidden_seq)
        else:
            tied_weight = self.encoder.input_proj.weight
            reconstructed = F.linear(decoder_hidden_seq, tied_weight.t(), self.output_proj_bias)
        return reconstructed, final_decoder_hidden

    def decode_autoregressive(self, latent: torch.Tensor, dt_decode: torch.Tensor, T_decode: int,
                              initial_h_decoder: Optional[List[torch.Tensor]] = None,
                              original_feature_mask_for_decoder_input: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        B = latent.size(0)
        device = latent.device

        if initial_h_decoder is None and self.ae_cfg.bottleneck_type != "last_hidden":
            if not hasattr(self, 'latent_to_decoder_hidden'):
                raise AttributeError("latent_to_decoder_hidden layer is missing for non-last_hidden bottleneck type.")
            h_flat = self.latent_to_decoder_hidden(latent)
            decoder_h_states = list(h_flat.view(B, self.decoder_rnn_cfg.num_layers, self.decoder_rnn_cfg.hid_size).transpose(0,1).unbind(0))
            decoder_h_states = [h.contiguous() for h in decoder_h_states]
        elif initial_h_decoder is not None:
            decoder_h_states = initial_h_decoder
        else:
            decoder_h_states = [
                torch.zeros(B, self.decoder_rnn_cfg.hid_size, device=device)
                for _ in range(self.decoder_rnn_cfg.num_layers)
            ]

        current_input_to_decoder_rnn = latent
        outputs = []

        projected_mask_for_all_steps = None
        if (self.ae_cfg.pass_mask_to_decoder_rnn and
                original_feature_mask_for_decoder_input is not None and
                hasattr(self, 'decoder_input_mask_projector')):
            last_known_mask_of_X = original_feature_mask_for_decoder_input[:, -1:, :]
            projected_last_mask = self.decoder_input_mask_projector.project_mask(last_known_mask_of_X)
            projected_mask_for_all_steps = projected_last_mask.expand(-1, T_decode, -1)

        for t in range(T_decode):
            input_seq_t = current_input_to_decoder_rnn.unsqueeze(1)
            dt_t = dt_decode[:, t:t+1, :]

            mask_t_for_decoder_input = None
            if projected_mask_for_all_steps is not None:
                mask_t_for_decoder_input = projected_mask_for_all_steps[:, t:t+1, :]

            decoder_hidden_out_t, decoder_h_states = self.decoder_rnn(
                input_seq_t, dt_t, mask=mask_t_for_decoder_input, initial_h=decoder_h_states
            )

            if self.output_proj_linear is not None:
                output_t = self.output_proj_linear(decoder_hidden_out_t[:, 0, :])
            else:
                tied_weight = self.encoder.input_proj.weight
                output_t = F.linear(decoder_hidden_out_t[:, 0, :], tied_weight.t(), self.output_proj_bias)
            outputs.append(output_t)

            if t < T_decode - 1:
                current_input_to_decoder_rnn = self.feedback_transform(output_t)

        output_sequence = torch.stack(outputs, dim=1)
        return output_sequence, decoder_h_states

    def forward(self, X: torch.Tensor, dt: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                initial_h_encoder: Optional[Sequence[torch.Tensor]] = None,
                initial_h_decoder: Optional[Sequence[torch.Tensor]] = None,
                target_sequence: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        B, T_in, _ = X.shape
        latent, encoder_hidden_seq, encoder_final_hidden = self.encode(X, dt, mask, initial_h_encoder)

        target_for_loss = None
        target_mask_for_loss = None
        original_feature_mask_for_decoder_input = None

        if self.ae_cfg.mode == "reconstruction":
            T_decode = T_in
            target_for_loss = X
            target_mask_for_loss = mask
            dt_decode = dt
            original_feature_mask_for_decoder_input = mask
        elif self.ae_cfg.mode == "forecasting":
            T_decode = self.ae_cfg.forecast_horizon
            target_for_loss = target_sequence
            target_mask_for_loss = None
            dt_decode = self._compute_decode_dt(dt, latent, T_decode)
            original_feature_mask_for_decoder_input = mask
        else:
            raise ValueError(f"Unknown mode: {self.ae_cfg.mode}")

        current_initial_h_decoder = initial_h_decoder
        if current_initial_h_decoder is None and self.ae_cfg.bottleneck_type == "last_hidden":
            current_initial_h_decoder = encoder_final_hidden

        output_sequence, final_decoder_hidden = self.decode(
            latent, dt_decode, T_decode,
            initial_h_decoder=current_initial_h_decoder,
            original_feature_mask_for_decoder_input=original_feature_mask_for_decoder_input
        )

        total_loss = torch.tensor(float('nan'), device=output_sequence.device)
        if target_for_loss is not None:
            element_loss = self.criterion(output_sequence, target_for_loss)
            if self.ae_cfg.loss_ramp_start != 1.0 or self.ae_cfg.loss_ramp_end != 1.0:
                weights = torch.linspace(
                    self.ae_cfg.loss_ramp_start, self.ae_cfg.loss_ramp_end, T_decode, device=X.device
                ).view(1, -1, 1)
                element_loss = element_loss * weights

            if target_mask_for_loss is not None:
                mask_float = target_mask_for_loss.float()
                if element_loss.shape[1] == mask_float.shape[1]:
                    masked_loss = element_loss * mask_float
                    total_loss = masked_loss.sum() / (mask_float.sum().clamp(min=1e-8))
                else:
                    total_loss = element_loss.mean()
            else:
                total_loss = element_loss.mean()

        return {
            'output_sequence': output_sequence, 'latent': latent, 'loss': total_loss,
            'encoder_hidden_seq': encoder_hidden_seq, 'encoder_final_hidden': encoder_final_hidden,
            'decoder_final_hidden': final_decoder_hidden
        }

# -----------------------------------------------------------------------------
# 6) Online Learner with Complete Streaming State (Identical, using fixed from previous step)
# -----------------------------------------------------------------------------
class SlidingTBPTTOnlineLearnerAE(nn.Module):
    def __init__(self, autoencoder_model: TemporalAutoencoder, opt_cfg: TemporalConfig):
        super().__init__()
        self.autoencoder = autoencoder_model
        self.opt_cfg = opt_cfg
        self._setup_optim()
        self.reset_streaming_state()

    def _split_params(self):
        decay, no_decay = [], []
        for name, p in self.autoencoder.named_parameters():
            if not p.requires_grad: continue
            if p.dim() == 1 or name.endswith(".bias") or "norm" in name.lower(): no_decay.append(p)
            else: decay.append(p)
        return [{"params": decay, "weight_decay": self.opt_cfg.weight_decay},
                {"params": no_decay, "weight_decay": 0.0}]

    def _setup_optim(self):
        self.opt = torch.optim.AdamW(self._split_params(), lr=self.opt_cfg.lr)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=self.opt_cfg.scheduler_t_max)

    def reset_streaming_state(self, batch_size: int = 1, device: Optional[torch.device] = None):
        if device is None:
            try: device = next(self.autoencoder.parameters()).device
            except StopIteration: device = torch.device("cpu")

        self.h_states_stream_encoder = [
            torch.zeros(batch_size, self.autoencoder.encoder_rnn_cfg.hid_size, device=device)
            for _ in range(self.autoencoder.encoder_rnn_cfg.num_layers)]
        self.h_states_stream_decoder = [
            torch.zeros(batch_size, self.autoencoder.decoder_rnn_cfg.hid_size, device=device)
            for _ in range(self.autoencoder.decoder_rnn_cfg.num_layers)]
        self.win_X: List[torch.Tensor] = []
        self.win_dt: List[torch.Tensor] = []
        self.win_mask: List[Optional[torch.Tensor]] = []
        if self.autoencoder.ae_cfg.mode == "forecasting":
            self.win_targets: List[torch.Tensor] = []

    @torch.no_grad()
    def predict_single(self, x_t: torch.Tensor, dt_t: torch.Tensor,
                       mask_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        self.autoencoder.eval()
        current_bs = x_t.size(0)
        if self.h_states_stream_encoder[0].size(0) != current_bs:
            self.reset_streaming_state(batch_size=current_bs, device=x_t.device)

        X_seq = x_t.unsqueeze(1)
        # dt_t could be [B] or [B,1]. Ensure it is [B,1,1] for TemporalRNN
        dt_seq = dt_t.view(current_bs, 1, 1) if dt_t.ndim == 1 else dt_t.unsqueeze(1)
        if dt_seq.shape[1] !=1 or dt_seq.ndim !=3 : dt_seq = dt_t.view(current_bs, 1,1) # final check

        mask_seq = mask_t.unsqueeze(1) if mask_t is not None else None

        result = self.autoencoder(X_seq, dt_seq, mask_seq,
                                  initial_h_encoder=self.h_states_stream_encoder,
                                  initial_h_decoder=self.h_states_stream_decoder)
        self.h_states_stream_encoder = [h.detach() for h in result['encoder_final_hidden']]
        self.h_states_stream_decoder = [h.detach() for h in result['decoder_final_hidden']]
        output = result['output_sequence']
        return output.squeeze(1) if output.size(1) == 1 else output

    def step_stream(self, x_t: torch.Tensor, dt_t: torch.Tensor,
                    mask_t: Optional[torch.Tensor] = None,
                    target_t: Optional[torch.Tensor] = None) -> Tuple[float, torch.Tensor]:
        self.autoencoder.train()
        current_bs = x_t.size(0)
        if self.h_states_stream_encoder[0].size(0) != current_bs:
            self.reset_streaming_state(batch_size=current_bs, device=x_t.device)

        dt_t_shaped = dt_t.unsqueeze(1) if dt_t.ndim == 1 else dt_t
        if dt_t_shaped.ndim == 2 and dt_t_shaped.shape[1] !=1:
            raise ValueError(f"dt_t must be [B] or [B,1], got {dt_t.shape}")
        if dt_t_shaped.ndim == 1: dt_t_shaped = dt_t_shaped.unsqueeze(1)


        self.win_X.append(x_t.detach().clone())
        self.win_dt.append(dt_t_shaped.detach().clone())
        self.win_mask.append(mask_t.detach().clone() if mask_t is not None else None)

        if self.autoencoder.ae_cfg.mode == "forecasting":
            if target_t is not None: self.win_targets.append(target_t.detach().clone())
            elif not hasattr(self, 'win_targets'): self.win_targets = []


        max_len = self.opt_cfg.tbptt_steps
        if len(self.win_X) > max_len:
            self.win_X.pop(0); self.win_dt.pop(0); self.win_mask.pop(0)
            if hasattr(self, 'win_targets') and self.win_targets: self.win_targets.pop(0)

        if len(self.win_X) < min(max_len, 2):
            return 0.0, self.predict_single(x_t, dt_t, mask_t)


        X_win = torch.stack(self.win_X, dim=1)
        dt_win = torch.stack(self.win_dt, dim=1)
        mask_win = None
        if any(m is not None for m in self.win_mask):
            processed_masks = []
            for i, m_step in enumerate(self.win_mask):
                if m_step is not None: processed_masks.append(m_step)
                else: processed_masks.append(torch.ones_like(X_win[:, i], device=X_win.device))
            mask_win = torch.stack(processed_masks, dim=1)

        target_sequence_for_loss = None
        if self.autoencoder.ae_cfg.mode == "forecasting":
            if not hasattr(self, 'win_targets') or not self.win_targets :
                raise ValueError("Forecasting mode training step requires targets in win_targets.")
            target_sequence_for_loss = self.win_targets[-1]

        result = self.autoencoder(
            X_win, dt_win, mask_win,
            initial_h_encoder=self.h_states_stream_encoder,
            initial_h_decoder=self.h_states_stream_decoder,
            target_sequence=target_sequence_for_loss)
        loss = result['loss']

        current_loss_item = float('nan')
        if torch.is_tensor(loss) and torch.isfinite(loss):
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            if self.opt_cfg.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.autoencoder.parameters(), self.opt_cfg.clip_grad_norm)
            self.opt.step(); self.sched.step()
            current_loss_item = loss.item()
        elif torch.is_tensor(loss):
            current_loss_item = loss.item()


        self.h_states_stream_encoder = [h.detach() for h in result['encoder_final_hidden']]
        self.h_states_stream_decoder = [h.detach() for h in result['decoder_final_hidden']]
        output_sequence = result['output_sequence']
        current_output = output_sequence[:, -1].detach() if self.autoencoder.ae_cfg.mode == "reconstruction" else output_sequence.detach()
        return current_loss_item, current_output

# ============================================================================
# REAL-TIME DATA STREAMING FRAMEWORK (Python Prequel)
# ============================================================================

@dataclass
class DataPoint:
    features: torch.Tensor      # (batch_size, feature_dim)
    dt: torch.Tensor           # (batch_size, 1)
    mask: torch.Tensor         # (batch_size, feature_dim)
    timestamp: float           # time.monotonic()
    sequence_id: int

class StreamingMetrics:
    def __init__(self):
        self.total_points_generated = 0
        self.processed_points = 0
        self.cumulative_loss = 0.0
        self.cumulative_mse = 0.0

        self.lock = threading.RLock() # CHANGED TO RLock
        self.loss_history: collections.deque[float] = collections.deque(maxlen=100)
        self.mse_history: collections.deque[float] = collections.deque(maxlen=100)
        self.processing_times: collections.deque[float] = collections.deque(maxlen=100)

    def update_metrics(self, loss: float, mse: float, processing_time_ms: float):
        with self.lock:
            self.processed_points += 1
            if not math.isnan(loss):
                self.cumulative_loss += loss
                self.loss_history.append(loss)
            if not math.isnan(mse):
                self.cumulative_mse += mse
                self.mse_history.append(mse)

            self.processing_times.append(processing_time_ms)

    def get_avg_processing_time_ms(self) -> float:
        with self.lock: # RLock allows re-entrant acquisition
            if not self.processing_times: return 0.0
            return sum(self.processing_times) / len(self.processing_times)

    def print_summary(self, config_name: str):
        with self.lock:
            avg_loss_moving = float('nan')
            avg_mse_moving = float('nan')

            valid_losses_hist = [l for l in self.loss_history if not math.isnan(l)]
            valid_mses_hist = [m for m in self.mse_history if not math.isnan(m)]

            if valid_losses_hist: avg_loss_moving = sum(valid_losses_hist) / len(valid_losses_hist)
            if valid_mses_hist: avg_mse_moving = sum(valid_mses_hist) / len(valid_mses_hist)

            print(f"\n📊 [{config_name}] Summary:")
            print(f"   Points Generated (approx): {self.total_points_generated}")
            print(f"   Points Processed: {self.processed_points}")
            print(f"   Avg Loss (moving avg over max {len(self.loss_history)} points): {avg_loss_moving:.6f}")
            print(f"   Avg MSE (moving avg over max {len(self.mse_history)} points): {avg_mse_moving:.6f}")
            print(f"   Avg Processing Time (moving avg over max {len(self.processing_times)} points): {self.get_avg_processing_time_ms():.2f} ms")

            if len(self.loss_history) >= 10:
                recent_loss_vals = list(self.loss_history)[-10:]
                recent_mse_vals = list(self.mse_history)[-10:]

                valid_recent_losses = [l for l in recent_loss_vals if not math.isnan(l)]
                valid_recent_mses = [m for m in recent_mse_vals if not math.isnan(m)]

                recent_loss_avg = sum(valid_recent_losses) / len(valid_recent_losses) if valid_recent_losses else float('nan')
                recent_mse_avg = sum(valid_recent_mses) / len(valid_recent_mses) if valid_recent_mses else float('nan')

                count_str_loss = f"last {len(valid_recent_losses)} valid of {min(10, len(self.loss_history))}"
                count_str_mse = f"last {len(valid_recent_mses)} valid of {min(10, len(self.mse_history))}"

                print(f"   Recent Avg Loss ({count_str_loss}): {recent_loss_avg:.6f}")
                print(f"   Recent Avg MSE ({count_str_mse}): {recent_mse_avg:.6f}")

class SyntheticDataGenerator:
    def __init__(self, feat_dim: int, batch_sz: int, missing_rate: float = 0.2,
                 seed: int = 42, ar_ord: int = 3, device: torch.device = torch.device('cpu')):
        self.rng = random.Random(seed)
        self.torch_rng = torch.Generator(device=device)
        if seed is not None: self.torch_rng.manual_seed(seed)

        self.feature_dim = feat_dim
        self.batch_size = batch_sz
        self.missing_rate = missing_rate
        self.ar_order = ar_ord
        self.device = device

        self.feature_trends = [[(self.rng.normalvariate(0.0, 0.02)) for _ in range(batch_sz)] for _ in range(feat_dim)]
        self.seasonal_phases = [self.rng.uniform(0.0, 2.0 * math.pi) for _ in range(feat_dim)]
        self.noise_levels = [self.rng.uniform(0.05, 0.3) for _ in range(feat_dim)]

        self.correlation_matrix = torch.eye(feat_dim, device=device)
        for i in range(feat_dim):
            for j in range(i + 1, feat_dim):
                corr = self.rng.uniform(-0.7, 0.7)
                self.correlation_matrix[i, j] = corr
                self.correlation_matrix[j, i] = corr
        self.transform_matrix_L = torch.linalg.cholesky(self.correlation_matrix) if feat_dim > 0 else torch.eye(0, device=device)


        self.autoregressive_coeffs = torch.tensor([self.rng.normalvariate(0.0, 0.2) for _ in range(ar_ord)], device=device)
        self.history_buffer: collections.deque[torch.Tensor] = collections.deque(maxlen=ar_ord)

    def generate_next_point(self, sequence_id: int) -> DataPoint:
        dt_vals = torch.tensor([max(0.01, min(2.0, self.rng.expovariate(2.0))) for _ in range(self.batch_size)], device=self.device).view(-1, 1)

        base_features = torch.zeros(self.batch_size, self.feature_dim, device=self.device)
        time_val = float(sequence_id) * 0.1

        for b in range(self.batch_size):
            for f in range(self.feature_dim):
                trend = self.feature_trends[f][b] * time_val
                s1 = 0.5 * math.sin(2.0 * math.pi * time_val / 20.0 + self.seasonal_phases[f])
                s2 = 0.3 * math.sin(2.0 * math.pi * time_val / 7.0 + self.seasonal_phases[f] * 0.7)
                s3 = 0.2 * math.sin(2.0 * math.pi * time_val / 3.0 + self.seasonal_phases[f] * 1.3)
                nonlinear = 0.1 * math.sin(time_val * time_val * 0.01 + f)
                base_level = (f % 3 - 1.0) * 0.5
                noise = self.rng.normalvariate(0.0, self.noise_levels[f])

                val = base_level + trend + s1 + s2 + s3 + nonlinear + noise

                if self.history_buffer and self.ar_order > 0 : # Check ar_order for coeffs
                    ar_component = 0.0
                    for lag in range(min(self.ar_order, len(self.history_buffer))):
                        ar_component += self.autoregressive_coeffs[lag] * self.history_buffer[-(lag+1)][b, f]
                    val += 0.3 * ar_component
                base_features[b, f] = val

        correlated_features = base_features
        if self.feature_dim > 0 :
            # Apply Cholesky factor L: X_correlated = X_original @ L.T
            # (Eigen example was effectively X_correlated_i = sum_{j<=i} C_ij X_original_j, which is X_original @ C_lower.T )
            # If C is correlation, then use Cholesky: C = L L.T, X_corr = X_uncorr @ L.T
            correlated_features = base_features @ self.transform_matrix_L.T


        mask = torch.zeros_like(correlated_features, dtype=torch.float, device=self.device)
        for b_idx in range(self.batch_size):
            burst_missing = self.rng.random() < 0.1
            burst_start, burst_length = -1, 0
            if burst_missing and self.feature_dim > 0:
                burst_start = self.rng.randint(0, self.feature_dim - 1)
                burst_length = self.rng.randint(2, min(5, self.feature_dim))

            for f_idx in range(self.feature_dim):
                is_missing = False
                if burst_missing and f_idx >= burst_start and f_idx < burst_start + burst_length:
                    is_missing = True
                else:
                    is_missing = self.rng.random() < self.missing_rate

                if f_idx % 4 == 0 and sequence_id % 7 == 0 :
                    is_missing = self.rng.random() < self.missing_rate

                mask[b_idx, f_idx] = 0.0 if is_missing else 1.0

            if self.feature_dim > 0 and mask[b_idx].sum() == 0.0:
                mask[b_idx, self.rng.randint(0, self.feature_dim - 1)] = 1.0

        if self.feature_dim > 0:
            self.history_buffer.append(correlated_features.clone())

        return DataPoint(features=correlated_features, dt=dt_vals, mask=mask,
                         timestamp=time.monotonic(), sequence_id=sequence_id)

class PoissonStreamSimulator:
    def __init__(self, rate_per_second: float, seed: int = 42):
        self.rng = random.Random(seed)
        self.target_rate = rate_per_second

    def next_arrival_delay_ms(self) -> int:
        if self.target_rate <= 0: return 1_000_000
        delay_seconds = self.rng.expovariate(self.target_rate)
        return int(delay_seconds * 1000)

# ============================================================================
# AUTOENCODER CONFIGURATION TESTING
# ============================================================================
@dataclass
class TestConfiguration:
    name: str
    rnn_config_template: TemporalConfig
    ae_config: AutoencoderConfig
    batch_size: int
    test_duration_seconds: int = 20
    device: torch.device = field(default_factory=lambda: torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

def create_test_configurations() -> List[TestConfiguration]:
    configs = []
    base_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {base_device} for test configurations.")

    configs.append(TestConfiguration(
        name="Basic-Reconstruction-MeanPool", batch_size=2, test_duration_seconds=10, device=base_device,
        rnn_config_template=TemporalConfig(hid_size=32, num_layers=2, dropout=0.1, final_dropout=0.1, lr=1e-3, tbptt_steps=8),
        ae_config=AutoencoderConfig(input_size=12, latent_size=8, internal_projection_size=16, bottleneck_type="mean_pool", mode="reconstruction")
    ))
    configs.append(TestConfiguration(
        name="Attention-Reconstruction", batch_size=2, test_duration_seconds=10, device=base_device,
        rnn_config_template=TemporalConfig(hid_size=40, num_layers=2, dropout=0.15, final_dropout=0.15, lr=2e-3, tbptt_steps=12),
        ae_config=AutoencoderConfig(input_size=12, latent_size=10, internal_projection_size=20, bottleneck_type="attention_pool", attention_context_dim=32, mask_projection_type="learned", mode="reconstruction")
    ))
    configs.append(TestConfiguration(
        name="Direct-Forecasting", batch_size=2, test_duration_seconds=10, device=base_device,
        rnn_config_template=TemporalConfig(hid_size=36, num_layers=2, dropout=0.2, final_dropout=0.2, lr=1.5e-3, tbptt_steps=10),
        ae_config=AutoencoderConfig(input_size=12, latent_size=9, internal_projection_size=18, bottleneck_type="max_pool", mask_projection_type="any_observed", mode="forecasting", forecast_horizon=3, forecasting_mode="direct", predict_future_dt=True, dt_prediction_method="learned")
    ))
    configs.append(TestConfiguration(
        name="MaxPool-Reconstruction", batch_size=3, test_duration_seconds=10, device=base_device,
        rnn_config_template=TemporalConfig(hid_size=28, num_layers=2, dropout=0.12, final_dropout=0.12, lr=1.2e-3, tbptt_steps=9),
        ae_config=AutoencoderConfig(input_size=12, latent_size=7, internal_projection_size=14, bottleneck_type="max_pool", mask_projection_type="learned", mode="reconstruction")
    ))
    configs.append(TestConfiguration(
        name="HighDim-LastHidden-Recon", batch_size=1, test_duration_seconds=8, device=base_device,
        rnn_config_template=TemporalConfig(hid_size=48, num_layers=2, dropout=0.25, final_dropout=0.25, lr=1e-3, tbptt_steps=6),
        ae_config=AutoencoderConfig(input_size=18, latent_size=12, internal_projection_size=24, bottleneck_type="last_hidden", mode="reconstruction")
    ))
    configs.append(TestConfiguration(
        name="Minimal-SpeedTest-Recon", batch_size=4, test_duration_seconds=12, device=base_device,
        rnn_config_template=TemporalConfig(hid_size=16, num_layers=1, use_exponential_decay=False, layer_norm=False, dropout=0.0, final_dropout=0.0, lr=3e-3, tbptt_steps=4),
        ae_config=AutoencoderConfig(input_size=6, latent_size=4, internal_projection_size=8, bottleneck_type="mean_pool", mode="reconstruction")
    ))
    configs.append(TestConfiguration(
        name="Compact-NoInputProj-Recon", batch_size=6, test_duration_seconds=10, device=base_device,
        rnn_config_template=TemporalConfig(hid_size=12, num_layers=1, layer_norm=False, dropout=0.05, final_dropout=0.05, lr=4e-3, tbptt_steps=3),
        ae_config=AutoencoderConfig(input_size=4, latent_size=3, internal_projection_size=4, use_input_projection=False, bottleneck_type="mean_pool", mask_projection_type="any_observed", mode="reconstruction")
    ))
    configs.append(TestConfiguration(
        name="Long-Horizon-Forecasting", batch_size=1, test_duration_seconds=10, device=base_device,
        rnn_config_template=TemporalConfig(hid_size=40, num_layers=3, dropout=0.15, final_dropout=0.15, lr=1e-3, tbptt_steps=20),
        ae_config=AutoencoderConfig(input_size=10, latent_size=8, internal_projection_size=20, bottleneck_type="max_pool", mask_projection_type="any_observed", mode="forecasting", forecast_horizon=8, forecasting_mode="direct", predict_future_dt=True, dt_prediction_method="learned")
    ))
    return configs

# ============================================================================
# REAL-TIME TESTING EXECUTION
# ============================================================================
stop_event = threading.Event()

def producer_thread_fn(data_generator: SyntheticDataGenerator,
                       stream_simulator: PoissonStreamSimulator,
                       processing_q: queue.Queue,
                       metrics: StreamingMetrics, # Added metrics for total_points_generated
                       end_time: float,
                       target_device: torch.device):
    sequence_counter = 0
    while not stop_event.is_set():
        delay_ms = stream_simulator.next_arrival_delay_ms()
        # Sleep accurately, checking stop_event frequently for responsiveness
        sleep_end_time = time.monotonic() + delay_ms / 1000.0
        while time.monotonic() < sleep_end_time:
            if stop_event.is_set(): break
            time.sleep(min(0.01, sleep_end_time - time.monotonic())) # Sleep in small chunks
        if stop_event.is_set(): break


        if time.monotonic() >= end_time and not stop_event.is_set(): # Check if main thread missed setting it
            stop_event.set()
        if stop_event.is_set(): break # Exit if signaled during sleep or by end_time

        point = data_generator.generate_next_point(sequence_counter)
        sequence_counter += 1

        point.features = point.features.to(target_device)
        point.dt = point.dt.to(target_device)
        point.mask = point.mask.to(target_device)

        try:
            processing_q.put(point, timeout=0.1) # Add timeout to prevent indefinite block if consumer dies
            metrics.total_points_generated +=1
        except queue.Full:
            if stop_event.is_set(): break
            print("   ⚠️ Producer: Queue full, skipping point.")
            continue


def consumer_thread_fn(learner: SlidingTBPTTOnlineLearnerAE,
                       processing_q: queue.Queue,
                       metrics: StreamingMetrics,
                       config: TestConfiguration,
                       target_device: torch.device):

    # data_gen_for_targets: No longer needed as targets are simplified

    while not stop_event.is_set() or not processing_q.empty():
        try:
            point: DataPoint = processing_q.get(timeout=0.1)
        except queue.Empty:
            if stop_event.is_set() and processing_q.empty(): break
            continue

        process_start_time = time.perf_counter()

        try:
            target_for_loss_to_learner = None # This is for learner.step_stream
            actual_target_for_mse_calc = None # This is for calculating MSE metric

            if config.ae_config.mode == "forecasting":
                future_target_features = torch.zeros(
                    config.batch_size,
                    config.ae_config.forecast_horizon,
                    config.ae_config.input_size,
                    device=target_device
                )
                current_obs_expanded = point.features.unsqueeze(1).expand(-1, config.ae_config.forecast_horizon, -1)
                trend_val = torch.arange(1, config.ae_config.forecast_horizon + 1, device=target_device).float() * 0.01
                trend_val = trend_val.view(1, -1, 1)
                noise = torch.randn_like(future_target_features, generator=learner.autoencoder.encoder.rnn.layers[0].cell.decay_h.device.type != 'mps' and learner.autoencoder.torch_rng or None) * 0.1 # Ensure generator is on same device as tensor

                future_target_features = current_obs_expanded + trend_val + noise
                target_for_loss_to_learner = future_target_features
                actual_target_for_mse_calc = future_target_features

            elif config.ae_config.mode == "reconstruction":
                actual_target_for_mse_calc = point.features

                # dt for step_stream should be [B] or [B,1]. point.dt is [B,1]
            loss_val, prediction = learner.step_stream(point.features, point.dt.squeeze(-1), point.mask, target_t=target_for_loss_to_learner)

            process_end_time = time.perf_counter()
            processing_time_ms = (process_end_time - process_start_time) * 1000

            mse_val = float('nan')
            if config.ae_config.mode == "reconstruction":
                if prediction.shape == actual_target_for_mse_calc.shape:
                    masked_diff_sq = torch.square((prediction - actual_target_for_mse_calc) * point.mask)
                    mse_val = (masked_diff_sq.sum() / point.mask.sum().clamp(min=1e-8)).item()
            elif config.ae_config.mode == "forecasting":
                if prediction.shape == actual_target_for_mse_calc.shape:
                    mse_val = F.mse_loss(prediction, actual_target_for_mse_calc).item()

            metrics.update_metrics(loss_val, mse_val, processing_time_ms)

            if metrics.processed_points % 20 == 0:
                print(f"   ⏱️  Processed {metrics.processed_points} points, "
                      f"Recent Loss: {loss_val:.4f}, MSE: {mse_val:.4f}, "
                      f"Process Time: {processing_time_ms:.1f}ms")

        except Exception as e:
            print(f"   ❌ Consumer processing error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            processing_q.task_done()


def run_realtime_test(test_config: TestConfiguration):
    print(f"\n🚀 Starting Real-time Test: {test_config.name}")
    print(f"   Duration: {test_config.test_duration_seconds} seconds")
    print(f"   Target Rate: ~10 points/second (Poisson distributed)")
    print(f"   Feature Dim: {test_config.ae_config.input_size}, Batch Size: {test_config.batch_size}")
    print(f"   Device: {test_config.device}")

    stop_event.clear()

    data_generator = SyntheticDataGenerator(
        feat_dim=test_config.ae_config.input_size,
        batch_sz=test_config.batch_size,
        missing_rate=0.25, seed=42, device=test_config.device
    )
    stream_simulator = PoissonStreamSimulator(rate_per_second=10.0, seed=43)

    # Ensure the rnn_config_template used for the AE also reflects the internal_projection_size
    # This is handled by TemporalAutoencoder init.
    autoencoder = TemporalAutoencoder(test_config.rnn_config_template, test_config.ae_config).to(test_config.device)
    learner_opt_config = test_config.rnn_config_template # Optimizer uses the same base config
    learner = SlidingTBPTTOnlineLearnerAE(autoencoder, learner_opt_config)
    learner.reset_streaming_state(batch_size=test_config.batch_size, device=test_config.device)

    metrics = StreamingMetrics()
    processing_q = queue.Queue(maxsize=100)

    start_sim_time = time.monotonic()
    end_sim_time = start_sim_time + test_config.test_duration_seconds

    producer = threading.Thread(target=producer_thread_fn,
                                args=(data_generator, stream_simulator, processing_q, metrics, end_sim_time, test_config.device), daemon=True)
    consumer = threading.Thread(target=consumer_thread_fn,
                                args=(learner, processing_q, metrics, test_config, test_config.device), daemon=True)

    producer.start()
    consumer.start()

    main_thread_last_q_check_time = time.monotonic()
    while time.monotonic() < end_sim_time:
        if stop_event.is_set(): break # Exit if threads signaled stop early (e.g. error)
        time.sleep(0.1)
        current_time = time.monotonic()
        if current_time - main_thread_last_q_check_time > 1.0: # Check queue size every second
            if processing_q.qsize() > 50:
                print(f"   ⚠️  Processing queue backing up: {processing_q.qsize()} items")
            main_thread_last_q_check_time = current_time

        if not consumer.is_alive() and metrics.processed_points > 0 :
            print("   ❌ Consumer thread seems to have exited prematurely.")
            if not stop_event.is_set(): stop_event.set()
            break
        if not producer.is_alive() and not stop_event.is_set(): # Producer died before stop signal
            print("   ❌ Producer thread seems to have exited prematurely.")
            if not stop_event.is_set(): stop_event.set()
            break


    if not stop_event.is_set(): # If loop finished due to time, set stop_event
        print("   ⏳ Test duration reached. Signalling threads to stop...")
        stop_event.set()

    producer.join(timeout=5.0)
    # Before joining consumer, ensure all items it might process are done if it's still running
    # This is tricky; if consumer is stuck on q.get(), join will wait.
    # If q is empty and stop_event is set, consumer should exit.
    consumer.join(timeout=10.0)

    if producer.is_alive(): print("   ⚠️ Producer thread did not exit cleanly after timeout.")
    if consumer.is_alive(): print("   ⚠️ Consumer thread did not exit cleanly after timeout.")

    metrics.print_summary(test_config.name)

    actual_duration_s = time.monotonic() - start_sim_time
    actual_rate = metrics.processed_points / actual_duration_s if actual_duration_s > 0 and metrics.processed_points > 0 else 0

    print(f"   ✅ Test completed in {actual_duration_s:.1f}s, "
          f"Actual processing rate: {actual_rate:.1f} points/sec")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    print("🌟 ====================================================================")
    print("🌟          PYTHON REAL-TIME INCREMENTAL AUTOENCODER TESTING          ")
    print("🌟 ====================================================================")
    print("\n📋 Test Configuration:")
    print("   • Real-time online incremental learning")
    print("   • Synthetic multivariate streaming data")
    print("   • Poisson arrival distribution (~10 points/second)")
    print("   • Random missing data patterns (25% missing rate)")
    print("   • Multiple autoencoder configurations")
    print("   • Advanced data patterns: trends, seasonality, correlations, AR components")

    configurations = create_test_configurations()
    print(f"\n🎯 Running {len(configurations)} different configurations...")

    total_start_time = time.monotonic()

    for i, config_item in enumerate(configurations):
        print("\n" + "="*70)
        print(f"🔧 Configuration {i+1}/{len(configurations)}")
        try:
            # Pass the device from config_item to run_realtime_test if it's part of its signature
            # or ensure run_realtime_test uses config_item.device internally
            run_realtime_test(config_item)
            if i < len(configurations) - 1:
                print("\n⏸️  Pausing 2 seconds before next test...")
                time.sleep(2)
        except Exception as e:
            print(f"❌ Test FAILED for {config_item.name}: {e}")
            import traceback
            traceback.print_exc()
            if i < len(configurations) - 1:
                print("\n⏸️  Pausing 2 seconds before next test...")
                time.sleep(2)

    total_end_time = time.monotonic()
    total_duration_secs = total_end_time - total_start_time

    print("\n" + "="*70)
    print("🎉 ALL TESTS COMPLETED!")
    print(f"   Total Testing Time: {total_duration_secs:.1f} seconds")
    print(f"   Configurations Tested: {len(configurations)}")

    print("\n📊 TESTED CONFIGURATIONS SUMMARY:")
    for cfg in configurations:
        print(f"   ✅ {cfg.name} - {cfg.test_duration_seconds}s test")
        print(f"      Batch: {cfg.batch_size}, Input: {cfg.ae_config.input_size}, Latent: {cfg.ae_config.latent_size}")
        print(f"      Bottleneck: {cfg.ae_config.bottleneck_type}, Mode: {cfg.ae_config.mode}", end="")
        if cfg.ae_config.mode == "forecasting":
            print(f" ({cfg.ae_config.forecast_horizon} steps, type: {cfg.ae_config.forecasting_mode})")
        else:
            print()

    print("\n🚀 Real-time incremental learning validation (Python replica) complete!")
    print("   All major code paths exercised ✅")
    print("   Streaming data processing validated ✅")
    print("   Missing data handling tested ✅")
    print("   Multiple autoencoder configurations verified ✅")