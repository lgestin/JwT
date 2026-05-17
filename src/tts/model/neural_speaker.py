from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from tts.model.transformer import (
    FeedForward,
    QKNorm,
    RMSNorm,
    SelfAttention,
    Transformer,
    TransformerConfig,
    precompute_freqs_cis,
)


class NeuralSpeaker(Protocol):
    def speak(self, text: "MaskedTensor") -> "MaskedTensor": ...


@dataclass
class RollingFlowConfig:
    transformer_config: TransformerConfig
    vocabulary_size: int
    mel_dim: int
    n_denoising_steps: int
    length_loss_weight: float = 1.0
    length_encoder_num_layers: int = 2
    max_mel_len: int = 2048


@dataclass
class MaskedTensor:
    values: torch.Tensor
    mask: torch.BoolTensor

    def __post_init__(self):
        assert self.values.shape[-1] == self.mask.shape[-1]
        assert self.values.ndim == self.mask.ndim + 1

    @property
    def shape(self):
        return self.values.shape

    @property
    def masked_shape(self):
        return self.mask.sum(-1)


_LJSPEECH_LOG_MEL_MEAN = -5.896610
_LJSPEECH_LOG_MEL_STD = 2.226763


class _TextEncoderBlock(nn.Module):
    """Transformer block without AdaLN — text-only, no timestep conditioning."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = SelfAttention(dim, num_heads)
        self.norm2 = RMSNorm(dim)
        self.ff = FeedForward(dim, mlp_ratio)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), freqs_cis, mask)
        x = x + self.ff(self.norm2(x))
        return x


class _SingleQueryCrossAttention(nn.Module):
    """Cross-attention with a single learned query (no RoPE on either side)."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.kv_proj = nn.Linear(dim, 2 * dim, bias=False)
        self.qk_norm = QKNorm(dim // num_heads)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """query: (B, 1, D), context: (B, L, D), context_mask: (B, L) bool."""
        q = self.q_proj(query)
        kv = self.kv_proj(context)
        q = rearrange(q, "B L (H D) -> B H L D", H=self.num_heads)
        k, v = rearrange(kv, "B L (K H D) -> K B H L D", K=2, H=self.num_heads)
        q, k = self.qk_norm(q, k)
        attn_mask = (
            context_mask[:, None, None, :] if context_mask is not None else None
        )
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = rearrange(x, "B H L D -> B L (H D)")
        return self.proj(x)


class TextLengthEncoder(nn.Module):
    """Predicts log(mel_frames / text_tokens) from text tokens alone.

    Architecture: token embedding → N self-attention blocks (no AdaLN) →
    a learned query cross-attends to the encoder outputs → MLP head → scalar.
    The scalar `pred` is interpreted as `log(L / n_text_tokens)`, so
    `L_hat = n_text_tokens * exp(pred)`.
    """

    def __init__(
        self,
        vocabulary_size: int,
        dim: int,
        num_heads: int,
        num_layers: int,
        mlp_ratio: float = 4.0,
        max_text_len: int = 8192,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocabulary_size, dim)
        self.blocks = nn.ModuleList(
            [_TextEncoderBlock(dim, num_heads, mlp_ratio) for _ in range(num_layers)]
        )
        self.norm = RMSNorm(dim)
        self.query = nn.Parameter(torch.randn(dim) * 0.02)
        self.cross_attn = _SingleQueryCrossAttention(dim, num_heads)
        self.head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, 1),
        )
        freqs_cis = precompute_freqs_cis(max_text_len, dim // num_heads, rope_theta)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, text: "MaskedTensor") -> torch.Tensor:
        """text.values: (B, 1, T_text); text.mask: (B, T_text). Returns (B,)."""
        text_ids = text.values.squeeze(-2)  # (B, T_text)
        B, T_text = text_ids.shape
        x = self.embed(text_ids)
        freqs_cis = self.freqs_cis[:, :, :T_text]
        attn_mask = text.mask[:, None, None, :]  # (B, 1, 1, T_text)
        for block in self.blocks:
            x = block(x, freqs_cis, attn_mask)
        x = self.norm(x)
        q = self.query.view(1, 1, -1).expand(B, 1, -1)
        pooled = self.cross_attn(q, x, context_mask=text.mask)  # (B, 1, D)
        return self.head(pooled).squeeze(-1).squeeze(-1)  # (B,)


class RollingFlowSpeaker(NeuralSpeaker, nn.Module):
    def __init__(self, cfg: RollingFlowConfig):
        nn.Module.__init__(self)
        self.cfg = cfg
        dim = cfg.transformer_config.dim
        self.text_in = nn.Embedding(cfg.vocabulary_size, dim)
        self.mel_in = nn.Linear(cfg.mel_dim, dim)
        self.mel_out = nn.Linear(dim, cfg.mel_dim)
        self.text_modality = nn.Parameter(torch.randn(dim) * 0.02)
        self.mel_modality = nn.Parameter(torch.randn(dim) * 0.02)
        self.transformer = Transformer(cfg.transformer_config)
        self.length_encoder = TextLengthEncoder(
            vocabulary_size=cfg.vocabulary_size,
            dim=dim,
            num_heads=cfg.transformer_config.num_heads,
            num_layers=cfg.length_encoder_num_layers,
            mlp_ratio=cfg.transformer_config.mlp_ratio,
            max_text_len=cfg.transformer_config.max_seq_len,
            rope_theta=cfg.transformer_config.rope_theta,
        )
        # Global log-mel stats (LJSpeech 24 kHz, BigVGAN log-mels). The model
        # operates in normalized space; speak() denormalizes before returning.
        self.register_buffer(
            "mel_mean", torch.tensor(_LJSPEECH_LOG_MEL_MEAN, dtype=torch.float32)
        )
        self.register_buffer(
            "mel_std", torch.tensor(_LJSPEECH_LOG_MEL_STD, dtype=torch.float32)
        )

    def forward(
        self,
        text: MaskedTensor,
        mels: MaskedTensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict velocity at every mel position.

        text.values: (B, 1, T_text)       text.mask: (B, T_text)
        mels.values: (B, mel_dim, T_mel)  mels.mask: (B, T_mel)
        t:           (B, T_mel)           per-mel-position timestep in [0, 1]
        returns:
            v_pred: (B, T_mel, mel_dim)   predicted velocity
        """
        B, mel_dim, T_mel = mels.values.shape
        text_ids = text.values.squeeze(-2)  # (B, T_text)
        T_text = text_ids.shape[-1]
        T = T_text + T_mel
        device = mels.values.device

        text_lens = text.mask.sum(-1)
        mel_lens = mels.mask.sum(-1)
        total_lens = text_lens + mel_lens

        # Project both modalities into the transformer's hidden dim and tag them.
        text_lat = self.text_in(text_ids) + self.text_modality
        mel_lat = self.mel_in(mels.values.transpose(1, 2)) + self.mel_modality
        dim = text_lat.shape[-1]

        # Pack [real text | real mels | trailing pad] per sample.
        arange = torch.arange(T, device=device).expand(B, T)
        in_text = F.pad(text.mask, (0, T_mel))
        in_mels = F.pad(mels.mask, (T_text, 0))
        pack_idx = torch.where(
            in_text,
            arange,
            T_text + (arange - text_lens.unsqueeze(1)),
        ).clamp(min=0, max=T - 1)

        x_concat = torch.cat([text_lat, mel_lat], dim=1)
        x_packed = torch.gather(x_concat, 1, pack_idx.unsqueeze(-1).expand(B, T, dim))

        # Pack per-position t: text positions are always clean (t=1).
        t_concat = torch.cat(
            [torch.ones(B, T_text, device=device, dtype=t.dtype), t], dim=1
        )
        t_packed = torch.gather(t_concat, 1, pack_idx)

        # Keep masks in packed coords. in_text/in_mels above are in original
        # coords and silently misalign the attention mask when text is padded.
        in_real_packed = arange < total_lens.unsqueeze(1)
        in_mels_packed = (arange >= text_lens.unsqueeze(1)) & in_real_packed

        # Attention keys: visible up to and including the first real t=0 (the
        # "next frontier"). Pure-noise positions beyond it carry no signal
        # and would only distract attention.
        is_zero_real = (t_packed == 0.0) & in_mels_packed
        keep_first_zero = is_zero_real.cumsum(-1) <= 1
        attn_keys = in_real_packed & keep_first_zero  # (B, T)
        attn_mask = attn_keys.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
        out_packed = self.transformer(x_packed, t_packed, attn_mask)
        v_pred_packed = self.mel_out(out_packed)  # (B, T, mel_dim)

        # Unpack: mel position i in sample b lives at packed position text_lens[b] + i.
        mel_idx = torch.arange(T_mel, device=device).expand(B, T_mel)
        unpack_idx = (text_lens.unsqueeze(1) + mel_idx).clamp(max=T - 1)
        v_pred = torch.gather(
            v_pred_packed, 1, unpack_idx.unsqueeze(-1).expand(B, T_mel, mel_dim)
        )
        return v_pred

    def training_step(
        self,
        text: MaskedTensor,
        mels: MaskedTensor,
        *,
        mel_front: torch.LongTensor | None = None,
        x_0: torch.Tensor | None = None,
        n: int | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.BoolTensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Sample a rolling front + noise, run forward, and return loss inputs.

        Optional args let callers pin the random choices for reproducibility:
        - mel_front: (B,) long, where each sample's denoising front lands
        - x_0:       (B, T_mel, mel_dim), the noise tensor mixed with x_1
        - n:         override for cfg.n_denoising_steps

        Returns (v_pred, v_target, v_mask, length_pred, length_target, t):
        - v_pred:        (B, T_mel, mel_dim)  predicted velocity
        - v_target:      (B, T_mel, mel_dim)  flow-matching velocity target
        - v_mask:        (B, T_mel)           supervision mask for the ramp
        - length_pred:   (B,)                 predicted log(L / n_text_tokens)
        - length_target: (B,)                 log(L / n_text_tokens) ground truth
        - t:             (B, T_mel)           per-position rolling timestep
        """
        B, mel_dim, T_mel = mels.values.shape
        device = mels.values.device
        n = n if n is not None else self.cfg.n_denoising_steps
        mel_lens = mels.mask.sum(-1)
        text_lens = text.mask.sum(-1)

        if mel_front is None:
            u = torch.rand(B, device=device)
            mel_front = (u * mel_lens.float()).long()
        # Normalize log-mels to roughly N(0, 1) so noise and signal share scale.
        x_1 = (mels.values.transpose(1, 2) - self.mel_mean) / self.mel_std
        if x_0 is None:
            x_0 = torch.randn_like(x_1)

        mel_idx = torch.arange(T_mel, device=device).expand(B, T_mel)
        t = torch.clamp(
            1.0 - (mel_idx - mel_front.unsqueeze(1)).float() / (n - 1),
            0.0,
            1.0,
        )  # (B, T_mel)

        x_t = (1 - t.unsqueeze(-1)) * x_0 + t.unsqueeze(-1) * x_1
        v_target = x_1 - x_0

        noisy_mels = MaskedTensor(values=x_t.transpose(1, 2), mask=mels.mask)
        v_pred = self.forward(text, noisy_mels, t)

        # Supervise the active rolling window: ramp (0 < t < 1) + first t=0.
        # Positions at relative-n from the front (the second t=0) are never
        # touched by the inference Euler loop, so supervising them trains a
        # behavior the model would never use.
        v_mask = (
            mels.mask
            & (mel_idx > mel_front.unsqueeze(1))
            & (mel_idx < mel_front.unsqueeze(1) + n)
        )

        # Global length predictor: a single scalar per sample, predicted from
        # text alone, representing log(frames-per-token). Reconstruction at
        # inference: L_hat = n_text_tokens * exp(pred).
        length_pred = self.length_encoder(text)
        length_target = torch.log(
            mel_lens.float() / text_lens.float().clamp(min=1)
        )

        return v_pred, v_target, v_mask, length_pred, length_target, t

    @torch.no_grad()
    def speak(
        self,
        text: MaskedTensor,
        *,
        x_0: torch.Tensor | None = None,
    ) -> MaskedTensor:
        """Generate mels by rolling-Euler integration up to a predicted length.

        The length encoder predicts per-sample log(L / n_text_tokens) from text
        alone; the recovered L_hat (clamped to [1, cfg.max_mel_len]) is the
        exact number of frames generated for each sample. The batched buffer
        grows up to the longest L_hat to keep attention context consistent.

        text:     MaskedTensor — values (B, 1, T_text), mask (B, T_text)
        x_0:      optional (B, mel_dim, >= cfg.max_mel_len) noise override for
                  reproducibility (sliced one frame per step as the buffer grows)

        Returns a MaskedTensor with values (B, mel_dim, T_out) and mask
        (B, T_out) where T_out = L_hat.max(). Each sample's mask sums to its
        predicted length; positions beyond that are False.
        """
        B = text.values.shape[0]
        device = text.values.device
        n = self.cfg.n_denoising_steps
        dt = 1.0 / (n - 1)
        mel_dim = self.cfg.mel_dim
        max_T = self.cfg.max_mel_len

        # Predict per-sample length from text alone.
        text_lens = text.mask.sum(-1)
        length_pred = self.length_encoder(text)  # (B,)
        L_hat = (text_lens.float() * torch.exp(length_pred)).round().long()
        L_hat = L_hat.clamp(min=1, max=max_T)
        T_out = int(L_hat.max().item())

        if x_0 is None:
            x_0 = torch.randn(B, mel_dim, max_T, device=device)
        else:
            assert x_0.shape[-1] >= max_T, (
                f"x_0 must have at least cfg.max_mel_len ({max_T}) frames along "
                f"the time axis, got {x_0.shape[-1]}"
            )

        mels_values = torch.empty(B, mel_dim, 0, device=device, dtype=x_0.dtype)

        for k in range(T_out + n - 2):
            # Grow the buffer by one new noise frame — the new t=0 frontier.
            if k < T_out:
                mels_values = torch.cat([mels_values, x_0[..., k : k + 1]], dim=-1)

            L = mels_values.shape[-1]
            mel_idx = torch.arange(L, device=device).expand(B, L)
            # Each sample's true (predicted) length caps in_mels so the
            # attention distribution matches training.
            buffer_mask = mel_idx < L_hat.clamp(max=L).unsqueeze(1)

            # t per position = (steps since it was added) / (n - 1), clamped.
            t = torch.clamp((k - mel_idx).float() / (n - 1), 0.0, 1.0)

            mels_mt = MaskedTensor(values=mels_values, mask=buffer_mask)
            v_pred = self.forward(text, mels_mt, t)

            in_window = (t < 1.0) & buffer_mask
            update = (v_pred * dt) * in_window.unsqueeze(-1)
            mels_values = mels_values + update.transpose(1, 2)

        mels_values = mels_values[..., :T_out]
        mel_idx = torch.arange(T_out, device=device).expand(B, T_out)
        mask = mel_idx < L_hat.unsqueeze(1)

        # Denormalize from N(0, 1) space to log-mel scale for downstream consumers.
        mels_values = mels_values * self.mel_std + self.mel_mean
        return MaskedTensor(values=mels_values, mask=mask)
