"""CPL Spectrum Prediction Model (Inference-Only).

Input: 5 process parameters (color, gray, thickness, stretch, is_red)
Output: CD[mdeg] spectrum (801 points, 400~800 nm)

This is a lightweight inference-only variant with no PyTorch Lightning dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn


# ============ RBF Encoding ============

class GaussianRBF(nn.Module):
    """Radial Basis Function encoding for continuous scalar features.

    Maps a normalized scalar x in [0, 1] to num_centers RBF bases,
    then projects to embed_dim via a learnable linear layer.
    """

    def __init__(self, num_centers: int = 16, gamma: float = 10.0):
        super().__init__()
        self.num_centers = num_centers
        self.gamma = gamma
        centers = torch.linspace(0.0, 1.0, num_centers).unsqueeze(0)  # [1, C]
        self.register_buffer("centers", centers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x [B, 1]. Returns: [B, num_centers]."""
        dist = x - self.centers  # [B, C]
        return torch.exp(-self.gamma * dist ** 2)


# ============ Feature Encoder ============

class FeatureEncoder(nn.Module):
    """Encode 5 input parameters into a latent vector.

    Discrete:
      color   -> Embedding(3, embed_dim)
      is_red  -> Embedding(2, embed_dim)
    Continuous (each scalar normalized to [0, 1]):
      gray      -> RBF + Linear -> embed_dim
      thickness -> RBF + Linear -> embed_dim
      stretch   -> RBF + Linear -> embed_dim
    Fusion:
      Concat 5 x embed_dim -> MLP -> hidden_dim
    """

    def __init__(
        self,
        embed_dim: int = 32,
        hidden_dim: int = 256,
        use_rbf: bool = True,
        rbf_centers: int = 16,
        rbf_gamma: float = 10.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_rbf = use_rbf

        # Discrete
        self.color_embed = nn.Embedding(3, embed_dim)
        self.is_red_embed = nn.Embedding(2, embed_dim)

        # Continuous
        if use_rbf:
            self.gray_rbf = GaussianRBF(rbf_centers, rbf_gamma)
            self.gray_proj = nn.Linear(rbf_centers, embed_dim)

            self.thickness_rbf = GaussianRBF(rbf_centers, rbf_gamma)
            self.thickness_proj = nn.Linear(rbf_centers, embed_dim)

            self.stretch_rbf = GaussianRBF(rbf_centers, rbf_gamma)
            self.stretch_proj = nn.Linear(rbf_centers, embed_dim)
        else:
            self.gray_proj = nn.Linear(1, embed_dim)
            self.thickness_proj = nn.Linear(1, embed_dim)
            self.stretch_proj = nn.Linear(1, embed_dim)

        # Fusion MLP
        total_dim = 5 * embed_dim
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        color_idx: torch.Tensor,
        gray_norm: torch.Tensor,
        thickness_norm: torch.Tensor,
        stretch_norm: torch.Tensor,
        is_red_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Returns latent vector [B, hidden_dim]."""
        c_emb = self.color_embed(color_idx)          # [B, E]
        r_emb = self.is_red_embed(is_red_idx)        # [B, E]

        if self.use_rbf:
            g_emb = self.gray_proj(self.gray_rbf(gray_norm))               # [B, E]
            t_emb = self.thickness_proj(self.thickness_rbf(thickness_norm)) # [B, E]
            s_emb = self.stretch_proj(self.stretch_rbf(stretch_norm))       # [B, E]
        else:
            g_emb = self.gray_proj(gray_norm)
            t_emb = self.thickness_proj(thickness_norm)
            s_emb = self.stretch_proj(stretch_norm)

        concat = torch.cat([c_emb, g_emb, t_emb, s_emb, r_emb], dim=-1)  # [B, 5E]
        return self.fusion(concat)  # [B, H]


# ============ Gaussian Smoothing Layer ============

class GaussianSmooth1D(nn.Module):
    """Fixed-weight Gaussian smoothing for 1D sequences.

    Args:
        kernel_size: must be odd
        sigma: standard deviation of Gaussian kernel
    """

    def __init__(self, kernel_size: int = 5, sigma: float = 1.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        x = torch.arange(kernel_size).float() - kernel_size // 2
        kernel = torch.exp(-x**2 / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        self.register_buffer("kernel", kernel.view(1, 1, -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, L] -> [B, C, L]"""
        B, C, L = x.shape
        x_flat = x.view(B * C, 1, L)
        x_smooth = nn.functional.conv1d(x_flat, self.kernel, padding=self.padding)
        return x_smooth.view(B, C, L)


# ============ CNN Decoder ============

class DecoderBlock(nn.Module):
    """Modern decoder block with residual connection.

    ConvTranspose1d -> GroupNorm -> (+ residual) -> SiLU -> Dropout
    """

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        # Modern normalization: GroupNorm (more stable than BatchNorm for small batches / sequences)
        self.norm = nn.GroupNorm(num_groups=min(32, out_ch), num_channels=out_ch)
        # Modern activation: SiLU (Swish)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Residual path: 1x1 projection + spatial upsampling via interpolate
        if in_ch != out_ch:
            self.skip_proj = nn.Conv1d(in_ch, out_ch, kernel_size=1)
        else:
            self.skip_proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, in_ch, L] -> [B, out_ch, 2*L]"""
        residual = self.skip_proj(x)
        out = self.conv(x)
        # Upsample residual to match output spatial dims
        if residual.shape[-1] != out.shape[-1]:
            residual = nn.functional.interpolate(
                residual, size=out.shape[-1], mode="linear", align_corners=False
            )
        out = self.norm(out)
        out = out + residual
        out = self.act(out)
        return self.dropout(out)


class CPLCNN(nn.Module):
    """Modern 1D transposed-convolution decoder with residual blocks.

    latent [B, H]
      -> Linear -> SiLU -> [B, base_ch * 8] -> reshape [B, base_ch, 8]
      -> n_layers x DecoderBlock -> [B, 4, 2^n_layers * 8]
      -> interpolate -> 801 -> [B, 4, 801]
      -> Conv1d(4->1) -> [B, 1, 801]
      -> squeeze -> [B, 801]
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        n_layers: int = 5,
        dropout: float = 0.1,
        base_ch: int = 128,
        init_len: int = 8,
        smooth_kernel: int = 5,
        smooth_sigma: float = 1.0,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.base_ch = base_ch
        self.init_len = init_len

        # Compute channel schedule: halve each layer, floor at 4
        self.channels = [self.base_ch]
        for i in range(1, n_layers):
            self.channels.append(max(self.base_ch // (2 ** i), 4))
        self.channels.append(4)  # final layer always outputs 4 channels

        self.in_len = init_len * (2 ** n_layers)
        self.upsample = nn.Linear(self.in_len, 801)

        self.init_proj = nn.Sequential(
            nn.Linear(hidden_dim, self.channels[0] * self.init_len),
            nn.SiLU(),
        )

        # Stack of decoder blocks with residual connections
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            self.blocks.append(
                DecoderBlock(self.channels[i], self.channels[i + 1], dropout=dropout)
            )

        self.final_conv = nn.Conv1d(4, 1, kernel_size=3, padding=1)
        # Fixed Gaussian smoothing at the output to suppress high-frequency noise
        self.smooth = GaussianSmooth1D(kernel_size=smooth_kernel, sigma=smooth_sigma)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        x = self.init_proj(latent)                              # [B, channels[0]*init_len]
        x = x.view(-1, self.channels[0], self.init_len)       # [B, channels[0], init_len]
        for block in self.blocks:
            x = block(x)                      # [B, channels[i+1], 2*len]
        x = self.upsample(x)                  # [B, 4, 801]
        x = self.final_conv(x)                # [B, 1, 801]
        x = self.smooth(x)                    # [B, 1, 801]  Gaussian smoothing
        return x.squeeze(1)                   # [B, 801]


# ============ Lightweight Inference Wrapper ============

class CPLInferenceModel(nn.Module):
    """Pure-PyTorch inference wrapper for CPL model.

    Loads weights directly from a PyTorch Lightning checkpoint without
    requiring pytorch_lightning to be installed.
    """

    def __init__(
        self,
        embed_dim: int = 32,
        hidden_dim: int = 256,
        use_rbf: bool = True,
        rbf_centers: int = 16,
        rbf_gamma: float = 10.0,
        dropout: float = 0.1,
        n_layers: int = 5,
        base_ch: int = 128,
        init_len: int = 8,
        smooth_kernel: int = 5,
        smooth_sigma: float = 1.0,
        spectrum_scale: float = 1.0,
        spectrum_mean: float = 0.0,
        spectrum_std: float = 1.0,
    ):
        super().__init__()
        self.encoder = FeatureEncoder(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            use_rbf=use_rbf,
            rbf_centers=rbf_centers,
            rbf_gamma=rbf_gamma,
            dropout=dropout,
        )
        self.decoder = CPLCNN(
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            base_ch=base_ch,
            init_len=init_len,
            smooth_kernel=smooth_kernel,
            smooth_sigma=smooth_sigma,
        )
        self.spectrum_scale = float(spectrum_scale) if spectrum_scale > 0 else 1.0
        self.spectrum_transform = "asinh"

        # Kept only so Lightning checkpoints with legacy compatibility buffers load cleanly.
        self.register_buffer("spectrum_mean", torch.tensor(spectrum_mean, dtype=torch.float32))
        self.register_buffer("spectrum_std", torch.tensor(spectrum_std, dtype=torch.float32))

    def forward(
        self,
        color_idx: torch.Tensor,
        gray_norm: torch.Tensor,
        thickness_norm: torch.Tensor,
        stretch_norm: torch.Tensor,
        is_red_idx: torch.Tensor,
    ) -> torch.Tensor:
        latent = self.encoder(color_idx, gray_norm, thickness_norm, stretch_norm, is_red_idx)
        return self.decoder(latent)

    def inverse_transform(self, spectrum_transformed: torch.Tensor) -> torch.Tensor:
        """Convert model output from asinh target space back to CD[mdeg]."""
        return torch.sinh(spectrum_transformed) * self.spectrum_scale

    def predict_spectrum(
        self,
        color_idx: torch.Tensor,
        gray_norm: torch.Tensor,
        thickness_norm: torch.Tensor,
        stretch_norm: torch.Tensor,
        is_red_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Predict CD[mdeg] spectra from normalized process parameters."""
        pred_transformed = self(color_idx, gray_norm, thickness_norm, stretch_norm, is_red_idx)
        return self.inverse_transform(pred_transformed)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, device: str = "auto") -> "CPLInferenceModel":
        """Load a CPLInferenceModel from a PyTorch Lightning checkpoint file.

        Args:
            checkpoint_path: Path to the .ckpt file.
            device: Target device ("auto", "cuda", or "cpu").

        Returns:
            CPLInferenceModel initialized and loaded with checkpoint weights.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        hparams = ckpt.get("hyper_parameters", {})
        if "spectrum_scale" not in hparams:
            raise ValueError(
                "Checkpoint does not contain spectrum_scale. "
                "This inverse-design package now expects the asinh-target model; "
                "use a checkpoint trained with spectrum_transform.type=asinh."
            )

        # Filter only the keys accepted by __init__
        init_keys = {
            "embed_dim", "hidden_dim", "use_rbf", "rbf_centers", "rbf_gamma",
            "dropout", "n_layers", "base_ch", "init_len", "smooth_kernel",
            "smooth_sigma", "spectrum_scale", "spectrum_mean", "spectrum_std",
        }
        model_kwargs = {k: v for k, v in hparams.items() if k in init_keys}

        model = cls(**model_kwargs)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        model.eval()
        model.to(device)
        return model
