"""CPL Spectrum Prediction Model.

Input: 5 process parameters (color, gray, thickness, stretch, is_red)
Output: CD[mdeg] spectrum (801 points, 400~800 nm)

Architecture:
  - FeatureEncoder: Embedding for discrete params, RBF+Linear for continuous params
  - CPLCNN: 1D transposed convolution decoder generating 801-point spectrum
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader


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


# ============ Lightning Module ============

class CPLLightningModule(pl.LightningModule):
    """PyTorch Lightning module wrapping FeatureEncoder + CPLCNN."""

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
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 10,
        max_epochs: int = 500,
        spectrum_scale: float = 1.0,
        raw_loss_weight: float = 1e-6,
        spectrum_mean: float = 0.0,
        spectrum_std: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = FeatureEncoder(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            use_rbf=use_rbf,
            rbf_centers=rbf_centers,
            rbf_gamma=rbf_gamma,
            dropout=dropout,
        )
        self.decoder = CPLCNN(
            hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout,
            base_ch=base_ch, init_len=init_len,
            smooth_kernel=smooth_kernel, smooth_sigma=smooth_sigma,
        )

        self.spectrum_scale = float(spectrum_scale) if spectrum_scale > 0 else 1.0
        self.raw_loss_weight = float(raw_loss_weight)

        # Kept for compatibility with older z-score checkpoints; unused by the asinh transform.
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

    def _common_step(self, batch: Tuple, batch_idx: int, stage: str) -> torch.Tensor:
        color_idx, gray_norm, thickness_norm, stretch_norm, is_red_idx, spectrum_transformed = batch
        pred_transformed = self(color_idx, gray_norm, thickness_norm, stretch_norm, is_red_idx)

        transform_loss = nn.functional.mse_loss(pred_transformed, spectrum_transformed)

        # Auxiliary loss in original CD[mdeg] scale keeps absolute peak intensity calibrated.
        pred_raw = torch.sinh(pred_transformed) * self.spectrum_scale
        tgt_raw = torch.sinh(spectrum_transformed) * self.spectrum_scale
        raw_loss = nn.functional.mse_loss(pred_raw, tgt_raw)

        loss = transform_loss + self.raw_loss_weight * raw_loss
        sync_dist = torch.distributed.is_available() and torch.distributed.is_initialized()
        self.log(f"{stage}_loss", loss, on_step=(stage == "train"), on_epoch=True, prog_bar=True, sync_dist=sync_dist)
        self.log(f"{stage}_asinh_mse", transform_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=sync_dist)
        self.log(f"{stage}_raw_mse", raw_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=sync_dist)
        return loss

    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        return self._common_step(batch, batch_idx, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        # Warmup -> CosineAnnealing
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=self.hparams.warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, self.hparams.max_epochs - self.hparams.warmup_epochs)
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[self.hparams.warmup_epochs]
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


# ============ Dataset ============

class CPLDataset(Dataset):
    """Torch Dataset wrapping the pickled CPL data."""

    def __init__(
        self,
        data: Dict[str, dict],
        color_map: Dict[str, int],
        gray_min: float,
        gray_max: float,
        thickness_min: float,
        thickness_max: float,
        stretch_min: float,
        stretch_max: float,
        spectrum_scale: float,
    ):
        self.data = data
        self.keys = list(data.keys())
        self.color_map = color_map

        self.gray_min, self.gray_max = gray_min, gray_max
        self.thickness_min, self.thickness_max = thickness_min, thickness_max
        self.stretch_min, self.stretch_max = stretch_min, stretch_max
        self.spectrum_scale = spectrum_scale

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> Tuple:
        key = self.keys[idx]
        color, gray, thickness, stretch, is_red = key.split("_")

        color_idx = self.color_map[color]
        is_red_idx = 1 if is_red == "True" else 0

        gray_norm = (float(gray) - self.gray_min) / (self.gray_max - self.gray_min)
        thickness_norm = (float(thickness) - self.thickness_min) / (self.thickness_max - self.thickness_min)
        stretch_norm = (float(stretch) - self.stretch_min) / (self.stretch_max - self.stretch_min)

        spectrum = np.array(self.data[key]["cd"], dtype=np.float32)
        spectrum_transformed = np.arcsinh(spectrum / self.spectrum_scale).astype(np.float32)

        return (
            torch.tensor(color_idx, dtype=torch.long),
            torch.tensor([gray_norm], dtype=torch.float32),
            torch.tensor([thickness_norm], dtype=torch.float32),
            torch.tensor([stretch_norm], dtype=torch.float32),
            torch.tensor(is_red_idx, dtype=torch.long),
            torch.tensor(spectrum_transformed, dtype=torch.float32),
        )


# ============ DataModule ============

class CPLDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule with stratified 9:1 split by color x is_red."""

    def __init__(
        self,
        data_path: str,
        batch_size: int = 32,
        split_ratio: float = 0.9,
        seed: int = 42,
        num_workers: int = 4,
        spectrum_scale_quantile: float = 0.75,
        train_keys_path: str | None = None,
        val_keys_path: str | None = None,
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.seed = seed
        self.num_workers = num_workers
        self.spectrum_scale_quantile = spectrum_scale_quantile
        self.train_keys_path = train_keys_path
        self.val_keys_path = val_keys_path

    @staticmethod
    def _read_keys(path: str | Path) -> List[str]:
        with open(path, "r") as f:
            return [line.strip() for line in f if line.strip()]

    def setup(self, stage: str = None):
        with open(self.data_path, "rb") as f:
            raw_data = pickle.load(f)

        # Gather parameter ranges from all data
        color_set = set()
        grays, thicknesses, stretches = [], [], []
        for key in raw_data.keys():
            color, gray, thickness, stretch, _is_red = key.split("_")
            color_set.add(color)
            grays.append(float(gray))
            thicknesses.append(float(thickness))
            stretches.append(float(stretch))

        self.color_map = {c: i for i, c in enumerate(sorted(color_set))}
        self.gray_min, self.gray_max = min(grays), max(grays)
        self.thickness_min, self.thickness_max = min(thicknesses), max(thicknesses)
        self.stretch_min, self.stretch_max = min(stretches), max(stretches)

        if self.train_keys_path and self.val_keys_path:
            train_keys = self._read_keys(self.train_keys_path)
            val_keys = self._read_keys(self.val_keys_path)
            missing = [k for k in train_keys + val_keys if k not in raw_data]
            if missing:
                raise ValueError(f"Split keys not found in data: {missing[:10]}")
            overlap = sorted(set(train_keys) & set(val_keys))
            if overlap:
                raise ValueError(f"Train/val key overlap detected: {overlap[:10]}")
            print(
                f"[CPLDataModule] Loaded external split keys: "
                f"{self.train_keys_path} ({len(train_keys)}), {self.val_keys_path} ({len(val_keys)})"
            )
        elif self.train_keys_path or self.val_keys_path:
            raise ValueError("train_keys_path and val_keys_path must be provided together.")
        else:
            # Stratified split by color + is_red (6 groups)
            rng = np.random.RandomState(self.seed)
            groups: Dict[str, List[str]] = {}
            for key in raw_data.keys():
                color, _gray, _thickness, _stretch, is_red = key.split("_")
                group_key = f"{color}_{is_red}"
                groups.setdefault(group_key, []).append(key)

            train_keys, val_keys = [], []
            for group_key, keys in groups.items():
                keys = list(rng.permutation(keys))
                n_train = int(len(keys) * self.split_ratio)
                train_keys.extend(keys[:n_train])
                val_keys.extend(keys[n_train:])

            # Save split keys to txt files
            data_dir = Path(self.data_path).parent
            with open(data_dir / "train_keys.txt", "w") as f:
                f.write("\n".join(sorted(train_keys)))
            with open(data_dir / "val_keys.txt", "w") as f:
                f.write("\n".join(sorted(val_keys)))
            print(
                f"[CPLDataModule] Saved split keys: {data_dir}/train_keys.txt ({len(train_keys)}), "
                f"{data_dir}/val_keys.txt ({len(val_keys)})"
            )

        # Compute asinh transform scale from train set only.
        train_spectra = [np.array(raw_data[k]["cd"], dtype=np.float32) for k in train_keys]
        all_train_cd = np.stack(train_spectra)
        self.spectrum_scale = float(np.quantile(np.abs(all_train_cd), self.spectrum_scale_quantile))
        if self.spectrum_scale <= 0:
            self.spectrum_scale = 1.0

        train_data = {k: raw_data[k] for k in train_keys}
        val_data = {k: raw_data[k] for k in val_keys}

        common_args = (
            self.color_map,
            self.gray_min, self.gray_max,
            self.thickness_min, self.thickness_max,
            self.stretch_min, self.stretch_max,
            self.spectrum_scale,
        )
        self.train_dataset = CPLDataset(train_data, *common_args)
        self.val_dataset = CPLDataset(val_data, *common_args)

        print(f"[CPLDataModule] Train: {len(train_keys)}, Val: {len(val_keys)}")
        print(f"[CPLDataModule] Color map: {self.color_map}")
        print(
            f"[CPLDataModule] Asinh spectrum scale={self.spectrum_scale:.4f} "
            f"(abs CD quantile={self.spectrum_scale_quantile:.2f})"
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
