
"""Small U-Net for dense wildfire (burned pixel) forecasting.
 
Input : (B, C_in, H, W) predictor stack at time t (plus static / seasonal channels)
Output: (B, 1, H, W) logits for burned pixels at t+1 (apply sigmoid for probabilities)
 
Works on random crops at train time and on the full grid at test time
(H and W can be arbitrary: the input is padded to a multiple of 2**(levels-1)
and the output is cropped back).
"""
 
from __future__ import annotations
 
from typing import Sequence
 
import torch
import torch.nn as nn
import torch.nn.functional as F
 
 
def _norm(num_channels: int, norm: str) -> nn.Module:
    if norm == "batch":
        return nn.BatchNorm2d(num_channels)
    if norm == "group":
        groups = min(8, num_channels)
        while num_channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, num_channels)
    raise ValueError(f"Unknown norm: {norm!r} (use 'batch' or 'group')")
 
 
class ConvBlock(nn.Module):
    """(Conv3x3 -> Norm -> ReLU) x 2, with optional spatial dropout in between."""
 
    def __init__(self, in_ch: int, out_ch: int, norm: str = "group", dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _norm(out_ch, norm),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _norm(out_ch, norm),
            nn.ReLU(inplace=True),
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
 
 
class Down(nn.Module):
    """MaxPool 2x2 -> ConvBlock."""
 
    def __init__(self, in_ch: int, out_ch: int, norm: str, dropout: float):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch, norm, dropout)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))
 
 
class Up(nn.Module):
    """Bilinear upsample x2 -> concat skip -> ConvBlock."""
 
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, norm: str, dropout: float):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch, norm, dropout)
 
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)
 
 
class UNet(nn.Module):
    """U-Net with configurable depth and width.
 
    Args:
        in_channels: number of input channels (predictors + extra channels).
        out_channels: 1 for binary burned / not burned.
        channels: channel width per level; len(channels) = number of levels.
            Default (32, 64, 128, 256) = 4 levels, 3 downsamplings.
        norm: "group" (robust to small batches / crops) or "batch".
        dropout: spatial dropout probability, applied in the deeper levels
            (the bottleneck and the last encoder / first decoder blocks).
        final_bias_prior: initial foreground probability used to set the bias of the
            output layer. With ~<1% fire pixels this stabilises early training.
    """
 
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        channels: Sequence[int] = (32, 64, 128, 256),
        norm: str = "group",
        dropout: float = 0.1,
        final_bias_prior: float | None = 0.01,
    ):
        super().__init__()
        if len(channels) < 2:
            raise ValueError("Need at least 2 levels.")
        self.channels = tuple(channels)
        self.levels = len(channels)
        self.divisor = 2 ** (self.levels - 1)
 
        # Encoder: no dropout on the highest-resolution level.
        self.inc = ConvBlock(in_channels, channels[0], norm, dropout=0.0)
        self.downs = nn.ModuleList(
            [
                Down(channels[i], channels[i + 1], norm, dropout if i >= self.levels - 3 else 0.0)
                for i in range(self.levels - 1)
            ]
        )
 
        # Decoder.
        self.ups = nn.ModuleList(
            [
                Up(
                    in_ch=channels[i + 1],
                    skip_ch=channels[i],
                    out_ch=channels[i],
                    norm=norm,
                    dropout=dropout if i >= self.levels - 3 else 0.0,
                )
                for i in reversed(range(self.levels - 1))
            ]
        )
 
        self.outc = nn.Conv2d(channels[0], out_channels, kernel_size=1)
        self._init_weights(final_bias_prior)
 
    def _init_weights(self, final_bias_prior: float | None) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Small weights on the 1x1 output layer so the initial prediction is set by the bias.
        nn.init.normal_(self.outc.weight, mean=0.0, std=1e-3)
        if final_bias_prior is not None:
            nn.init.constant_(
                self.outc.bias,
                torch.logit(torch.tensor(final_bias_prior)).item(),
            )
 
    def _pad_to_multiple(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        h, w = x.shape[-2:]
        pad_h = (-h) % self.divisor
        pad_w = (-w) % self.divisor
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        return x, (h, w)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, (h, w) = self._pad_to_multiple(x)
 
        skips = []
        x = self.inc(x)
        skips.append(x)
        for down in self.downs:
            x = down(x)
            skips.append(x)
 
        skips.pop()  # the bottleneck is x itself, not a skip connection
        for up in self.ups:
            x = up(x, skips.pop())
 
        logits = self.outc(x)
        return logits[..., :h, :w]
 
 
def build_unet(cfg: dict) -> UNet:
    """Build from a config dict, e.g. the `model` section of a YAML file."""
    return UNet(
        in_channels=cfg["in_channels"],
        out_channels=cfg.get("out_channels", 1),
        channels=tuple(cfg.get("channels", (32, 64, 128, 256))),
        norm=cfg.get("norm", "group"),
        dropout=cfg.get("dropout", 0.1),
        final_bias_prior=cfg.get("final_bias_prior", 0.01),
    )
 
 
if __name__ == "__main__":
    model = UNet(in_channels=20)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params / 1e6:.2f} M")
    # The full 1800x3600 grid needs a GPU (or tiled inference); use smaller sizes here.
    for shape in [(2, 20, 128, 128), (1, 20, 450, 900), (1, 20, 101, 77)]:
        with torch.no_grad():
            y = model(torch.randn(*shape))
        assert y.shape == (shape[0], 1, shape[2], shape[3]), y.shape
        print(shape, "->", tuple(y.shape))