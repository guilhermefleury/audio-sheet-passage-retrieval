"""
Sheet-side data augmentation matching the paper's `full_aug` config:

  system_translation : random vertical pixel shift of the staff (±5 by default)
  sheet_scaling      : random zoom in/out by a factor in [0.95, 1.05]

Both transforms are applied with a SINGLE sampled value per passage sequence
(not per snippet). This keeps the GRU input coherent across the sequence; the
original paper samples per-snippet because it cuts snippets on the fly from a
full page, but on pre-cut snippets a per-sequence transform is the closest
behavioural match.

All transforms operate on tensors of shape [Ns, 1, H, W] and return the same
shape so the existing collate function still works.
"""

from __future__ import annotations

import random
from typing import Sequence

import torch
import torch.nn.functional as F


class RandomVerticalShift:
    """Shift all snippets in a passage by the same random ±max_shift pixels."""

    def __init__(self, max_shift: int = 5, pad_value: float = 1.0) -> None:
        self.max_shift = int(max_shift)
        self.pad_value = float(pad_value)

    def __call__(self, sheet_seq: torch.Tensor) -> torch.Tensor:
        if self.max_shift <= 0:
            return sheet_seq
        dy = random.randint(-self.max_shift, self.max_shift)
        if dy == 0:
            return sheet_seq
        out = torch.roll(sheet_seq, shifts=dy, dims=2)
        if dy > 0:
            out[:, :, :dy, :] = self.pad_value
        else:
            out[:, :, dy:, :] = self.pad_value
        return out


class RandomScale:
    """
    Random zoom centered on the snippet.

    Samples a single scale factor for the whole sequence, resamples via
    bilinear interpolation, then center-crops (scale > 1) or pads with
    `pad_value` (scale < 1) to recover the original [H, W].
    """

    def __init__(
        self,
        scale_range: Sequence[float] = (0.95, 1.05),
        pad_value: float = 1.0,
    ) -> None:
        self.scale_low, self.scale_high = float(scale_range[0]), float(scale_range[1])
        self.pad_value = float(pad_value)

    def __call__(self, sheet_seq: torch.Tensor) -> torch.Tensor:
        if self.scale_low == 1.0 and self.scale_high == 1.0:
            return sheet_seq
        scale = random.uniform(self.scale_low, self.scale_high)
        if abs(scale - 1.0) < 1e-6:
            return sheet_seq

        _, _, H, W = sheet_seq.shape
        new_H = max(1, int(round(H * scale)))
        new_W = max(1, int(round(W * scale)))
        scaled = F.interpolate(
            sheet_seq, size=(new_H, new_W),
            mode="bilinear", align_corners=False,
        )

        if scale > 1.0:
            top = (new_H - H) // 2
            left = (new_W - W) // 2
            return scaled[:, :, top:top + H, left:left + W]

        # scale < 1.0 -> pad back to (H, W)
        top = (H - new_H) // 2
        bot = H - new_H - top
        left = (W - new_W) // 2
        right = W - new_W - left
        return F.pad(scaled, (left, right, top, bot), value=self.pad_value)


class Compose:
    """Apply a sequence of transforms in order."""

    def __init__(self, transforms: Sequence) -> None:
        self.transforms = list(transforms)

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


def build_sheet_train_transform(
    translation: int = 5,
    scale_range: Sequence[float] = (0.95, 1.05),
):
    """
    Convenience factory matching the paper's full_aug sheet augmentation.
    Pass translation=0 or scale_range=(1.0, 1.0) to disable either piece.
    """
    ops = []
    if translation > 0:
        ops.append(RandomVerticalShift(max_shift=translation))
    if scale_range[0] != 1.0 or scale_range[1] != 1.0:
        ops.append(RandomScale(scale_range=scale_range))
    return Compose(ops) if ops else None


# ---------------------------------------------------------------------------
# Spectrogram-side transforms
# ---------------------------------------------------------------------------

class RandomSpecShift:
    """
    Random ±max_shift frame shift of an audio spectrogram sequence in time.

    Implements the paper's `onset_translation` augmentation (msmd_config.yaml
    full_aug.onset_translation = 1). Applied per-passage (same shift for all
    snippets in the sequence), with zero-padding on the wrap-around region
    so the model sees silence at the edges instead of cyclic noise.

    Input/output shape: [Na, 1, F, T] with F=92, T=20 in the paper setting.
    """

    def __init__(self, max_shift: int = 1, pad_value: float = 0.0) -> None:
        self.max_shift = int(max_shift)
        self.pad_value = float(pad_value)

    def __call__(self, spec_seq: torch.Tensor) -> torch.Tensor:
        if self.max_shift <= 0:
            return spec_seq
        dx = random.randint(-self.max_shift, self.max_shift)
        if dx == 0:
            return spec_seq
        out = torch.roll(spec_seq, shifts=dx, dims=3)
        if dx > 0:
            out[:, :, :, :dx] = self.pad_value
        else:
            out[:, :, :, dx:] = self.pad_value
        return out


def build_spec_train_transform(onset_translation: int = 1):
    """
    Factory matching the paper's full_aug audio augmentation.
    Pass onset_translation=0 to disable.
    """
    if onset_translation <= 0:
        return None
    return RandomSpecShift(max_shift=onset_translation)
