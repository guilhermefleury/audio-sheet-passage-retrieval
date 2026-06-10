"""
Cross-modal recurrent encoder for audio-sheet music passage retrieval.

Implements Figure 2 / Table 1 of:
  Carvalho & Widmer, "Passage Summarization with Recurrent Models for
  Audio-Sheet Music Retrieval", ISMIR 2023.

Architecture per modality:
  CNN (VGG-style): snippet → 32-D snippet embedding
  GRU (128 hidden): sequence of snippet embeddings → context vector
  FC:               context vector → final passage embedding (emb_dim)

Input contract (from passage_sequence_collate_fn):
  sheet_seq : [B, max_Ns, 1, 160, 180]
  sheet_len : [B]  (LongTensor)
  spec_seq  : [B, max_Na, 1,  92,  20]
  spec_len  : [B]  (LongTensor)
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence


# ---------------------------------------------------------------------------
# Weight initialisation (orthogonal for conv/linear, as in original repo)
# ---------------------------------------------------------------------------

def _init_weights(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# Per-frequency-bin spectrogram normalisation
# ---------------------------------------------------------------------------

class TemporalBatchNorm(nn.Module):
    """
    Per-frequency-bin BatchNorm for spectrogram inputs.

    Input layout (after SequenceEncoder's permute): [N, 1, T, F]
        N = number of snippets in the batch
        T = time frames per snippet (20 for the paper's audio)
        F = frequency bins (92 for the paper's audio)

    Normalises each frequency bin independently across batch + time, matching
    the paper's audio path (`normalize_input=True` in msmd_config.yaml +
    CNNEncoder of lcasr-main/lcasr/models/vgg_model.py).
    """

    def __init__(self, num_bands: int, affine: bool = False) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(num_bands, affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape                        # [N, C, T, F]
        x = x.reshape((-1,) + x.shape[-2:])    # [N*C, T, F]
        x = x.permute(0, 2, 1)                 # [N*C, F, T]  - F as BN channel
        x = self.bn(x)
        x = x.permute(0, 2, 1)                 # [N*C, T, F]
        return x.reshape(shape)


# ---------------------------------------------------------------------------
# CNN snippet encoder
# ---------------------------------------------------------------------------

class CNNEncoder(nn.Module):
    """
    VGG-style CNN that maps a single snippet to a snippet_emb_dim-D vector.

    Sheet snippet input : [N, 1, 160, 180]  (after H/W permute: [N, 1, 180, 160])
    Audio snippet input : [N, 1,  92,  20]  (after H/W permute: [N, 1,  20,  92])

    Architecture (Table 1):
      4x block of (Conv3x3-BN-ELU, Conv3x3-BN-ELU, MaxPool2)
      followed by Conv1x1-BN  (96 → snippet_emb_dim channels)
      followed by FC           (flattened → snippet_emb_dim)
    """

    def __init__(
        self,
        linear_input_size: int,
        snippet_emb_dim: int = 32,
        num_filters: int = 24,
        groupnorm: bool = False,
        normalize_input: bool = False,
        num_freq_bins: int = 92,
    ) -> None:
        super().__init__()
        nf = num_filters

        def norm(channels: int) -> nn.Module:
            return nn.GroupNorm(1, channels) if groupnorm else nn.BatchNorm2d(channels)

        layers: List[nn.Module] = []
        if normalize_input:
            layers.append(TemporalBatchNorm(num_freq_bins, affine=False))
        layers.extend([
            # Block 1:  1 → 24
            nn.Conv2d(1,      nf,      3, padding=1), norm(nf),      nn.ELU(inplace=True),
            nn.Conv2d(nf,     nf,      3, padding=1), norm(nf),      nn.ELU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2: 24 → 48
            nn.Conv2d(nf,     nf * 2,  3, padding=1), norm(nf * 2), nn.ELU(inplace=True),
            nn.Conv2d(nf * 2, nf * 2,  3, padding=1), norm(nf * 2), nn.ELU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3: 48 → 96
            nn.Conv2d(nf * 2, nf * 4,  3, padding=1), norm(nf * 4), nn.ELU(inplace=True),
            nn.Conv2d(nf * 4, nf * 4,  3, padding=1), norm(nf * 4), nn.ELU(inplace=True),
            nn.MaxPool2d(2),
            # Block 4: 96 → 96
            nn.Conv2d(nf * 4, nf * 4,  3, padding=1), norm(nf * 4), nn.ELU(inplace=True),
            nn.Conv2d(nf * 4, nf * 4,  3, padding=1), norm(nf * 4), nn.ELU(inplace=True),
            nn.MaxPool2d(2),
            # 1x1 projection: 96 → snippet_emb_dim
            nn.Conv2d(nf * 4, snippet_emb_dim, 1), norm(snippet_emb_dim),
        ])
        self.cnn = nn.Sequential(*layers)
        self.fc = nn.Linear(linear_input_size, snippet_emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N, 1, H, W]  ->  [N, snippet_emb_dim]"""
        x = self.cnn(x)       # [N, snippet_emb_dim, h, w]
        x = x.flatten(1)      # [N, snippet_emb_dim * h * w]
        return self.fc(x)     # [N, snippet_emb_dim]


# ---------------------------------------------------------------------------
# Per-modality sequence encoder (CNN + GRU + FC)
# ---------------------------------------------------------------------------

class SequenceEncoder(nn.Module):
    """
    Encodes a padded batch of snippet sequences into per-passage embeddings.

    Forward:
        seq     : [B, max_T, 1, H, W]  — padded snippet sequences
        lengths : [B]                   — true sequence lengths
    Returns:
        emb : [B, emb_dim]              — (not yet L2-normalised)
    """

    def __init__(
        self,
        cnn_linear_input_size: int,
        snippet_emb_dim: int = 32,
        rnn_hidden: int = 128,
        emb_dim: int = 64,
        groupnorm: bool = False,
        normalize_input: bool = False,
        num_freq_bins: int = 92,
    ) -> None:
        super().__init__()
        self.cnn = CNNEncoder(
            cnn_linear_input_size,
            snippet_emb_dim,
            groupnorm=groupnorm,
            normalize_input=normalize_input,
            num_freq_bins=num_freq_bins,
        )
        self.gru = nn.GRU(
            input_size=snippet_emb_dim,
            hidden_size=rnn_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.fc = nn.Linear(rnn_hidden, emb_dim)

    def forward(self, seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        B = seq.shape[0]

        # ---- CNN: process only real snippets, skip padding entirely ----
        # Unpack padded tensor → list of real-length tensors, then cat.
        # This avoids running the CNN over zero-padded positions, which
        # would waste ~50 % of compute and memory when sequences vary in length.
        real_parts = [seq[i, : lengths[i].item()] for i in range(B)]  # list of [Ns_i, 1, H, W]
        cat_snips = torch.cat(real_parts, dim=0)            # [sum(lengths), 1, H, W]
        cat_snips = cat_snips.permute(0, 1, 3, 2).contiguous()  # swap H,W (original convention)
        cnn_out = self.cnn(cat_snips)                       # [sum(lengths), snippet_emb_dim]

        # Split back into per-sequence tensors and pack for GRU
        split_embs = list(torch.split(cnn_out, lengths.tolist(), dim=0))
        packed = pack_sequence(split_embs, enforce_sorted=False)
        _, h_n = self.gru(packed)                           # h_n: [1, B, rnn_hidden]

        # Last hidden state → passage embedding
        emb = self.fc(h_n.squeeze(0))                       # [B, emb_dim]
        return emb


# ---------------------------------------------------------------------------
# Full two-pathway cross-modal encoder
# ---------------------------------------------------------------------------

class CrossModalEncoder(nn.Module):
    """
    Two-pathway cross-modal encoder (sheet + audio), Figure 2 of the paper.

    Input  (from passage_sequence_collate_fn):
        sheet_seq : [B, max_Ns, 1, 160, 180]
        sheet_len : [B]
        spec_seq  : [B, max_Na, 1,  92,  20]
        spec_len  : [B]

    Output:
        sheet_emb : [B, emb_dim]  — L2-normalised
        audio_emb : [B, emb_dim]  — L2-normalised
    """

    # Spatial size after 4x MaxPool(2) on the permuted (W, H) input:
    #   Sheet [1, 160, 180] -> permute -> [1, 180, 160]:
    #       180//2//2//2//2 = 11,  160//2//2//2//2 = 10  ->  32*11*10 = 3520
    #   Audio [1,  92,  20] -> permute -> [1,  20,  92]:
    #        20//2//2//2//2 =  1,   92//2//2//2//2 =  5  ->  32* 1* 5 =  160
    _SHEET_LINEAR: int = 3520
    _AUDIO_LINEAR: int = 160

    def __init__(
        self,
        snippet_emb_dim: int = 32,
        rnn_hidden: int = 128,
        emb_dim: int = 64,
        audio_normalize_input: bool = True,
    ) -> None:
        super().__init__()
        self.sheet_enc = SequenceEncoder(
            self._SHEET_LINEAR, snippet_emb_dim, rnn_hidden, emb_dim,
            groupnorm=False,
        )
        # Audio path: GroupNorm in CNN body + TemporalBatchNorm on raw input,
        # matching the paper's audio_path (vgg_model.py SequenceEncoder with
        # normalize_input=True for is_audio=True).
        self.audio_enc = SequenceEncoder(
            self._AUDIO_LINEAR, snippet_emb_dim, rnn_hidden, emb_dim,
            groupnorm=True,
            normalize_input=audio_normalize_input,
            num_freq_bins=92,
        )
        self.apply(_init_weights)

    def forward(
        self,
        sheet_seq: torch.Tensor,
        sheet_len: torch.Tensor,
        spec_seq: torch.Tensor,
        spec_len: torch.Tensor,
    ):
        sheet_emb = self.sheet_enc(sheet_seq, sheet_len)   # [B, emb_dim]
        audio_emb = self.audio_enc(spec_seq, spec_len)     # [B, emb_dim]

        # L2-normalise so dot product == cosine similarity
        sheet_emb = F.normalize(sheet_emb, p=2, dim=1)
        audio_emb = F.normalize(audio_emb, p=2, dim=1)

        return sheet_emb, audio_emb
