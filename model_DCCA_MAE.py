# model_ema.py 
import copy
import time
from typing import Dict, Tuple, Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

_EPS = 1e-8

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
class MaskedDecoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = None, 
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        layers = []
        for i in range(num_layers - 1):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.decoder = nn.Sequential(*layers).to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)

class AudioEncoder(nn.Module):
    """Audio encoder with improved architecture"""
    def __init__(self, input_size=128, hidden_size=1024, dropout=0.2):
        super(AudioEncoder, self).__init__()
        self.network = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        ).to(DEVICE)
    
    def forward(self, x):
        return self.network(x)


class VisualEncoder(nn.Module):
    """Visual encoder with improved architecture"""
    def __init__(self, input_size=1024, hidden_size=1024,  dropout=0.2):
        super(VisualEncoder, self).__init__()
        self.network = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        ).to(DEVICE)
    
    def forward(self, x):
        return self.network(x)


class CrossModalFusion(nn.Module): # the same as dcca_vegas.py
    """Optional cross-modal attention fusion layer"""
    def __init__(self, embed_dim: int, num_heads: int = 64, dropout: float = 0.1):
        super().__init__()
        self.audio_to_visual_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        ).to(DEVICE)
#         self.visual_to_audio_attn = nn.MultiheadAttention(
#             embed_dim, num_heads, dropout=dropout, batch_first=True
#         ).to(DEVICE)
        self.norm_a = nn.LayerNorm(embed_dim).to(DEVICE)
        self.norm_v = nn.LayerNorm(embed_dim).to(DEVICE)
        self.ffn_a = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, embed_dim)
        ).to(DEVICE)
        self.ffn_v = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, embed_dim)
        ).to(DEVICE)
        self.dropout = nn.Dropout(dropout).to(DEVICE)

    def forward(self, audio_emb: torch.Tensor, visual_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a = audio_emb.unsqueeze(1)  # (B, 1, E)
        v = visual_emb.unsqueeze(1)

        # Audio attends to Visual (A <- V)
        a2v, _ = self.audio_to_visual_attn(query=v, key=a, value=a)
        a = a+self.dropout(a2v)
        a = self.norm_a(a)
        a = a+self.dropout(self.ffn_a(a))

        v = self.norm_v(v)
        v = self.dropout(self.ffn_v(v))

        return a.squeeze(1), v.squeeze(1)

class Projector(nn.Module):
    """Simple projector - matches your 51% model"""
    def __init__(self, in_dim, out_dim, hidden=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, out_dim),
        ).to(DEVICE)
    
    def forward(self, x):
        return self.net(x)  # No normalization here!

class GradMask(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask):
        ctx.save_for_backward(mask)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        (mask,) = ctx.saved_tensors
        return grad_output * mask, None

class UTAV_MAE(nn.Module):
    def __init__(self,
                 audio_input_dim: int = 128,
                 visual_input_dim: int = 1024,
                 embed_dim: int = 1024,
                 out_dim: int = 128,
                 mask_ratio: float = 0.5,
                 temporal_pool: str = "mean"):
        super().__init__()
        assert 0.0 <= mask_ratio < 1.0
        assert temporal_pool in ("mean", "max", "none")

        self.mask_ratio = mask_ratio
        self.temporal_pool = temporal_pool

        # Encoders (shared weights)
        self.audio_encoder  = AudioEncoder(audio_input_dim, embed_dim)
        self.visual_encoder = VisualEncoder(visual_input_dim, embed_dim)

        # Decoders (reconstruction)
        self.audio_decoder_mask  = MaskedDecoder(embed_dim, audio_input_dim)
        self.visual_decoder_mask = MaskedDecoder(embed_dim, visual_input_dim)
        
        # Cross-modal decoders
        self.cross_a2v = MaskedDecoder(embed_dim, visual_input_dim)
        self.cross_v2a = MaskedDecoder(embed_dim, audio_input_dim)

        # Semantic projection (retrieval stream)
        self.proj_a = nn.Linear(embed_dim, out_dim)
        self.proj_v = nn.Linear(embed_dim, out_dim)

        # Cross-modal fusion
        self.cross_modal_fusion = CrossModalFusion(embed_dim, num_heads=256)

    def mask_for_mae(self, x, ratio):
        """Masking for MAE: zero out masked regions"""
        mask = (torch.rand_like(x) > ratio).float()
        x_masked = x * mask
        return GradMask.apply(x_masked, mask), mask

    def mask_for_cca(self, x, ratio):
        """Masking for CCA: keep original values, mask gradients only"""
        mask = (torch.rand_like(x) > ratio).float()
        return GradMask.apply(x, mask), mask

    def forward(self, audio_x, visual_x, use_cca_path=True, use_mae_path=True):
        """
        Dual-path forward pass for joint CCA+MAE training
        
        Args:
            audio_x: Audio input
            visual_x: Visual input
            use_cca_path: Enable CCA path (unmasked input)
            use_mae_path: Enable MAE path (masked input)
        
        Returns:
            Dictionary with all outputs for flexible loss computation
        """
        outputs = {}
        
        # ============================================
        # PATH 1: CCA Path (Unmasked for correlation)
        # ============================================
        if use_cca_path:
            # Use gradient masking only (keep original values)
            masked_audio_cca, mask_a_cca = self.mask_for_cca(audio_x, self.mask_ratio)
            masked_visual_cca, mask_v_cca = self.mask_for_cca(visual_x, self.mask_ratio)
            
            a_embed_cca = self.audio_encoder(masked_audio_cca)
            v_embed_cca = self.visual_encoder(masked_visual_cca)
            
            # Cross-modal fusion
            z_a_fused, z_v_fused = self.cross_modal_fusion(a_embed_cca, v_embed_cca)
            
            # Normalize for CCA
            z_a_cca = F.normalize(self.proj_a(z_a_fused), dim=-1)
            z_v_cca = F.normalize(self.proj_v(z_v_fused), dim=-1)
            
            outputs['cca'] = {
                'z_a': z_a_cca,
                'z_v': z_v_cca,
            }
        
        # ============================================
        # PATH 2: MAE Path (Masked input for reconstruction)
        # ============================================
        if use_mae_path:
            # Actually mask the input (zero out)
            masked_audio_mae, mask_a_mae = self.mask_for_mae(audio_x, self.mask_ratio)
            masked_visual_mae, mask_v_mae = self.mask_for_mae(visual_x, self.mask_ratio)
            
            a_embed_mae = self.audio_encoder(masked_audio_mae)
            v_embed_mae = self.visual_encoder(masked_visual_mae)
            
            # Reconstruction
            rec_audio = self.audio_decoder_mask(a_embed_mae)
            rec_visual = self.visual_decoder_mask(v_embed_mae)
            
            # Get semantic features for consistency
            z_a_fused_mae, z_v_fused_mae = self.cross_modal_fusion(a_embed_mae, v_embed_mae)
            z_a_mae = F.normalize(self.proj_a(z_a_fused_mae), dim=-1)
            z_v_mae = F.normalize(self.proj_v(z_v_fused_mae), dim=-1)
            
            outputs['mae'] = {
                'rec_audio': rec_audio,
                'rec_visual': rec_visual,
                'rec_a2v': rec_a2v,
                'rec_v2a': rec_v2a,
                'z_a': z_a_mae,
                'z_v': z_v_mae
            }
        
        # Store original inputs
        outputs['inputs'] = {'audio': audio_x, 'visual': visual_x}
        
        return outputs

    