import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import time
import math
from typing import Tuple, Dict, List
import json
from pathlib import Path

from config import BaseConfig, get_config
from data_loader import load_data
from model_DCCA_MAE import UTAV_MAE
from metrics import compute_map_metricty66
from linear_cca import linear_cca

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def get_topk_weights(sim: torch.Tensor, k: int = 5, temperature: float = 0.07, min_score: float = 0.0):
    """Get soft weights for top-k similar pairs"""
    B = sim.size(0)
    device = sim.device
    
    sim_masked = sim.clone()
    sim_masked.fill_diagonal_(float('-inf'))  # Exclude diagonal
    sim_masked[sim_masked < min_score] = float('-inf')

    k = min(k, B - 1)  # Ensure k doesn't exceed available pairs
    topk_vals, topk_idx = torch.topk(sim_masked, k=k, dim=1, largest=True, sorted=False)
    
    weights = torch.zeros_like(sim, device=device)
    topk_vals_safe = torch.where(torch.isfinite(topk_vals), topk_vals, 
                                 torch.full_like(topk_vals, -1e9))
    topk_w = F.softmax(topk_vals_safe / (temperature + 1e-12), dim=1)

    rows = torch.arange(B, device=device).unsqueeze(1).repeat(1, k)
    weights[rows, topk_idx] = topk_w
    weights = weights * (~torch.eye(B, device=device).bool()).float()
    
    return weights


def soft_multi_positive_infoNCE(z_q, z_k, pos_weights, temperature=0.07):
    """Multi-positive InfoNCE with soft weighting"""
    z_q = F.normalize(z_q, dim=-1)
    z_k = F.normalize(z_k, dim=-1)

    logits = torch.matmul(z_q, z_k.t()) / temperature
    exp_logits = torch.exp(logits)

    pos_scores = (exp_logits * pos_weights).sum(dim=1).clamp(min=1e-8)
    neg_mask = (1 - pos_weights) * (1 - torch.eye(z_q.size(0), device=z_q.device))
    neg_scores = (exp_logits * neg_mask).sum(dim=1).clamp(min=1e-8)

    loss = -torch.log(pos_scores / (pos_scores + neg_scores + 1e-8))
    return loss.mean()


class DCCALoss(nn.Module):
    def __init__(self, out_dim, reg=1e-4):
        super().__init__()
        self.out_dim = out_dim
        self.reg = reg

    def forward(self, h1, h2):
        h1 = h1 - h1.mean(dim=0)
        h2 = h2 - h2.mean(dim=0)
        m = h1.size(0)

        S11 = (h1.T @ h1) / (m - 1) + self.reg * torch.eye(h1.size(1), device=h1.device)
        S22 = (h2.T @ h2) / (m - 1) + self.reg * torch.eye(h2.size(1), device=h2.device)
        S12 = (h1.T @ h2) / (m - 1)

        eigvals = torch.linalg.eigvals(
            torch.linalg.inv(S11) @ S12 @ torch.linalg.inv(S22) @ S12.T
        )
        corr = eigvals.real.topk(self.out_dim).values.sum()
        return -corr / self.out_dim


class MultiTaskLoss(nn.Module):
    """Learnable multi-task loss weights using uncertainty"""
    def __init__(self, num_tasks=6):  # Updated to 6 tasks
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, losses):
        total = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total += precision * loss + self.log_vars[i]
        return total


class TrainingLogger:
    """Comprehensive logger for training metrics"""
    def __init__(self, save_dir='training_logs'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.train_losses = {
            'total_loss': [], 'cca_loss': [], 'mae_loss': [],
            'loss_rec': [], 'loss_cross': [], 'loss_consistency': [],
            'loss_contrastive': [], 'loss_alignment': [], 'loss_matching': []
        }
        
        self.test_metrics = {
            'audio2visual': [], 'visual2audio': [], 'average_map': []
        }
        
        self.batch_losses = []
        self.learned_weights = []
        self.metadata = {'start_time': time.time(), 'config': {}}
    
    def log_batch(self, epoch: int, batch: int, loss_dict: Dict):
        self.batch_losses.append({'epoch': epoch, 'batch': batch, **loss_dict})
    
    def log_epoch_train(self, epoch: int, total_loss: float, cca_loss: float, 
                       mae_loss: float, loss_dict: Dict):
        self.train_losses['total_loss'].append(total_loss)
        self.train_losses['cca_loss'].append(cca_loss)
        self.train_losses['mae_loss'].append(mae_loss)
        self.train_losses['loss_rec'].append(loss_dict['rec'])
        self.train_losses['loss_cross'].append(loss_dict['cross'])
        self.train_losses['loss_consistency'].append(loss_dict['consist'])
        self.train_losses['loss_contrastive'].append(loss_dict['contrast'])
        self.train_losses['loss_alignment'].append(loss_dict['align'])
        self.train_losses['loss_matching'].append(loss_dict['match'])
        
        if 'weights' in loss_dict:
            self.learned_weights.append({
                'epoch': epoch,
                'weights': loss_dict['weights'].tolist()
            })
    
    def log_epoch_test(self, epoch: int, a2v: float, v2a: float, avg_map: float):
        self.test_metrics['audio2visual'].append(a2v)
        self.test_metrics['visual2audio'].append(v2a)
        self.test_metrics['average_map'].append(avg_map)
    
    def save_logs(self, prefix: str = 'training'):
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        np.savez(self.save_dir / f'{prefix}_losses_{timestamp}.npz', **self.train_losses)
        np.savez(self.save_dir / f'{prefix}_test_metrics_{timestamp}.npz', **self.test_metrics)
        
        if self.batch_losses:
            with open(self.save_dir / f'{prefix}_batch_losses_{timestamp}.json', 'w') as f:
                json.dump(self.batch_losses, f, indent=2)
        
        if self.learned_weights:
            with open(self.save_dir / f'{prefix}_learned_weights_{timestamp}.json', 'w') as f:
                json.dump(self.learned_weights, f, indent=2)
        
        self.metadata['end_time'] = time.time()
        self.metadata['total_time'] = self.metadata['end_time'] - self.metadata['start_time']
        with open(self.save_dir / f'{prefix}_metadata_{timestamp}.json', 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"\n💾 Logs saved to: {self.save_dir}")
        return timestamp
    
    def set_config(self, config_dict: Dict):
        self.metadata['config'] = config_dict


def applied_linear_cca(audio, visual, output_dim, mode_type="train", input_info=None):
    if mode_type == "train":
        w = [None, None]
        m = [None, None]
        w[0], w[1], m[0], m[1] = linear_cca(audio, visual, output_dim)
        return w, m
    elif mode_type == "test":
        if input_info is None:
            raise ValueError("input_info required for test mode")
        w, m = input_info
        audio -= m[0].reshape([1, -1]).repeat(audio.shape[0], axis=0)
        audio = np.dot(audio, w[0])
        visual -= m[1].reshape([1, -1]).repeat(visual.shape[0], axis=0)
        visual = np.dot(visual, w[1])
        return audio, visual

def compute_combined_loss(model, teacher_model, audio_x, visual_x, 
                         cca_loss_fn, multi_task_loss, epoch, 
                         warmup_epochs=5, total_epochs=100):
    """
    Compute loss with dual-path strategy
    
    Strategy:
    - Warmup: MAE only (learn good representations)
    - Main training: CCA + MAE with separate forward passes
    """
    
    # ============================================
    # PHASE 1: Warmup (MAE only, masked input)
    # ============================================
    if epoch < warmup_epochs:
        outputs = model(audio_x, visual_x, use_cca_path=False, use_mae_path=True)
        mae_outputs = outputs['mae']
        
        # Teacher forward (unmasked)
        with torch.no_grad():
            original_mask = teacher_model.mask_ratio
            teacher_model.mask_ratio = 0.0
            teacher_outputs = teacher_model(audio_x, visual_x, use_cca_path=True, use_mae_path=False)
            teacher_model.mask_ratio = original_mask
        
        # MAE losses
        loss_rec = (F.mse_loss(mae_outputs['rec_audio'], audio_x) + 
                   F.mse_loss(mae_outputs['rec_visual'], visual_x))
        
        loss_cross = (F.mse_loss(mae_outputs['rec_a2v'], visual_x) + 
                     F.mse_loss(mae_outputs['rec_v2a'], audio_x))
        
        # Teacher-student consistency
        loss_consistency = (F.mse_loss(mae_outputs['z_a'], teacher_outputs['cca']['z_a']) +
                           F.mse_loss(mae_outputs['z_v'], teacher_outputs['cca']['z_v']))
        
        # Contrastive loss
        z_fused_t = 0.5 * (teacher_outputs['cca']['z_a'].detach() + 
                          teacher_outputs['cca']['z_v'].detach())
        sim_t = torch.matmul(F.normalize(z_fused_t), F.normalize(z_fused_t).t())
        pos_weights = get_topk_weights(sim_t, k=5, temperature=0.05)
        
        loss_contrastive = (
            soft_multi_positive_infoNCE(mae_outputs['z_a'], mae_outputs['z_v'], 
                                       pos_weights, temperature=0.05) +
            soft_multi_positive_infoNCE(mae_outputs['z_v'], mae_outputs['z_a'], 
                                       pos_weights, temperature=0.05)
        ) / 2
        
        # Fixed weights during warmup
        cross_weight = min(1.0, epoch / max(1, warmup_epochs - 1))
        total_loss = (1.0 * loss_rec + 
                     cross_weight * loss_cross + 
                     0.1 * loss_consistency + 
                     0.05 * loss_contrastive)
        
        loss_dict = {
            'total': total_loss.item(),
            'cca': 0.0,
            'rec': loss_rec.item(),
            'cross': loss_cross.item(),
            'consist': loss_consistency.item(),
            'contrast': loss_contrastive.item(),
            'mode': 'warmup'
        }
        
        return total_loss, loss_dict
    
    # ============================================
    # PHASE 2: Joint CCA + MAE Training
    # ============================================
    else:
        # Forward pass with BOTH paths
        outputs = model(audio_x, visual_x, use_cca_path=True, use_mae_path=True)
        cca_outputs = outputs['cca']
        mae_outputs = outputs['mae']
        
        # Teacher forward (unmasked)
        with torch.no_grad():
            original_mask = teacher_model.mask_ratio
            teacher_model.mask_ratio = 0.0
            teacher_outputs = teacher_model(audio_x, visual_x, use_cca_path=True, use_mae_path=False)
            teacher_model.mask_ratio = original_mask
        
        # ============================================
        # CCA Loss (on unmasked path)
        # ============================================
        cca_loss = cca_loss_fn(cca_outputs['z_a'], cca_outputs['z_v'])
        
        # ============================================
        # MAE Losses (on masked path)
        # ============================================
        loss_rec = (F.mse_loss(mae_outputs['rec_audio'], audio_x) + 
                   F.mse_loss(mae_outputs['rec_visual'], visual_x))
        
        loss_cross = (F.mse_loss(mae_outputs['rec_a2v'], visual_x) + 
                     F.mse_loss(mae_outputs['rec_v2a'], audio_x))
        
        # Consistency: MAE path should match teacher
        loss_consistency = (F.mse_loss(mae_outputs['z_a'], teacher_outputs['cca']['z_a']) +
                           F.mse_loss(mae_outputs['z_v'], teacher_outputs['cca']['z_v']))
        
        # Contrastive loss
        z_fused_t = 0.5 * (teacher_outputs['cca']['z_a'].detach() + 
                          teacher_outputs['cca']['z_v'].detach())
        sim_t = torch.matmul(F.normalize(z_fused_t), F.normalize(z_fused_t).t())
        pos_weights = get_topk_weights(sim_t, k=5, temperature=0.05)
        
        loss_contrastive = (
            soft_multi_positive_infoNCE(mae_outputs['z_a'], mae_outputs['z_v'], 
                                       pos_weights, temperature=0.05) +
            soft_multi_positive_infoNCE(mae_outputs['z_v'], mae_outputs['z_a'], 
                                       pos_weights, temperature=0.05)
        ) / 2
        
        # ============================================
        # Multi-task loss balancing
        # ============================================
#         losses = [cca_loss, loss_rec, loss_consistency, loss_contrastive]
        losses = [cca_loss, loss_rec, loss_cross, loss_consistency, loss_contrastive]

        total_loss = multi_task_loss(losses)
        
        with torch.no_grad():
            learned_weights = torch.exp(-multi_task_loss.log_vars).cpu().numpy()
        
        loss_dict = {
            'total': total_loss.item(),
            'cca': cca_loss.item(),
            'rec': loss_rec.item(),
            'cross': loss_cross.item(),
            'consist': loss_consistency.item(),
            'contrast': loss_contrastive.item(),
            'mode': 'joint',
            'weights': learned_weights
        }
        
        return total_loss, loss_dict
def compute_total_loss(cca_loss, mae_loss, epoch, total_epochs):
    """Cosine schedule: gradually shift from CCA to MAE"""
    lambda_t = 0.5 * (1 - math.cos(math.pi * epoch / total_epochs))
    return (1 - lambda_t) * cca_loss + lambda_t * mae_loss


def compute_mae_loss(model, audio_x, visual_x, rec_audio, rec_visual,
                     rec_a2v, rec_v2a, z_a, z_v, z_a_t, z_v_t,
                     multi_task_loss, epoch, warmup_epochs=5, top_k=5):
    """
    Improved warm-start approach with gradual transition
    """
    # 1. Intra-modal reconstruction
    loss_rec = F.mse_loss(rec_audio, audio_x) + F.mse_loss(rec_visual, visual_x)

    # 2. Cross-modal reconstruction
    loss_cross = F.mse_loss(rec_a2v, visual_x) + F.mse_loss(rec_v2a, audio_x)

    # 3. Teacher-student consistency
    loss_consistency = F.mse_loss(z_a, z_a_t) + F.mse_loss(z_v, z_v_t)

    # 4. Contrastive loss
    z_fused_t = 0.5 * (z_a_t.detach() + z_v_t.detach())
    sim_t = torch.matmul(F.normalize(z_fused_t), F.normalize(z_fused_t).t())
    pos_weights = get_topk_weights(sim_t, k=top_k, temperature=0.05)

    loss_contrastive = (
        soft_multi_positive_infoNCE(z_a, z_v, pos_weights, temperature=0.05) +
        soft_multi_positive_infoNCE(z_v, z_a, pos_weights, temperature=0.05)
    ) / 2

    # ============================================
    # WARM-START STRATEGY (Your Approach + Improvements)
    # ============================================
    
    if epoch < warmup_epochs:
        # Phase 1: Fixed weights with gradual cross-modal introduction
        # Gradually increase cross-modal from 0.0 to 1.0 over warmup
        cross_weight = min(1.0, epoch / max(1, warmup_epochs - 1))
        
        lambda_rec = 1.0
        lambda_cross = cross_weight  # 0.0 → 1.0
        lambda_contrastive = 0.05
        lambda_consistency = 0.1
        
        total_loss = (
            lambda_rec * loss_rec +
            lambda_cross * loss_cross +
            lambda_contrastive * loss_contrastive +
            lambda_consistency * loss_consistency
        )
        
        loss_dict = {
            'rec': loss_rec.item(),
            'cross': loss_cross.item(),
            'consist': loss_consistency.item(),
            'contrast': loss_contrastive.item(),
            'mode': 'fixed'
        }
    
    else:
        # Phase 2: Learnable weights
        losses = [loss_rec, loss_cross, loss_consistency, loss_contrastive]
        total_loss = multi_task_loss(losses)
        
        # For logging: get current learned weights
        with torch.no_grad():
            learned_weights = torch.exp(-multi_task_loss.log_vars).cpu().numpy()
        
        loss_dict = {
            'rec': loss_rec.item(),
            'cross': loss_cross.item(),
            'consist': loss_consistency.item(),
            'contrast': loss_contrastive.item(),
            'mode': 'learnable',
            'weights': learned_weights
        }
    
    return total_loss, loss_dict


def train_one_epoch(model, loss_fn, optimizer, scheduler, data_loaders, device,
                   warmup_epochs=5, output_dim=32, num_epochs=30, logger=None,
                   use_cca_loss=False):
    """
    Refined training loop
    
    Args:
        use_cca_loss: If True, add CCA loss for maximum performance
    """
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_w, best_m = None, None

    # Multi-task loss module (6 tasks now)
    multi_task_loss = MultiTaskLoss(num_tasks=6).to(device)
    
    # EMA Teacher
    teacher_model = copy.deepcopy(model)
    for p in teacher_model.parameters():
        p.requires_grad = False
    
    added_multitask_params = False
    
    for epoch in range(num_epochs):
        print(f'\n{"="*70}')
        print(f'Epoch {epoch+1}/{num_epochs}')
        if epoch < warmup_epochs:
            print(f'MODE: Fixed weights (warmup {epoch+1}/{warmup_epochs})')
        else:
            print(f'MODE: Learnable weights')
        print(f'{"="*70}')
        
        # Add multi-task params after warmup
        if not added_multitask_params and epoch >= warmup_epochs:
            print(" Switching to learnable task weights...")
            optimizer.add_param_group({
                'params': multi_task_loss.parameters(),
                'lr': 1e-3
            })
            added_multitask_params = True
        
        ema_decay = min(0.95 + 0.05 * (epoch / num_epochs), 0.999)

        # ============ TRAINING PHASE ============
        model.train()
        multi_task_loss.train()
        
        running_loss = 0.0
        running_cca_loss = 0.0
        running_mae_loss = 0.0
        
        epoch_loss_rec = 0.0
        epoch_loss_cross = 0.0
        epoch_loss_consist = 0.0
        epoch_loss_contrast = 0.0
        
        train_audio, train_visual = [], []

        for batch_idx, (audios, visuals, labels) in enumerate(data_loaders['train']):
            audios = audios.to(device)
            visuals = visuals.to(device)
            
            optimizer.zero_grad()
            
            # Student forward
            (audio_x, visual_x), (rec_audio, rec_visual), \
            (rec_a2v, rec_v2a), (z_a, z_v) = model(audios, visuals)
            
            # Teacher forward (no masking for stability)
            with torch.no_grad():
                original_mask_ratio = teacher_model.mask_ratio
                teacher_model.mask_ratio = 0.0
                _, _, _, (z_a_t, z_v_t) = teacher_model(audios, visuals)
                teacher_model.mask_ratio = original_mask_ratio
            
            # DCCA loss (optional)
            cca_loss = loss_fn(z_a, z_v) if use_cca_loss else torch.tensor(0.0, device=device)
            
            # MAE loss (refined)
            mae_loss, loss_dict = compute_mae_loss(
                model, audio_x, visual_x, rec_audio, rec_visual,
                rec_a2v, rec_v2a, z_a, z_v, z_a_t, z_v_t,
                multi_task_loss, epoch, warmup_epochs=warmup_epochs, top_k=5
            )
            
            # Combined loss
            if use_cca_loss:
                total_loss = compute_total_loss(cca_loss, mae_loss, epoch, num_epochs)
            else:
                total_loss = mae_loss
            
            # Backward
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # EMA update
            with torch.no_grad():
                for t_p, s_p in zip(teacher_model.parameters(), model.parameters()):
                    t_p.data.mul_(ema_decay).add_(s_p.data, alpha=(1 - ema_decay))
            
            # Accumulate losses
            running_loss += total_loss.item()
            running_cca_loss += cca_loss.item() if use_cca_loss else 0.0
            running_mae_loss += mae_loss.item()
            epoch_loss_rec += loss_dict['rec']
            epoch_loss_cross += loss_dict['cross']
            epoch_loss_consist += loss_dict['consist']
            epoch_loss_contrast += loss_dict['contrast']
            
            train_audio.append(z_a.detach().cpu().numpy())
            train_visual.append(z_v.detach().cpu().numpy())

            # Console logging
            if (batch_idx + 1) % 10 == 0:
                log_str = (f"  Batch {batch_idx+1}: Loss={total_loss.item():.4f} | "
                          f"Rec={loss_dict['rec']:.3f} | Cross={loss_dict['cross']:.3f} | "
                          f"Contrast={loss_dict['contrast']:.3f}")
                
                if loss_dict['mode'] == 'learnable':
                    weights = loss_dict['weights']
                    log_str += f"\n    Weights: {weights}"
                
                print(log_str)
        
        # Average losses
        num_batches = len(data_loaders['train'])
        avg_total_loss = running_loss / num_batches
        avg_cca_loss = running_cca_loss / num_batches
        avg_mae_loss = running_mae_loss / num_batches
        
        avg_loss_dict = {
            'rec': epoch_loss_rec / num_batches,
            'cross': epoch_loss_cross / num_batches,
            'consist': epoch_loss_consist / num_batches,
            'contrast': epoch_loss_contrast / num_batches
        }
        
        if epoch >= warmup_epochs:
            with torch.no_grad():
                avg_loss_dict['weights'] = torch.exp(-multi_task_loss.log_vars).cpu().numpy()
        
        # Train linear CCA
        train_audio = np.concatenate(train_audio)
        train_visual = np.concatenate(train_visual)
        train_w, train_m = applied_linear_cca(train_audio, train_visual, output_dim, "train")

        # ============ EVALUATION PHASE ============
        teacher_model.eval()
        t_audios, t_visuals, t_labels = [], [], []

        with torch.no_grad():
            for audios, visuals, labels in data_loaders['test']:
                audios = audios.to(device)
                visuals = visuals.to(device)
                
                _, _, _, (z_a, z_v) = teacher_model(audios, visuals)
                t_audios.append(z_a.cpu().numpy())
                t_visuals.append(z_v.cpu().numpy())
                t_labels.append(labels.cpu().numpy())

        _t_audios = np.concatenate(t_audios)
        _t_visuals = np.concatenate(t_visuals)
        t_labels = np.concatenate(t_labels).argmax(1)

        # Apply linear CCA
        audio_emb, visual_emb = applied_linear_cca(
            _t_audios, _t_visuals, output_dim, "test", input_info=(train_w, train_m)
        )
        
        # Also compute direct embedding performance
        audio_direct = _t_audios / (np.linalg.norm(_t_audios, axis=1, keepdims=True) + 1e-8)
        visual_direct = _t_visuals / (np.linalg.norm(_t_visuals, axis=1, keepdims=True) + 1e-8)
        a2v_d, v2a_d, avg_d = compute_map_metric(audio_direct, visual_direct, t_labels)

        # Compute MAP
        a2v, v2a, avg_map = compute_map_metric(audio_emb, visual_emb, t_labels)

        # Log results
        if logger:
            logger.log_epoch_train(epoch, avg_total_loss, avg_cca_loss, avg_mae_loss, avg_loss_dict)
            logger.log_epoch_test(epoch, a2v, v2a, avg_map)

        print(f"\n Epoch {epoch+1} Summary:")
        print(f"   Training Loss: {avg_total_loss:.4f} (CCA: {avg_cca_loss:.4f}, MAE: {avg_mae_loss:.4f})")
        print(f"   Testing (Direct): A2V={a2v_d:.4f}, V2A={v2a_d:.4f}, AVG={avg_d:.4f}")
        print(f"   Testing (w/CCA):  A2V={a2v:.4f}, V2A={v2a:.4f}, AVG={avg_map:.4f}")

        # Track best
        if avg_map > best_acc:
            print(f"   ✨ New best MAP: {best_acc:.4f} → {avg_map:.4f}")
            best_acc = avg_map
            best_a2v, best_v2a = a2v, v2a
            best_model_wts = copy.deepcopy(teacher_model.state_dict())
            best_w, best_m = train_w, train_m

        scheduler.step()

    # Load best weights
    teacher_model.load_state_dict(best_model_wts)
    teacher_model.mask_ratio = 0.0

    total_time = time.time() - since
    print(f"\n{'='*70}")
    print(f" Training complete in {total_time//60:.0f}m {total_time%60:.0f}s")
    print(f" Best MAP: {best_acc:.4f} (A2V: {best_a2v:.4f}, V2A: {best_v2a:.4f})")
    print(f"{'='*70}")

    return teacher_model, (best_w, best_m), logger


# ============ MAIN EXECUTION ============
if __name__ == "__main__":
    prefix = input("Enter prefix (e.g., 'train', 'test', 'val'): ")
    confirm_message = input("Combination type: ")
    use_cca = input("Use CCA loss? (y/n, default=n): ").lower() == 'y'
    
    embedding_dir = Path('learned_embedding')
    embedding_dir.mkdir(exist_ok=True)
    log_dir = Path('training_logs')
    log_dir.mkdir(exist_ok=True)
    
    filename = embedding_dir / f'{prefix}_embeddings_and_labels.npz'

    myconfig = get_config("vegas")
    dataloader, input_database = load_data(myconfig.DATA_PATH, BaseConfig.BATCH_SIZE_VEGAS)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Hyperparameters
    output_dim_size = 10
    cca_output_size = 32
    N_epoch = 20
    Mask_ratio = 0.2
    Warmup_epochs = 5

    logger = TrainingLogger(save_dir='training_logs')
    
    config = {
        'prefix': prefix,
        'output_dim_size': output_dim_size,
        'cca_output_size': cca_output_size,
        'n_epochs': N_epoch,
        'mask_ratio': Mask_ratio,
        'warmup_epochs': Warmup_epochs,
        'use_cca_loss': use_cca,
        'learning_rate': 3e-4,
        'weight_decay': 1e-4,
        'batch_size': BaseConfig.BATCH_SIZE_VEGAS,
        'device': device,
        'combination': confirm_message
    }
    logger.set_config(config)

    model = UTAV_MAE(
        audio_input_dim=128,
        visual_input_dim=1024,
        embed_dim=1024,
        out_dim=cca_output_size,
        mask_ratio=Mask_ratio,
        temporal_pool="mean"
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    loss_fn = DCCALoss(out_dim=cca_output_size, reg=1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_epoch)

    print("\n" + "="*70)
    print(f" Starting Training {'WITH' if use_cca else 'WITHOUT'} CCA Loss")
    print("="*70)
    
    best_model, best_inputs_info, logger = train_one_epoch(
        model, loss_fn, optimizer, scheduler, dataloader, device,
        warmup_epochs=Warmup_epochs,
        output_dim=output_dim_size,
        num_epochs=N_epoch,
        logger=logger,
        use_cca_loss=use_cca
    )

    timestamp = logger.save_logs(prefix=prefix)

    # Final evaluation
    best_model.mask_ratio = 0.0
    best_model.eval()
    print('\n...Final Evaluation...')
    
    with torch.no_grad():
        _, _, _, (audio_feature, visual_feature) = best_model(
            torch.tensor(input_database['audio_test']).to(device),
            torch.tensor(input_database['visual_test']).to(device)
        )

    label = torch.argmax(torch.tensor(input_database['label_test']), dim=1)
    audio_feature, visual_feature = applied_linear_cca(
        audio_feature.detach().cpu().numpy(),
        visual_feature.detach().cpu().numpy(),
        output_dim=output_dim_size,
        mode_type="test",
        input_info=best_inputs_info
    )

     # Save embeddings
    np.savez(filename,
             audio_feature=audio_feature,
             visual_feature=visual_feature,
             label=label,
             timestamp=timestamp)

    print(f"\n Embeddings saved to: {filename}")

    a2v, v2a, avg = compute_map_metric(audio_feature, visual_feature, label)
    print(f"\n Final Test Results:")
    print(f"   Audio→Visual: {a2v:.4f}")
    print(f"   Visual→Audio: {v2a:.4f}")
    print(f"   Average MAP: {avg:.4f}")
    print(f"\n{confirm_message}")
    
    print(f"\n All results saved with timestamp: {timestamp}")
    
        