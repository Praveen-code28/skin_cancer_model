"""
Training module for skin cancer classification.
Includes training loop with AMP, class-weighted loss, Focal Loss, 
CosineAnnealingLR, early stopping, and gradient accumulation.
"""

import time
import logging
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from utils import AverageMeter, EarlyStopping, get_device
from models import SkinCancerModel, save_checkpoint, load_checkpoint


logger = logging.getLogger("skin_cancer_pipeline")


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy with Label Smoothing.
    """
    
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = float(smoothing)
        self.confidence = 1.0 - self.smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(inputs, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def get_criterion(config: Dict[str, Any], class_weights: Optional[torch.Tensor] = None) -> nn.Module:
    """
    Get loss function based on configuration.
    
    Args:
        config: Configuration dictionary
        class_weights: Optional class weights tensor
    
    Returns:
        Loss function module
    """
    loss_config = config['training']['loss']
    loss_type = loss_config['type']
    
    if loss_type == 'ce':
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        logger.info("Using CrossEntropyLoss" + (" with class weights" if class_weights is not None else ""))
    elif loss_type == 'focal':
        criterion = FocalLoss(
            alpha=float(loss_config.get('focal_alpha', 0.25)),
            gamma=float(loss_config.get('focal_gamma', 2.0))
        )
        logger.info(f"Using FocalLoss (alpha={float(loss_config.get('focal_alpha', 0.25))}, gamma={float(loss_config.get('focal_gamma', 2.0))})")
    elif loss_type == 'label_smoothing':
        criterion = LabelSmoothingCrossEntropy(smoothing=float(loss_config.get('label_smoothing', 0.1)))
        logger.info(f"Using LabelSmoothingCrossEntropy (smoothing={float(loss_config.get('label_smoothing', 0.1))})")
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    return criterion


def get_optimizer(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """
    Get optimizer based on configuration.
    
    Args:
        model: Model to optimize
        config: Configuration dictionary
    
    Returns:
        Optimizer instance
    """
    train_config = config['training']
    lr = float(train_config['learning_rate'])
    weight_decay = float(train_config['weight_decay'])
    optimizer_type = train_config['optimizer']
    
    # Separate backbone and classifier parameters for differential learning rates
    backbone_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if 'classifier' in name:
            classifier_params.append(param)
        else:
            backbone_params.append(param)
    
    param_groups = [
        {'params': backbone_params, 'lr': lr * 0.1},  # Lower LR for pretrained backbone
        {'params': classifier_params, 'lr': lr}
    ]
    
    if optimizer_type == 'adamw':
        optimizer = AdamW(param_groups, weight_decay=weight_decay)
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(param_groups, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(param_groups, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    logger.info(f"Using {optimizer_type.upper()} optimizer with lr={lr}, weight_decay={weight_decay}")
    logger.info(f"Backbone LR: {lr * 0.1}, Classifier LR: {lr}")
    
    return optimizer


def get_scheduler(optimizer: torch.optim.Optimizer, config: Dict[str, Any]) -> Any:
    """
    Get learning rate scheduler based on configuration.
    
    Args:
        optimizer: Optimizer instance
        config: Configuration dictionary
    
    Returns:
        Scheduler instance
    """
    train_config = config['training']
    scheduler_type = train_config['scheduler']
    scheduler_params = train_config.get('scheduler_params', {})
    
    if scheduler_type == 'cosine_annealing_warm_restarts':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=int(scheduler_params.get('T_0', 10)),
            T_mult=int(scheduler_params.get('T_mult', 2)),
            eta_min=float(scheduler_params.get('eta_min', 1e-6))
        )
    elif scheduler_type == 'cosine_annealing':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(train_config['epochs']),
            eta_min=float(scheduler_params.get('eta_min', 1e-6))
        )
    elif scheduler_type == 'reduce_lr_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            min_lr=float(scheduler_params.get('eta_min', 1e-6))
        )
    elif scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(scheduler_params.get('step_size', 10)),
            gamma=float(scheduler_params.get('gamma', 0.1))
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")
    
    logger.info(f"Using {scheduler_type} scheduler")
    
    return scheduler


def _compute_epoch_metrics(all_preds: np.ndarray, all_labels: np.ndarray, num_classes: int) -> Dict[str, float]:
    """
    Compute epoch-level metrics with warning suppression.
    
    Args:
        all_preds: All predictions for the epoch
        all_labels: All labels for the epoch
        num_classes: Number of classes
    
    Returns:
        Dictionary with epoch metrics
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='y_pred contains classes not in y_true')
        
        acc = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0, labels=list(range(num_classes)))
        bal_acc = balanced_accuracy_score(all_labels, all_preds)
    
    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'balanced_accuracy': bal_acc
    }


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    config: Dict[str, Any],
    epoch: int,
    num_classes: int,
    log_interval: int = 50
) -> Dict[str, float]:
    """
    Train for one epoch with real-time progress display.
    
    Args:
        model: Model to train
        loader: Training DataLoader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        scaler: GradScaler for AMP
        config: Configuration dictionary
        epoch: Current epoch number
        num_classes: Number of classes
        log_interval: Logging interval for progress
    
    Returns:
        Dictionary with training metrics
    """
    model.train()
    
    loss_meter = AverageMeter()
    epoch_preds = []
    epoch_labels = []
    
    grad_accum_steps = int(config['training'].get('gradient_accumulation_steps', 1))
    grad_clip = float(config['training'].get('gradient_clip', 1.0))
    use_amp = bool(config['training'].get('amp', True))
    
    # Determine device type for AMP
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    
    optimizer.zero_grad()
    
    # Setup progress bar
    use_tqdm = HAS_TQDM and logger.isEnabledFor(logging.INFO)
    
    if use_tqdm:
        pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False, dynamic_ncols=True)
        iterator = enumerate(pbar)
    else:
        iterator = enumerate(loader)
    
    epoch_start = time.time()
    
    for step, batch in iterator:
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        
        # Forward pass with AMP
        if use_amp and device_type == 'cuda':
            with autocast(device_type=device_type):
                logits = model(images)
                loss = criterion(logits, labels)
                loss = loss / grad_accum_steps
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss = loss / grad_accum_steps
        
        # Backward pass
        if use_amp and device_type == 'cuda':
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation step
        if (step + 1) % grad_accum_steps == 0:
            if use_amp and device_type == 'cuda':
                scaler.unscale_(optimizer)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            
            optimizer.zero_grad()
        
        # Collect predictions for epoch-level metrics
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            loss_val = loss.item() * grad_accum_steps
            
            loss_meter.update(loss_val, images.size(0))
            
            epoch_preds.append(preds.detach().cpu().numpy())
            epoch_labels.append(labels.detach().cpu().numpy())
        
        # Update progress bar with live metrics
        if use_tqdm:
            elapsed = time.time() - epoch_start
            it_per_sec = (step + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(loader) - step - 1) / it_per_sec if it_per_sec > 0 else 0
            
            vram_gb = torch.cuda.memory_allocated() / 1e9 if device.type == 'cuda' else 0
            lr = optimizer.param_groups[0]['lr']
            
            pbar.set_postfix({
                'Loss': f'{loss_meter.avg:.4f}',
                'LR': f'{lr:.2e}',
                'VRAM': f'{vram_gb:.2f}GB',
                'ETA': f'{remaining:.0f}s'
            })
        elif step % log_interval == 0 and step > 0:
            # Non-tqdm logging at intervals
            vram_gb = torch.cuda.memory_allocated() / 1e9 if device.type == 'cuda' else 0
            logger.info(
                f"Epoch {epoch} | Step {step}/{len(loader)} | "
                f"Loss: {loss_meter.avg:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e} | "
                f"VRAM: {vram_gb:.2f}GB"
            )
    
    # Compute epoch-level metrics on all accumulated predictions
    all_preds = np.concatenate(epoch_preds)
    all_labels = np.concatenate(epoch_labels)
    epoch_metrics = _compute_epoch_metrics(all_preds, all_labels, num_classes)
    
    return {
        'loss': loss_meter.avg,
        'accuracy': epoch_metrics['accuracy'],
        'macro_f1': epoch_metrics['macro_f1'],
        'balanced_accuracy': epoch_metrics['balanced_accuracy']
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: Dict[str, Any],
    num_classes: int
) -> Dict[str, float]:
    """
    Validate model on validation set.
    
    Args:
        model: Model to validate
        loader: Validation DataLoader
        criterion: Loss function
        device: Device to validate on
        config: Configuration dictionary
        num_classes: Number of classes
    
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    
    loss_meter = AverageMeter()
    all_preds = []
    all_labels = []
    all_probs = []
    
    use_amp = bool(config['training'].get('amp', True))
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    
    for batch in loader:
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        
        if use_amp and device_type == 'cuda':
            with autocast(device_type=device_type):
                logits = model(images)
                loss = criterion(logits, labels)
        else:
            logits = model(images)
            loss = criterion(logits, labels)
        
        loss_meter.update(loss.item(), images.size(0))
        
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    
    val_metrics = _compute_epoch_metrics(all_preds, all_labels, num_classes)
    
    metrics = {
        'loss': loss_meter.avg,
        'accuracy': val_metrics['accuracy'],
        'balanced_accuracy': val_metrics['balanced_accuracy'],
        'macro_f1': val_metrics['macro_f1'],
        'per_class_f1': f1_score(all_labels, all_preds, average=None, zero_division=0, labels=list(range(num_classes))).tolist(),
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }
    
    return metrics


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: Dict[str, Any],
    class_weights: Optional[torch.Tensor] = None,
    checkpoint_dir: Optional[Path] = None,
    resume_from: Optional[Path] = None,
    model_name: str = "model"
) -> Dict[str, Any]:
    """
    Full training loop with early stopping and checkpointing.
    
    Args:
        model: Model to train
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        config: Configuration dictionary
        class_weights: Optional class weights for loss
        checkpoint_dir: Directory to save checkpoints
        resume_from: Optional checkpoint to resume from
        model_name: Name of model for display
    
    Returns:
        Training history dictionary
    """
    device = get_device(config['experiment']['device'])
    model = model.to(device)
    
    num_classes = len(config.get('class_names', ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']))
    
    # Setup training components
    criterion = get_criterion(config, class_weights.to(device) if class_weights is not None else None)
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    scaler = GradScaler(device='cuda' if device.type == 'cuda' else 'cpu', enabled=bool(config['training'].get('amp', True)))
    
    # Early stopping
    early_stopping_config = config['training']['early_stopping']
    early_stopping = EarlyStopping(
        patience=int(early_stopping_config['patience']),
        min_delta=float(early_stopping_config['min_delta']),
        mode=early_stopping_config['mode'],
        monitor=early_stopping_config['monitor']
    ) if early_stopping_config['enabled'] else None
    
    # Resume from checkpoint
    start_epoch = 0
    best_metric = -np.inf if early_stopping_config['mode'] == 'max' else np.inf
    history = {
        'train_loss': [], 'train_acc': [], 'train_macro_f1': [], 'train_bal_acc': [],
        'val_loss': [], 'val_acc': [], 'val_macro_f1': [], 'val_bal_acc': [],
        'lr': []
    }
    
    if resume_from and resume_from.exists():
        metadata = load_checkpoint(model, resume_from, device, optimizer, scheduler)
        start_epoch = metadata['epoch'] + 1
        best_metric = metadata['best_metric']
        logger.info(f"Resumed from epoch {start_epoch}")
    
    epochs = int(config['training']['epochs'])
    logger.info(f"Starting training for {epochs} epochs (from epoch {start_epoch})")
    
    # Print header for progress
    logger.info(f"Model: {model_name} | Device: {device} | Epochs: {epochs} | Batches/epoch: {len(train_loader)}")
    
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, config, epoch, num_classes
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, config, num_classes)
        
        # Learning rate scheduling
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_metrics['macro_f1'])
        else:
            scheduler.step()
        
        # Record history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['train_macro_f1'].append(train_metrics['macro_f1'])
        history['train_bal_acc'].append(train_metrics['balanced_accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_macro_f1'].append(val_metrics['macro_f1'])
        history['val_bal_acc'].append(val_metrics['balanced_accuracy'])
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        epoch_time = time.time() - epoch_start
        
        # Checkpointing and best metric tracking
        is_best = False
        if checkpoint_dir:
            monitor_metric = val_metrics[early_stopping_config['monitor']]
            
            if early_stopping_config['mode'] == 'max':
                if monitor_metric > best_metric:
                    best_metric = monitor_metric
                    is_best = True
            else:
                if monitor_metric < best_metric:
                    best_metric = monitor_metric
                    is_best = True
            
            if config['training'].get('save_best_only', True) and is_best:
                save_checkpoint(
                    model, optimizer, scheduler, epoch, best_metric, config,
                    checkpoint_dir / f"best_model.pt", is_best=True
                )
            
            if config['training'].get('save_last', True):
                save_checkpoint(
                    model, optimizer, scheduler, epoch, best_metric, config,
                    checkpoint_dir / f"last_model.pt", is_best=False
                )
        
        # Early stopping check
        early_stop = False
        if early_stopping:
            monitor_metric = val_metrics[early_stopping_config['monitor']]
            early_stop = early_stopping(monitor_metric, epoch)
        
        # Print clean epoch summary
        vram_gb = torch.cuda.memory_allocated() / 1e9 if device.type == 'cuda' else 0
        best_marker = " [*NEW BEST*]" if is_best else ""
        
        logger.info(
            f"\n--------------------------------------------------------------------------------\n"
            f"[{model_name}] Epoch {epoch+1:02d}/{epochs:02d} Complete ({epoch_time:.1f}s) | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e} | VRAM: {vram_gb:.2f} GB{best_marker}\n"
            f"  Train -> Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.4f} | "
            f"Macro F1: {train_metrics['macro_f1']:.4f} | Bal Acc: {train_metrics['balanced_accuracy']:.4f}\n"
            f"  Val   -> Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | "
            f"Macro F1: {val_metrics['macro_f1']:.4f} | Bal Acc: {val_metrics['balanced_accuracy']:.4f}{best_marker}\n"
            f"--------------------------------------------------------------------------------"
        )
        
        if early_stop:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            logger.info(f"Best {early_stopping_config['monitor']}: {early_stopping.best_score:.4f} at epoch {early_stopping.best_epoch+1}")
            break
    
    logger.info("Training completed!")
    
    return history


if __name__ == "__main__":
    import yaml
    from utils import setup_logging, set_seed, create_directories
    from dataset import load_and_split_data, create_data_loaders, compute_class_weights
    from models import create_model, get_model_list
    
    setup_logging(Path("test_logs"), "INFO", True)
    set_seed(42)
    
    with open("../configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    dirs = create_directories(config)
    
    # Load data
    train_df, val_df, test_df, label_encoder = load_and_split_data(config)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(config, train_df, val_df, test_df, label_encoder)
    
    # Compute class weights
    class_weights = compute_class_weights(train_df, label_encoder) if config['data']['use_class_weights'] else None
    
    # Test with first model
    model_config = config['models'][0]
    model = create_model(config, model_config)
    
    # Quick training test (1 epoch)
    config['training']['epochs'] = 1
    history = train_model(model, train_loader, val_loader, config, class_weights, dirs['checkpoints'])
    
    print("Training test passed!")
    print(f"History keys: {history.keys()}")