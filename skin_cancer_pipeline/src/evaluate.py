"""
Evaluation module for skin cancer classification.
Computes comprehensive metrics, generates ROC/PR curves, confusion matrices, and Grad-CAM visualizations.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    classification_report, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, auc
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from utils import get_class_names, save_json
from models import get_last_conv_layer


logger = logging.getLogger("skin_cancer_pipeline")


class GradCAM:
    """
    Grad-CAM implementation for model interpretability.
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class index (if None, uses predicted class)
        
        Returns:
            Heatmap as numpy array (H, W)
        """
        self.model.eval()
        
        # Ensure input tensor requires gradients for backward pass
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        
        # Forward and backward pass with gradients enabled
        with torch.enable_grad():
            # Forward pass
            logits = self.model(input_tensor)
            
            if target_class is None:
                target_class = logits.argmax(dim=1).item()
            
            # Zero gradients
            self.model.zero_grad()
            
            # Backward pass for target class
            target_logit = logits[0, target_class]
            target_logit.backward(retain_graph=True)
            
            # Get gradients and activations
            gradients = self.gradients[0]  # (C, H, W)
            activations = self.activations[0]  # (C, H, W)
            
            # Ensure tensors are on the same device
            device = activations.device
            gradients = gradients.to(device)
            
            # Compute weights (global average pooling of gradients)
            weights = gradients.mean(dim=(1, 2))  # (C,)
            
            # Weighted combination of activations
            cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=device)
            for i, w in enumerate(weights):
                cam += w * activations[i]
            
            # ReLU
            cam = F.relu(cam)
            
            # Normalize
            if cam.max() > 0:
                cam = cam / cam.max()
            
            # Resize to input size
            cam = F.interpolate(
                cam.unsqueeze(0).unsqueeze(0),
                size=input_tensor.shape[2:],
                mode='bilinear',
                align_corners=False
            ).squeeze()
        
        return cam.cpu().numpy()
    
    def __call__(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        return self.generate(input_tensor, target_class)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str]
) -> Dict[str, Any]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (n_samples, n_classes)
        class_names: List of class names
    
    Returns:
        Dictionary of metrics
    """
    n_classes = len(class_names)
    
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'macro_precision': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'macro_recall': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
    }
    
    # Per-class metrics - use labels parameter to ensure all classes are included
    labels = list(range(n_classes))
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0, labels=labels)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0, labels=labels)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0, labels=labels)
    
    metrics['per_class'] = {}
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    for i, name in enumerate(class_names):
        # Specificity = TN / (TN + FP)
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        metrics['per_class'][name] = {
            'precision': float(per_class_precision[i]),
            'recall_sensitivity': float(per_class_recall[i]),
            'specificity': float(specificity),
            'f1': float(per_class_f1[i]),
            'support': int((y_true == i).sum())
        }
    
    # ROC-AUC (One-vs-Rest and One-vs-One)
    try:
        metrics['roc_auc_ovr'] = float(roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro'))
        metrics['roc_auc_ovr_weighted'] = float(roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted'))
    except ValueError:
        metrics['roc_auc_ovr'] = 0.0
        metrics['roc_auc_ovr_weighted'] = 0.0
    
    try:
        metrics['roc_auc_ovo'] = float(roc_auc_score(y_true, y_prob, multi_class='ovo', average='macro'))
    except ValueError:
        metrics['roc_auc_ovo'] = 0.0
    
    # Per-class ROC-AUC
    per_class_roc_auc = {}
    for i, name in enumerate(class_names):
        try:
            # One-vs-Rest for each class
            y_true_binary = (y_true == i).astype(int)
            per_class_roc_auc[name] = float(roc_auc_score(y_true_binary, y_prob[:, i]))
        except ValueError:
            per_class_roc_auc[name] = 0.0
    metrics['per_class_roc_auc'] = per_class_roc_auc
    
    # PR-AUC
    try:
        metrics['pr_auc_macro'] = float(average_precision_score(y_true, y_prob, average='macro'))
    except ValueError:
        metrics['pr_auc_macro'] = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    metrics['confusion_matrix'] = cm.tolist()
    # Normalize confusion matrix, handling zero rows
    cm_norm = cm.astype('float')
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    metrics['confusion_matrix_normalized'] = (cm_norm / row_sums).tolist()
    
    # Classification report
    metrics['classification_report'] = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0,
        labels=list(range(n_classes))
    )
    
    return metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Path,
    normalize: Optional[str] = 'true',
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Plot and save confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: Class names
        save_path: Path to save figure
        normalize: Normalization mode ('true', 'pred', 'all', or None)
        figsize: Figure size
    """
    if normalize == 'true':
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = '.2f'
        title = 'Normalized Confusion Matrix (True Labels)'
    elif normalize == 'pred':
        cm_norm = cm.astype('float') / cm.sum(axis=0, keepdims=True)
        fmt = '.2f'
        title = 'Normalized Confusion Matrix (Predicted Labels)'
    elif normalize == 'all':
        cm_norm = cm.astype('float') / cm.sum()
        fmt = '.2f'
        title = 'Normalized Confusion Matrix (All)'
    else:
        cm_norm = cm
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm_norm, annot=True, fmt=fmt, cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    plt.title(title, fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    save_path: Path,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Plot multi-class ROC curves.
    
    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        class_names: Class names
        save_path: Path to save figure
        figsize: Figure size
    """
    n_classes = len(class_names)
    
    plt.figure(figsize=figsize)
    
    # Plot each class
    for i, name in enumerate(class_names):
        y_true_binary = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
    
    # Plot macro-average
    try:
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
        fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
        roc_auc_macro = auc(fpr, tpr)
        plt.plot(fpr, tpr, 'k--', lw=2, label=f'Macro-average (AUC = {roc_auc_macro:.3f})')
    except Exception:
        pass
    
    plt.plot([0, 1], [0, 1], 'k:', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Multi-class ROC Curves (One-vs-Rest)', fontsize=14)
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    save_path: Path,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Plot multi-class Precision-Recall curves.
    
    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        class_names: Class names
        save_path: Path to save figure
        figsize: Figure size
    """
    n_classes = len(class_names)
    
    plt.figure(figsize=figsize)
    
    for i, name in enumerate(class_names):
        y_true_binary = (y_true == i).astype(int)
        precision, recall, _ = precision_recall_curve(y_true_binary, y_prob[:, i])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, lw=2, label=f'{name} (AUC = {pr_auc:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Multi-class Precision-Recall Curves', fontsize=14)
    plt.legend(loc='lower left', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Path,
    figsize: Tuple[int, int] = (15, 10)
) -> None:
    """
    Plot training and validation curves.
    
    Args:
        history: Training history dictionary
        save_path: Path to save figure
        figsize: Figure size
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Macro F1
    axes[1, 0].plot(epochs, history['train_macro_f1'], 'b-', label='Train Macro F1')
    axes[1, 0].plot(epochs, history['val_macro_f1'], 'r-', label='Val Macro F1')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Macro F1')
    axes[1, 0].set_title('Training and Validation Macro F1')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Balanced Accuracy
    axes[1, 1].plot(epochs, history['train_bal_acc'], 'b-', label='Train Bal Acc')
    axes[1, 1].plot(epochs, history['val_bal_acc'], 'r-', label='Val Bal Acc')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Balanced Accuracy')
    axes[1, 1].set_title('Training and Validation Balanced Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_per_class_metrics(
    metrics: Dict[str, Any],
    class_names: List[str],
    save_path: Path,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Plot per-class metrics as grouped bar chart.
    
    Args:
        metrics: Metrics dictionary
        class_names: Class names
        save_path: Path to save figure
        figsize: Figure size
    """
    per_class = metrics['per_class']
    
    x = np.arange(len(class_names))
    width = 0.2
    
    precision = [per_class[name]['precision'] for name in class_names]
    recall = [per_class[name]['recall_sensitivity'] for name in class_names]
    specificity = [per_class[name]['specificity'] for name in class_names]
    f1 = [per_class[name]['f1'] for name in class_names]
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - 1.5*width, precision, width, label='Precision', color='skyblue')
    ax.bar(x - 0.5*width, recall, width, label='Recall (Sensitivity)', color='lightgreen')
    ax.bar(x + 0.5*width, specificity, width, label='Specificity', color='salmon')
    ax.bar(x + 1.5*width, f1, width, label='F1 Score', color='gold')
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Metrics', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_gradcam_visualizations(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: List[str],
    config: Dict[str, Any],
    save_dir: Path,
    device: torch.device,
    model_name: str = None
) -> None:
    """
    Generate Grad-CAM visualizations for test samples.
    
    Args:
        model: Trained model
        test_loader: Test DataLoader
        class_names: Class names
        config: Configuration dictionary
        save_dir: Directory to save visualizations
        device: Device to run on
        model_name: Name of the model architecture (e.g., 'efficientnet_b0', 'convnext_tiny', 'resnet50d')
    """
    gradcam_config = config['evaluation']['gradcam']
    if not gradcam_config.get('enabled', True):
        return
    
    logger.info("Generating Grad-CAM visualizations...")
    
    # Get target layer - use provided model_name or fallback to first model in config
    timm_name = model_name if model_name else config['models'][0]['timm_name']
    try:
        target_layer = get_last_conv_layer(model, timm_name)
    except Exception as e:
        logger.warning(f"Could not auto-detect target layer for {timm_name}: {e}")
        return
    
    gradcam = GradCAM(model, target_layer)
    
    # Denormalization for visualization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    num_samples = gradcam_config.get('num_samples_per_class', 5)
    alpha = gradcam_config.get('alpha', 0.5)
    colormap = gradcam_config.get('colormap', 'jet')
    
    # Create model-specific subdirectory
    model_subdir = save_dir / (model_name if model_name else 'default')
    model_subdir.mkdir(parents=True, exist_ok=True)
    
    # Collect samples per class
    class_samples = {i: [] for i in range(len(class_names))}
    
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            preds = model(images).argmax(dim=1)
            
            for i in range(len(images)):
                true_label = labels[i].item()
                pred_label = preds[i].item()
                
                # Only collect correct predictions (TP) and incorrect (FN)
                if true_label == pred_label and len(class_samples[true_label]) < num_samples:
                    class_samples[true_label].append({
                        'image': images[i].cpu(),
                        'true_label': true_label,
                        'pred_label': pred_label,
                        'image_id': batch['image_id'][i] if 'image_id' in batch else f'sample_{i}'
                    })
                elif true_label != pred_label and len(class_samples[true_label]) < num_samples:
                    class_samples[true_label].append({
                        'image': images[i].cpu(),
                        'true_label': true_label,
                        'pred_label': pred_label,
                        'image_id': batch['image_id'][i] if 'image_id' in batch else f'sample_{i}'
                    })
    
    # Generate visualizations
    for class_idx, samples in class_samples.items():
        for j, sample in enumerate(samples):
            image_tensor = sample['image'].unsqueeze(0).to(device)
            true_label = sample['true_label']
            pred_label = sample['pred_label']
            
            # Generate Grad-CAM for true class
            cam_true = gradcam(image_tensor, target_class=true_label)
            
            # Generate Grad-CAM for predicted class
            cam_pred = gradcam(image_tensor, target_class=pred_label)
            
            # Prepare original image
            img = sample['image'].permute(1, 2, 0).numpy()
            img = img * std + mean
            img = np.clip(img, 0, 1)
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(img)
            axes[0].set_title(f"Original\nTrue: {class_names[true_label]}, Pred: {class_names[pred_label]}")
            axes[0].axis('off')
            
            # Grad-CAM for true class
            axes[1].imshow(img)
            axes[1].imshow(cam_true, cmap=colormap, alpha=alpha)
            axes[1].set_title(f"Grad-CAM (True: {class_names[true_label]})")
            axes[1].axis('off')
            
            # Grad-CAM for predicted class
            axes[2].imshow(img)
            axes[2].imshow(cam_pred, cmap=colormap, alpha=alpha)
            axes[2].set_title(f"Grad-CAM (Pred: {class_names[pred_label]})")
            axes[2].axis('off')
            
            plt.tight_layout()
            
            # Save
            status = "TP" if true_label == pred_label else "FN"
            save_path = model_subdir / f"{class_names[class_idx]}_{status}_{sample['image_id']}_{j}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
    
    logger.info(f"Saved Grad-CAM visualizations to {model_subdir}")


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    save_dirs: Dict[str, Path],
    class_names: List[str]
) -> Dict[str, Any]:
    """
    Full evaluation on test set.
    
    Args:
        model: Trained model
        test_loader: Test DataLoader
        config: Configuration dictionary
        device: Device to evaluate on
        save_dirs: Dictionary of save directories
        class_names: Class names
    
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_image_ids = []
    
    for batch in test_loader:
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_image_ids.extend(batch['image_id'])
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    
    # Compute metrics
    metrics = compute_metrics(all_labels, all_preds, all_probs, class_names)
    
    # Save predictions
    if config['output'].get('save_predictions', True):
        pred_df = pd.DataFrame({
            'image_id': all_image_ids,
            'true_label': [class_names[i] for i in all_labels],
            'pred_label': [class_names[i] for i in all_preds]
        })
        for i, name in enumerate(class_names):
            pred_df[f'prob_{name}'] = all_probs[:, i]
        pred_df.to_csv(save_dirs['metrics'] / 'test_predictions.csv', index=False)
    
    # Save metrics
    save_json(metrics, save_dirs['metrics'] / 'test_metrics.json')
    
    # Generate plots
    logger.info("Generating evaluation plots...")
    
    # Confusion matrix
    plot_confusion_matrix(
        np.array(metrics['confusion_matrix']),
        class_names,
        save_dirs['plots'] / 'confusion_matrix.png',
        normalize=config['evaluation']['confusion_matrix'].get('normalize', 'true')
    )
    plot_confusion_matrix(
        np.array(metrics['confusion_matrix']),
        class_names,
        save_dirs['plots'] / 'confusion_matrix_raw.png',
        normalize=None
    )
    
    # ROC curves
    plot_roc_curves(all_labels, all_probs, class_names, save_dirs['plots'] / 'roc_curves.png')
    
    # PR curves
    plot_pr_curves(all_labels, all_probs, class_names, save_dirs['plots'] / 'pr_curves.png')
    
    # Per-class metrics
    plot_per_class_metrics(metrics, class_names, save_dirs['plots'] / 'per_class_metrics.png')
    
    # Grad-CAM
    # Determine model's timm_name from config based on the model architecture
    model_timm_name = None
    for m in config.get('models', []):
        if m.get('name') in str(model.__class__).lower() or m.get('timm_name', '').lower() in str(model.__class__).lower():
            model_timm_name = m.get('timm_name')
            break
    if not model_timm_name:
        model_timm_name = config['models'][0]['timm_name']
    
    generate_gradcam_visualizations(
        model, test_loader, class_names, config, save_dirs['gradcam'], device, model_timm_name
    )
    
    # Log summary
    logger.info("=" * 60)
    logger.info("TEST SET EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Accuracy:           {metrics['accuracy']:.4f}")
    logger.info(f"Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}")
    logger.info(f"Macro F1:           {metrics['macro_f1']:.4f}")
    logger.info(f"Macro Precision:    {metrics['macro_precision']:.4f}")
    logger.info(f"Macro Recall:       {metrics['macro_recall']:.4f}")
    logger.info(f"ROC-AUC (OvR):      {metrics['roc_auc_ovr']:.4f}")
    logger.info(f"ROC-AUC (OvO):      {metrics['roc_auc_ovo']:.4f}")
    logger.info(f"PR-AUC (Macro):     {metrics['pr_auc_macro']:.4f}")
    logger.info("-" * 60)
    for name in class_names:
        pc = metrics['per_class'][name]
        logger.info(
            f"{name:4s} | Prec: {pc['precision']:.3f} | "
            f"Rec: {pc['recall_sensitivity']:.3f} | "
            f"Spec: {pc['specificity']:.3f} | "
            f"F1: {pc['f1']:.3f} | "
            f"Supp: {pc['support']} | "
            f"AUC: {metrics['per_class_roc_auc'][name]:.3f}"
        )
    logger.info("=" * 60)
    
    return metrics


def run_eda(config: Dict[str, Any], save_dir: Path) -> None:
    """
    Run exploratory data analysis and save visualizations.
    
    Args:
        config: Configuration dictionary
        save_dir: Directory to save EDA outputs
    """
    import pandas as pd
    
    logger.info("Running EDA...")
    
    root_path = Path(config['data']['root_path'])
    labels_csv = root_path / config['data']['labels_csv']
    df = pd.read_csv(labels_csv)
    
    class_names = get_class_names(config)
    
    # Class distribution
    plt.figure(figsize=(10, 6))
    split_order = ['train', 'val', 'test']
    df_plot = df.copy()
    df_plot['split'] = pd.Categorical(df_plot['split'], categories=split_order, ordered=True)
    
    ax = df_plot.groupby(['split', 'label']).size().unstack().plot(
        kind='bar', stacked=True, figsize=(10, 6), colormap='tab20'
    )
    ax.set_title('Class Distribution by Split', fontsize=14)
    ax.set_xlabel('Split', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_dir / 'class_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Sample grid (if exists)
    sample_grid_path = root_path / 'sample_grid.png'
    if sample_grid_path.exists():
        import shutil
        shutil.copy(sample_grid_path, save_dir / 'sample_grid.png')
    
    # Save class stats
    stats = df.groupby(['split', 'label']).size().unstack(fill_value=0)
    stats.to_csv(save_dir / 'class_distribution.csv')
    
    logger.info(f"EDA completed. Results saved to {save_dir}")


if __name__ == "__main__":
    import yaml
    from utils import setup_logging, set_seed, get_device, create_directories
    from dataset import load_and_split_data, create_data_loaders
    from models import create_model, load_checkpoint
    
    setup_logging(Path("test_logs"), "INFO", True)
    set_seed(42)
    
    with open("../configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    dirs = create_directories(config)
    
    # Run EDA
    run_eda(config, dirs['eda'])
    
    print("Evaluation module test passed!")