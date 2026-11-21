#!/usr/bin/env python
"""
Zero-retraining ensemble & post-hoc logit adjustment script.

Loads saved checkpoints (resnet50d + efficientnet_b0), runs inference on validation/test sets,
grid-searches ensemble weights and temperature-scaled prior adjustment on validation set,
applies best params to test set, and saves comprehensive metrics.
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    classification_report, roc_auc_score, average_precision_score,
)

import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils import load_config, set_seed, get_device, create_directories, validate_config, get_class_names
from dataset import load_and_split_data, create_data_loaders
from models import create_model, load_checkpoint


def setup_logging(log_dir: Path, log_level: str = "INFO") -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ensemble_eval")
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_format = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    file_handler = logging.FileHandler(log_dir / 'ensemble_eval.log')
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_format = logging.Formatter('%(asctime)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger


def save_json(data: Dict[str, Any], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4, default=str)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str]
) -> Dict[str, Any]:
    """Compute comprehensive evaluation metrics."""
    n_classes = len(class_names)
    labels = list(range(n_classes))
    
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'macro_precision': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'macro_recall': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
    }
    
    # Per-class metrics
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0, labels=labels)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0, labels=labels)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0, labels=labels)
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    metrics['per_class'] = {}
    for i, name in enumerate(class_names):
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
    
    # ROC-AUC
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
    metrics['confusion_matrix'] = cm.tolist()
    cm_norm = cm.astype('float')
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    metrics['confusion_matrix_normalized'] = (cm_norm / row_sums).tolist()
    
    metrics['classification_report'] = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0, labels=labels
    )
    
    return metrics


@torch.no_grad()
def get_logits_and_labels(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference and return raw logits and labels."""
    model.eval()
    all_logits = []
    all_labels = []
    
    for batch in loader:
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        
        logits = model(images)
        
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    
    return np.concatenate(all_logits), np.concatenate(all_labels)


def apply_logit_adjustment(
    logits: np.ndarray,
    class_priors: np.ndarray,
    tau: float
) -> np.ndarray:
    """Apply temperature-scaled prior adjustment: logits - tau * log(priors)."""
    # logits shape: (N, C), class_priors shape: (C,)
    log_priors = np.log(class_priors + 1e-12)  # add small epsilon for numerical stability
    adjusted = logits - tau * log_priors
    return adjusted


def get_predictions(logits: np.ndarray) -> np.ndarray:
    """Get class predictions from logits."""
    return logits.argmax(axis=1)


def get_probabilities(logits: np.ndarray) -> np.ndarray:
    """Get class probabilities from logits using softmax."""
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def grid_search_ensemble(
    logits_resnet: np.ndarray,
    logits_effnet: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    class_priors: np.ndarray,
    ensemble_weights: List[float],
    temperatures: List[float]
) -> Tuple[float, float, Dict[str, float]]:
    """Grid search over ensemble weight and temperature on validation set."""
    
    best_macro_f1 = -1.0
    best_w = 0.5
    best_tau = 0.0
    best_metrics = {}
    
    print("\nGrid search on validation set:")
    print(f"  Ensemble weights: {ensemble_weights}")
    print(f"  Temperatures: {temperatures}")
    
    for w in ensemble_weights:
        logits_ens = w * logits_resnet + (1 - w) * logits_effnet
        
        for tau in temperatures:
            logits_adj = apply_logit_adjustment(logits_ens, class_priors, tau)
            preds = get_predictions(logits_adj)
            probs = get_probabilities(logits_adj)
            
            metrics = compute_metrics(labels, preds, probs, class_names)
            
            if metrics['macro_f1'] > best_macro_f1:
                best_macro_f1 = metrics['macro_f1']
                best_w = w
                best_tau = tau
                best_metrics = metrics
                print(f"  New best: w={w:.2f}, tau={tau:.2f}, Macro F1={metrics['macro_f1']:.4f}")
    
    print(f"\nBest validation params: w={best_w:.2f}, tau={best_tau:.2f}, Macro F1={best_macro_f1:.4f}")
    
    return best_w, best_tau, best_metrics


def main():
    parser = argparse.ArgumentParser(description="Zero-retraining ensemble evaluation")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--skip-tests", action="store_true", help="Skip validation tests")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    config = validate_config(config)
    config['experiment']['num_workers'] = 0  # Avoid multiprocessing issues
    
    # Setup
    dirs = create_directories(config)
    logger = setup_logging(dirs['test_logs'], config['output']['log_level'])
    
    set_seed(config['experiment']['seed'])
    device = get_device(config['experiment']['device'])
    
    class_names = get_class_names(config)
    n_classes = len(class_names)
    
    # Load data
    logger.info("Loading data...")
    train_df, val_df, test_df, label_encoder = load_and_split_data(config)
    train_loader, val_loader, test_loader = create_data_loaders(config, train_df, val_df, test_df, label_encoder)
    
    # Compute class priors from training set
    class_counts = np.zeros(n_classes)
    for batch in train_loader:
        labels = batch['label']
        class_counts += np.bincount(labels.numpy(), minlength=n_classes)
    class_priors = class_counts / class_counts.sum()
    logger.info(f"Training class priors: {dict(zip(class_names, class_priors))}")
    
    # Load models
    logger.info("Loading models...")
    
    # EfficientNet-B0
    effnet_config = config['models'][0]
    effnet_model = create_model(config, effnet_config).to(device)
    effnet_ckpt = Path(config['output']['results_dir']) / 'checkpoints' / effnet_config['name'] / 'best_model.pt'
    if effnet_ckpt.exists():
        load_checkpoint(effnet_model, effnet_ckpt, device)
        logger.info(f"Loaded {effnet_config['name']} from {effnet_ckpt}")
    else:
        logger.error(f"Checkpoint not found: {effnet_ckpt}")
        return
    
    # ResNet50d
    resnet_config = config['models'][2]
    resnet_model = create_model(config, resnet_config).to(device)
    resnet_ckpt = Path(config['output']['results_dir']) / 'checkpoints' / resnet_config['name'] / 'best_model.pt'
    if resnet_ckpt.exists():
        load_checkpoint(resnet_model, resnet_ckpt, device)
        logger.info(f"Loaded {resnet_config['name']} from {resnet_ckpt}")
    else:
        logger.error(f"Checkpoint not found: {resnet_ckpt}")
        return
    
    # Get logits on validation set
    logger.info("Running inference on validation set...")
    val_logits_resnet, val_labels = get_logits_and_labels(resnet_model, val_loader, device)
    val_logits_effnet, _ = get_logits_and_labels(effnet_model, val_loader, device)
    
    # Get logits on test set
    logger.info("Running inference on test set...")
    test_logits_resnet, test_labels = get_logits_and_labels(resnet_model, test_loader, device)
    test_logits_effnet, _ = get_logits_and_labels(effnet_model, test_loader, device)
    
    # Grid search on validation set
    ensemble_weights = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
    temperatures = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
    
    best_w, best_tau, val_metrics = grid_search_ensemble(
        val_logits_resnet, val_logits_effnet, val_labels,
        class_names, class_priors, ensemble_weights, temperatures
    )
    
    # Apply best params to test set
    logger.info(f"\nApplying best params (w={best_w:.2f}, tau={best_tau:.2f}) to test set...")
    
    test_logits_ens = best_w * test_logits_resnet + (1 - best_w) * test_logits_effnet
    test_logits_adj = apply_logit_adjustment(test_logits_ens, class_priors, best_tau)
    
    test_preds = get_predictions(test_logits_adj)
    test_probs = get_probabilities(test_logits_adj)
    
    test_metrics = compute_metrics(test_labels, test_preds, test_probs, class_names)
    
    # Also evaluate individual models on test set for comparison
    logger.info("\nEvaluating individual models on test set...")
    
    # ResNet50d alone
    resnet_preds = get_predictions(test_logits_resnet)
    resnet_probs = get_probabilities(test_logits_resnet)
    resnet_metrics = compute_metrics(test_labels, resnet_preds, resnet_probs, class_names)
    
    # EfficientNet-B0 alone
    effnet_preds = get_predictions(test_logits_effnet)
    effnet_probs = get_probabilities(test_logits_effnet)
    effnet_metrics = compute_metrics(test_labels, effnet_preds, effnet_probs, class_names)
    
    # Raw ensemble (w=0.5, tau=0)
    raw_ens_logits = 0.5 * test_logits_resnet + 0.5 * test_logits_effnet
    raw_ens_preds = get_predictions(raw_ens_logits)
    raw_ens_probs = get_probabilities(raw_ens_logits)
    raw_ens_metrics = compute_metrics(test_labels, raw_ens_preds, raw_ens_probs, class_names)
    
    # Print comparison table
    print("\n" + "=" * 100)
    print("BEFORE vs AFTER COMPARISON TABLE")
    print("=" * 100)
    print(f"{'Model':<25} {'Acc':>8} {'BalAcc':>8} {'MacroF1':>8} {'ROC-AUC':>8} "
          f"{'AK_Rec':>8} {'SCC_Rec':>8} {'MEL_Rec':>8} {'DF_Rec':>8}")
    print("-" * 100)
    
    def fmt_recall(m, name):
        return f"{m['per_class'][name]['recall_sensitivity']:.4f}"
    
    for name, m in [
        ("EfficientNet-B0", effnet_metrics),
        ("ResNet50d", resnet_metrics),
        ("Ensemble (w=0.5)", raw_ens_metrics),
        (f"Ensemble+Adj (w={best_w:.1f}, tau={best_tau:.2f})", test_metrics),
    ]:
        print(f"{name:<25} {m['accuracy']:>8.4f} {m['balanced_accuracy']:>8.4f} "
              f"{m['macro_f1']:>8.4f} {m['roc_auc_ovr']:>8.4f} "
              f"{fmt_recall(m, 'AK'):>8} {fmt_recall(m, 'SCC'):>8} "
              f"{fmt_recall(m, 'MEL'):>8} {fmt_recall(m, 'DF'):>8}")
    
    print("=" * 100)
    
    # Save comprehensive results
    results = {
        'best_ensemble_weight': float(best_w),
        'best_temperature': float(best_tau),
        'class_priors': class_priors.tolist(),
        'class_names': class_names,
        'validation_metrics': val_metrics,
        'test_metrics': {
            'efficientnet_b0': effnet_metrics,
            'resnet50d': resnet_metrics,
            'ensemble_raw': raw_ens_metrics,
            'ensemble_adjusted': test_metrics,
        },
        'best_params': {
            'ensemble_weight': float(best_w),
            'temperature': float(best_tau),
        }
    }
    
    save_json(results, Path(config['output']['results_dir']) / 'metrics' / 'ensemble_report.json')
    logger.info(f"Saved ensemble report to {Path(config['output']['results_dir']) / 'metrics' / 'ensemble_report.json'}")
    
    # Save comparison CSV
    import csv
    csv_path = Path(config['output']['results_dir']) / 'metrics' / 'ensemble_comparison.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Accuracy', 'Balanced_Accuracy', 'Macro_F1', 'ROC_AUC_OVR',
                        'AK_Recall', 'SCC_Recall', 'MEL_Recall', 'DF_Recall',
                        'AK_Precision', 'SCC_Precision', 'MEL_Precision', 'DF_Precision'])
        
        for name, m in [
            ("EfficientNet-B0", effnet_metrics),
            ("ResNet50d", resnet_metrics),
            ("Ensemble_Raw", raw_ens_metrics),
            ("Ensemble_Adjusted", test_metrics),
        ]:
            writer.writerow([
                name,
                f"{m['accuracy']:.4f}",
                f"{m['balanced_accuracy']:.4f}",
                f"{m['macro_f1']:.4f}",
                f"{m['roc_auc_ovr']:.4f}",
                f"{m['per_class']['AK']['recall_sensitivity']:.4f}",
                f"{m['per_class']['SCC']['recall_sensitivity']:.4f}",
                f"{m['per_class']['MEL']['recall_sensitivity']:.4f}",
                f"{m['per_class']['DF']['recall_sensitivity']:.4f}",
                f"{m['per_class']['AK']['precision']:.4f}",
                f"{m['per_class']['SCC']['precision']:.4f}",
                f"{m['per_class']['MEL']['precision']:.4f}",
                f"{m['per_class']['DF']['precision']:.4f}",
            ])
    
    logger.info(f"Saved comparison CSV to {csv_path}")
    
    logger.info("\nEnsemble evaluation completed successfully!")


if __name__ == "__main__":
    main()