"""
Tests for evaluation metrics and artifact generation on dummy predictions.
"""

import pytest
import numpy as np
import torch
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluate import (
    compute_metrics, plot_confusion_matrix, plot_roc_curves,
    plot_pr_curves, plot_training_curves, plot_per_class_metrics,
    GradCAM
)
from utils import get_class_names


@pytest.fixture
def class_names():
    return ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']


@pytest.fixture
def dummy_predictions(class_names):
    """Generate dummy predictions for testing."""
    np.random.seed(42)
    n_samples = 1000
    n_classes = len(class_names)
    
    # Create imbalanced ground truth (similar to real data)
    class_probs = np.array([0.03, 0.13, 0.10, 0.01, 0.18, 0.51, 0.02, 0.02])  # sum = 1.0
    class_probs = class_probs / class_probs.sum()  # Ensure exact sum to 1
    y_true = np.random.choice(n_classes, n_samples, p=class_probs)
    
    # Create predictions with some noise (80% accuracy)
    y_pred = y_true.copy()
    n_errors = int(n_samples * 0.2)
    error_indices = np.random.choice(n_samples, n_errors, replace=False)
    y_pred[error_indices] = np.random.choice(n_classes, n_errors)
    
    # Create probabilities (higher for correct class)
    y_prob = np.random.dirichlet(np.ones(n_classes), n_samples)
    # Boost probability for true class
    for i in range(n_samples):
        y_prob[i, y_true[i]] += 0.5
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    
    return y_true, y_pred, y_prob


class TestComputeMetrics:
    """Tests for metric computation."""
    
    def test_compute_metrics_keys(self, dummy_predictions, class_names):
        y_true, y_pred, y_prob = dummy_predictions
        metrics = compute_metrics(y_true, y_pred, y_prob, class_names)
        
        # Check all expected keys exist
        expected_keys = [
            'accuracy', 'balanced_accuracy', 'macro_f1', 'macro_precision',
            'macro_recall', 'per_class', 'roc_auc_ovr', 'roc_auc_ovo',
            'per_class_roc_auc', 'pr_auc_macro', 'confusion_matrix',
            'confusion_matrix_normalized', 'classification_report'
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"
    
    def test_metrics_range(self, dummy_predictions, class_names):
        y_true, y_pred, y_prob = dummy_predictions
        metrics = compute_metrics(y_true, y_pred, y_prob, class_names)
        
        # All metrics should be in [0, 1]
        for key in ['accuracy', 'balanced_accuracy', 'macro_f1', 'macro_precision', 'macro_recall',
                    'roc_auc_ovr', 'roc_auc_ovo', 'pr_auc_macro']:
            assert 0 <= metrics[key] <= 1, f"{key} = {metrics[key]} out of range"
    
    def test_per_class_metrics(self, dummy_predictions, class_names):
        y_true, y_pred, y_prob = dummy_predictions
        metrics = compute_metrics(y_true, y_pred, y_prob, class_names)
        
        for name in class_names:
            assert name in metrics['per_class']
            pc = metrics['per_class'][name]
            for metric in ['precision', 'recall_sensitivity', 'specificity', 'f1']:
                assert 0 <= pc[metric] <= 1, f"{name}.{metric} = {pc[metric]} out of range"
            assert pc['support'] >= 0
    
    def test_confusion_matrix_shape(self, dummy_predictions, class_names):
        y_true, y_pred, y_prob = dummy_predictions
        metrics = compute_metrics(y_true, y_pred, y_prob, class_names)
        
        cm = metrics['confusion_matrix']
        assert len(cm) == len(class_names)
        assert len(cm[0]) == len(class_names)
        assert sum(sum(row) for row in cm) == len(y_true)
    
    def test_confusion_matrix_normalized(self, dummy_predictions, class_names):
        y_true, y_pred, y_prob = dummy_predictions
        metrics = compute_metrics(y_true, y_pred, y_prob, class_names)
        
        cm_norm = metrics['confusion_matrix_normalized']
        # Each row should sum to 1 (or 0 if no samples)
        for row in cm_norm:
            row_sum = sum(row)
            assert abs(row_sum - 1.0) < 1e-6 or row_sum == 0, f"Row sum = {row_sum}"
    
    def test_perfect_predictions(self, class_names):
        """Test metrics with perfect predictions."""
        n_classes = len(class_names)
        y_true = np.array([0, 1, 2, 3, 4, 5, 6, 7] * 10)
        y_pred = y_true.copy()
        y_prob = np.zeros((len(y_true), n_classes))
        for i, t in enumerate(y_true):
            y_prob[i, t] = 1.0
        
        metrics = compute_metrics(y_true, y_pred, y_prob, class_names)
        
        assert metrics['accuracy'] == 1.0
        assert metrics['balanced_accuracy'] == 1.0
        assert metrics['macro_f1'] == 1.0
        assert metrics['roc_auc_ovr'] == 1.0
        assert metrics['roc_auc_ovo'] == 1.0
        assert metrics['pr_auc_macro'] == 1.0
    
    def test_random_predictions(self, class_names):
        """Test metrics with random predictions."""
        np.random.seed(123)
        n_samples = 500
        n_classes = len(class_names)
        
        y_true = np.random.randint(0, n_classes, n_samples)
        y_pred = np.random.randint(0, n_classes, n_samples)
        y_prob = np.random.dirichlet(np.ones(n_classes), n_samples)
        
        metrics = compute_metrics(y_true, y_pred, y_prob, class_names)
        
        # Should not crash and produce valid values
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['macro_f1'] <= 1


class TestPlottingFunctions:
    """Tests for plotting functions (check they run without error)."""
    
    def test_plot_confusion_matrix(self, dummy_predictions, class_names, tmp_path):
        y_true, y_pred, y_prob = dummy_predictions
        metrics = compute_metrics(y_true, y_pred, y_prob, class_names)
        
        save_path = tmp_path / "cm.png"
        plot_confusion_matrix(np.array(metrics['confusion_matrix']), class_names, save_path)
        assert save_path.exists()
    
    def test_plot_confusion_matrix_normalized(self, dummy_predictions, class_names, tmp_path):
        y_true, y_pred, y_prob = dummy_predictions
        metrics = compute_metrics(y_true, y_pred, y_prob, class_names)
        
        save_path = tmp_path / "cm_norm.png"
        plot_confusion_matrix(np.array(metrics['confusion_matrix']), class_names, save_path, normalize='true')
        assert save_path.exists()
    
    def test_plot_roc_curves(self, dummy_predictions, class_names, tmp_path):
        y_true, y_pred, y_prob = dummy_predictions
        
        save_path = tmp_path / "roc.png"
        plot_roc_curves(y_true, y_prob, class_names, save_path)
        assert save_path.exists()
    
    def test_plot_pr_curves(self, dummy_predictions, class_names, tmp_path):
        y_true, y_pred, y_prob = dummy_predictions
        
        save_path = tmp_path / "pr.png"
        plot_pr_curves(y_true, y_prob, class_names, save_path)
        assert save_path.exists()
    
    def test_plot_training_curves(self, tmp_path):
        history = {
            'train_loss': [0.5, 0.4, 0.3, 0.25, 0.2],
            'val_loss': [0.6, 0.5, 0.4, 0.35, 0.3],
            'train_acc': [0.7, 0.75, 0.8, 0.85, 0.88],
            'val_acc': [0.65, 0.7, 0.75, 0.8, 0.82],
            'train_macro_f1': [0.6, 0.65, 0.7, 0.75, 0.78],
            'val_macro_f1': [0.55, 0.6, 0.65, 0.7, 0.72],
            'train_bal_acc': [0.65, 0.7, 0.75, 0.8, 0.83],
            'val_bal_acc': [0.6, 0.65, 0.7, 0.75, 0.77],
            'lr': [1e-3, 1e-3, 5e-4, 5e-4, 1e-4]
        }
        
        save_path = tmp_path / "training.png"
        plot_training_curves(history, save_path)
        assert save_path.exists()
    
    def test_plot_per_class_metrics(self, dummy_predictions, class_names, tmp_path):
        y_true, y_pred, y_prob = dummy_predictions
        metrics = compute_metrics(y_true, y_pred, y_prob, class_names)
        
        save_path = tmp_path / "per_class.png"
        plot_per_class_metrics(metrics, class_names, save_path)
        assert save_path.exists()


class TestGradCAM:
    """Tests for Grad-CAM (requires a model)."""
    
    def test_gradcam_creation(self):
        """Test GradCAM can be instantiated with a simple model."""
        import torch.nn as nn
        
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(16, 8)
            
            def forward(self, x):
                x = self.conv(x)
                x = self.pool(x).flatten(1)
                return self.fc(x)
        
        model = SimpleModel()
        target_layer = model.conv
        
        gradcam = GradCAM(model, target_layer)
        assert gradcam is not None
    
    def test_gradcam_generate(self):
        import torch.nn as nn
        
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(16, 8)
            
            def forward(self, x):
                x = self.conv(x)
                x = self.pool(x).flatten(1)
                return self.fc(x)
        
        model = SimpleModel()
        model.eval()
        target_layer = model.conv
        
        gradcam = GradCAM(model, target_layer)
        
        x = torch.randn(1, 3, 224, 224, requires_grad=True)
        cam = gradcam.generate(x, target_class=0)
        
        assert cam.shape == (224, 224)
        assert cam.dtype == np.float32
        assert 0 <= cam.min() <= cam.max() <= 1


class TestEdgeCases:
    """Tests for edge cases in evaluation."""
    
    def test_single_class_present(self, class_names):
        """Test when only one class is present in predictions."""
        y_true = np.array([0, 0, 0, 1, 1, 1])  # Only classes 0 and 1
        y_pred = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([
            [0.9, 0.1, 0, 0, 0, 0, 0, 0],
            [0.8, 0.2, 0, 0, 0, 0, 0, 0],
            [0.7, 0.3, 0, 0, 0, 0, 0, 0],
            [0.1, 0.9, 0, 0, 0, 0, 0, 0],
            [0.2, 0.8, 0, 0, 0, 0, 0, 0],
            [0.3, 0.7, 0, 0, 0, 0, 0, 0],
        ])
        
        metrics = compute_metrics(y_true, y_pred, y_prob, class_names)
        assert metrics['accuracy'] == 1.0
    
    def test_empty_class_in_predictions(self, class_names):
        """Test when a class has no predictions."""
        y_true = np.array([0, 1, 2] * 10)
        y_pred = np.array([0, 1, 1] * 10)  # Class 2 never predicted
        y_prob = np.random.dirichlet(np.ones(8), 30)
        
        metrics = compute_metrics(y_true, y_pred, y_prob, class_names)
        # Should not crash, class 2 should have 0 precision/recall/f1
        assert metrics['per_class']['BKL']['f1'] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])