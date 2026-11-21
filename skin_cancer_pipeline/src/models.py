"""
Model definitions for skin cancer classification.
Wraps timm models with custom heads for 8-class classification.
"""

import logging
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
import timm
from torch.nn import functional as F

from src.utils import get_class_names, print_model_summary


logger = logging.getLogger("skin_cancer_pipeline")


class SkinCancerModel(nn.Module):
    """
    Wrapper for timm models with custom classification head.
    """
    
    def __init__(
        self,
        model_name: str,
        num_classes: int = 8,
        pretrained: bool = True,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        hidden_dim: int = 512
    ):
        """
        Initialize model.
        
        Args:
            model_name: timm model name (e.g., 'efficientnet_b0', 'convnext_tiny', 'resnet50d')
            num_classes: Number of output classes
            pretrained: Whether to use ImageNet pretrained weights
            drop_rate: Dropout rate for classifier head
            drop_path_rate: Stochastic depth rate (for ConvNeXt, etc.)
            hidden_dim: Hidden dimension for custom head
        """
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Create backbone using timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier head
            global_pool='',  # Remove global pooling
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy)
            if isinstance(features, (list, tuple)):
                features = features[-1]
            if features.dim() == 4:
                # Global average pooling
                features = F.adaptive_avg_pool2d(features, 1).flatten(1)
            self.feature_dim = features.shape[1]
        
        logger.info(f"Backbone feature dimension: {self.feature_dim}")
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Initialize classifier weights
        self._init_classifier()
    
    def _init_classifier(self):
        """Initialize classifier weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
        
        Returns:
            Logits of shape (B, num_classes)
        """
        features = self.backbone(x)
        
        # Handle different backbone output formats
        if isinstance(features, (list, tuple)):
            features = features[-1]
        
        # Global average pooling if needed
        if features.dim() == 4:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        elif features.dim() == 3:
            # For ViT-like models: (B, seq_len, dim) -> (B, dim)
            features = features.mean(dim=1)
        
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before classifier.
        
        Args:
            x: Input tensor
        
        Returns:
            Feature tensor
        """
        features = self.backbone(x)
        if isinstance(features, (list, tuple)):
            features = features[-1]
        if features.dim() == 4:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        elif features.dim() == 3:
            features = features.mean(dim=1)
        return features


def create_model(config: Dict[str, Any], model_config: Dict[str, Any]) -> SkinCancerModel:
    """
    Create model from configuration.
    
    Args:
        config: Main configuration dictionary
        model_config: Model-specific configuration
    
    Returns:
        SkinCancerModel instance
    """
    num_classes = len(get_class_names(config))
    
    # Use safe defaults with .get() to avoid KeyError when config is minimal
    training_config = config.get('training', {})
    data_config = config.get('data', {})
    
    model = SkinCancerModel(
        model_name=model_config['timm_name'],
        num_classes=num_classes,
        pretrained=model_config.get('pretrained', True),
        drop_rate=training_config.get('dropout', 0.3),
        drop_path_rate=0.1 if 'convnext' in model_config['timm_name'] else 0.0,
        hidden_dim=512
    )
    
    logger.info(f"Created model: {model_config['name']} ({model_config['timm_name']})")
    print_model_summary(model)
    
    return model


def get_model_list(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Get list of models to train from config."""
    return config['models']


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load on
        optimizer: Optional optimizer to restore
        scheduler: Optional scheduler to restore
    
    Returns:
        Checkpoint metadata dictionary
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    metadata = {
        'epoch': checkpoint.get('epoch', 0),
        'best_metric': checkpoint.get('best_metric', 0),
        'config': checkpoint.get('config', {})
    }
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    logger.info(f"Loaded checkpoint from epoch {metadata['epoch']}, best metric: {metadata['best_metric']:.4f}")
    
    return metadata


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    best_metric: float,
    config: Dict[str, Any],
    save_path: Path,
    is_best: bool = False
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch
        best_metric: Best validation metric
        config: Configuration dictionary
        save_path: Path to save checkpoint
        is_best: Whether this is the best model
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_metric': best_metric,
        'config': config
    }
    
    torch.save(checkpoint, save_path)
    
    if is_best:
        best_path = save_path.parent / f"best_{save_path.name}"
        torch.save(checkpoint, best_path)
        logger.info(f"Saved best model to {best_path}")
    
    logger.info(f"Saved checkpoint to {save_path}")


def get_last_conv_layer(model: nn.Module, model_name: str) -> nn.Module:
    """
    Get the last convolutional layer for Grad-CAM.
    
    Args:
        model: Model instance
        model_name: Name of the model architecture
    
    Returns:
        Last convolutional layer module
    """
    backbone = model.backbone
    
    # Try to find the last conv layer automatically
    if 'efficientnet' in model_name.lower():
        # EfficientNet: blocks[-1] -> conv_head
        return backbone.conv_head
    elif 'convnext' in model_name.lower():
        # ConvNeXt: stages[-1] -> blocks[-1] -> conv_dw (depthwise conv)
        # timm uses conv_dw for depthwise convolution in ConvNeXt blocks
        last_block = backbone.stages[-1].blocks[-1]
        if hasattr(last_block, 'conv_dw'):
            return last_block.conv_dw
        elif hasattr(last_block, 'dwconv'):
            return last_block.dwconv
        else:
            # Fallback: find the depthwise conv in the last block
            for name, module in last_block.named_modules():
                if isinstance(module, nn.Conv2d) and module.groups == module.in_channels:
                    return module
            raise ValueError(f"Could not find depthwise conv in ConvNeXt last block")
    elif 'resnet' in model_name.lower() or 'resnetd' in model_name.lower():
        # ResNet: layer4[-1] -> conv3
        return backbone.layer4[-1].conv3
    elif 'vit' in model_name.lower() or 'deit' in model_name.lower():
        # ViT: blocks[-1] -> attn or mlp (not conv, use generic fallback)
        pass
    else:
        pass
    
    # Generic: find last Conv2d
    conv_layers = []
    for name, module in backbone.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append((name, module))
    
    if conv_layers:
        logger.info(f"Auto-detected last conv layer: {conv_layers[-1][0]}")
        return conv_layers[-1][1]
    else:
        raise ValueError(f"Could not find Conv2d layer in {model_name}")


if __name__ == "__main__":
    import yaml
    from utils import setup_logging, set_seed, get_device
    
    setup_logging(Path("test_logs"), "INFO", True)
    set_seed(42)
    
    with open("../configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    device = get_device(config['experiment']['device'])
    
    for model_config in config['models']:
        print(f"\n{'='*60}")
        print(f"Testing model: {model_config['name']}")
        print(f"{'='*60}")
        
        model = create_model(config, model_config).to(device)
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224).to(device)
        with torch.no_grad():
            logits = model(x)
        
        print(f"Output shape: {logits.shape}")
        assert logits.shape == (2, 8), f"Expected (2, 8), got {logits.shape}"
        
        # Test Grad-CAM layer detection
        try:
            last_conv = get_last_conv_layer(model, model_config['timm_name'])
            print(f"Last conv layer: {last_conv}")
        except Exception as e:
            print(f"Grad-CAM layer detection failed: {e}")
        
        print("Model test passed!")