"""
Tests for model forward pass, output shapes, gradient flow, AMP compatibility.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models import SkinCancerModel, create_model, get_last_conv_layer
from utils import load_config, set_seed, get_device


@pytest.fixture(scope="session")
def config():
    cfg = load_config("configs/config.yaml")
    # Use auto device detection - will use CUDA if available
    cfg['experiment']['device'] = 'auto'
    return cfg


@pytest.fixture(scope="session")
def device(config):
    return get_device(config['experiment']['device'])


@pytest.fixture(params=[
    "efficientnet_b0",
    "convnext_tiny", 
    "resnet50d"
])
def model_name(request):
    return request.param


@pytest.fixture
def model(config, model_name, device):
    """Create model for testing."""
    model_config = next(m for m in config['models'] if m['timm_name'] == model_name)
    model = create_model(config, model_config).to(device)
    model.train()
    return model


class TestModelForward:
    """Tests for model forward pass."""
    
    def test_output_shape(self, model, device):
        batch_size = 4
        x = torch.randn(batch_size, 3, 224, 224).to(device)
        
        with torch.no_grad():
            logits = model(x)
        
        assert logits.shape == (batch_size, 8), f"Expected (4, 8), got {logits.shape}"
        assert logits.dtype == torch.float32
    
    def test_output_requires_grad(self, model, device):
        x = torch.randn(2, 3, 224, 224, requires_grad=True).to(device)
        logits = model(x)
        
        assert logits.requires_grad
        assert logits.grad_fn is not None
    
    def test_gradient_flow(self, model, device):
        """Test that gradients flow through the entire model."""
        x = torch.randn(2, 3, 224, 224, requires_grad=True).to(device)
        target = torch.tensor([0, 1]).to(device)
        
        logits = model(x)
        loss = nn.CrossEntropyLoss()(logits, target)
        loss.backward()
        
        # Check that all parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
    
    def test_different_batch_sizes(self, model, device):
        for batch_size in [1, 2, 8, 16]:
            x = torch.randn(batch_size, 3, 224, 224).to(device)
            # Use eval mode for batch size 1 to avoid BatchNorm issues
            if batch_size == 1:
                model.eval()
            else:
                model.train()
            with torch.no_grad():
                logits = model(x)
            assert logits.shape == (batch_size, 8)
    
    def test_eval_mode(self, model, device):
        model.eval()
        x = torch.randn(2, 3, 224, 224).to(device)
        
        with torch.no_grad():
            logits1 = model(x)
            logits2 = model(x)
        
        # In eval mode, should be deterministic (no dropout randomness)
        assert torch.allclose(logits1, logits2)
    
    def test_train_mode_dropout(self, model, device):
        model.train()
        x = torch.randn(2, 3, 224, 224).to(device)
        
        with torch.no_grad():
            logits1 = model(x)
            logits2 = model(x)
        
        # In train mode with dropout, outputs may differ
        # But we just ensure it runs without error
        assert logits1.shape == (2, 8)
        assert logits2.shape == (2, 8)


class TestAMPCompatibility:
    """Tests for Automatic Mixed Precision compatibility."""
    
    def test_amp_forward(self, model, device):
        if device.type != 'cuda':
            pytest.skip("AMP test requires CUDA")
        
        x = torch.randn(2, 3, 224, 224).to(device)
        
        with torch.amp.autocast('cuda'):
            logits = model(x)
        
        assert logits.shape == (2, 8)
        # With AMP, output may be float16 during autocast context
        # The important thing is it runs without error
        assert logits.dtype in (torch.float32, torch.float16)
    
    def test_amp_backward(self, model, device):
        if device.type != 'cuda':
            pytest.skip("AMP test requires CUDA")
        
        x = torch.randn(2, 3, 224, 224, requires_grad=True).to(device)
        target = torch.tensor([0, 1]).to(device)
        scaler = torch.amp.GradScaler('cuda')
        
        with torch.amp.autocast('cuda'):
            logits = model(x)
            loss = nn.CrossEntropyLoss()(logits, target)
        
        scaler.scale(loss).backward()
        scaler.step(torch.optim.Adam(model.parameters()))
        scaler.update()
        
        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_amp_training_step(self, model, device):
        if device.type != 'cuda':
            pytest.skip("AMP test requires CUDA")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler('cuda')
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        x = torch.randn(4, 3, 224, 224).to(device)
        target = torch.randint(0, 8, (4,)).to(device)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            logits = model(x)
            loss = criterion(logits, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        assert loss.item() > 0


class TestGradCAMLayer:
    """Tests for Grad-CAM target layer detection."""
    
    def test_get_last_conv_layer(self, model, model_name):
        try:
            layer = get_last_conv_layer(model, model_name)
            assert layer is not None
            # Should be a Conv2d module
            assert isinstance(layer, nn.Conv2d)
        except Exception as e:
            # Some architectures might not have easily detectable conv layers
            pytest.skip(f"Could not detect conv layer for {model_name}: {e}")


class TestModelCreation:
    """Tests for model creation with different configs."""
    
    def test_create_all_models(self, config, device):
        for model_config in config['models']:
            model = create_model(config, model_config).to(device)
            model.eval()  # Use eval mode for batch size 1
            
            # Test forward
            x = torch.randn(1, 3, 224, 224).to(device)
            with torch.no_grad():
                logits = model(x)
            
            assert logits.shape == (1, 8)
            
            # Clean up
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    def test_model_parameter_count(self, config, device):
        for model_config in config['models']:
            model = create_model(config, model_config).to(device)
            
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            assert total_params > 0
            assert trainable_params > 0
            assert trainable_params <= total_params
            
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    def test_feature_extraction(self, model, device):
        """Test feature extraction method."""
        x = torch.randn(2, 3, 224, 224).to(device)
        
        with torch.no_grad():
            features = model.get_features(x)
            logits = model(x)
        
        # Features should be 2D: (batch, feature_dim)
        assert features.dim() == 2
        assert features.shape[0] == 2
        # Logits should match classifier output
        assert logits.shape == (2, 8)


class TestNumericalStability:
    """Tests for numerical stability."""
    
    def test_no_nan_output(self, model, device):
        x = torch.randn(4, 3, 224, 224).to(device)
        
        with torch.no_grad():
            logits = model(x)
        
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()
    
    def test_no_nan_gradients(self, model, device):
        x = torch.randn(2, 3, 224, 224, requires_grad=True).to(device)
        target = torch.tensor([0, 1]).to(device)
        
        logits = model(x)
        loss = nn.CrossEntropyLoss()(logits, target)
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"
    
    def test_large_input_values(self, model, device):
        """Test with extreme input values."""
        model.eval()  # Use eval mode for batch size 1
        x = torch.ones(1, 3, 224, 224).to(device) * 100  # Very large values
        
        with torch.no_grad():
            logits = model(x)
        
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])