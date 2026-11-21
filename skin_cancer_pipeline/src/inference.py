"""
Production-ready inference module for skin cancer classification ensemble.

Provides SkinCancerEnsemblePredictor class that:
- Loads trained ResNet50d and EfficientNet-B0 checkpoints
- Applies validated ensemble weights (w=0.50) and prior adjustment (tau=0.25)
- Performs preprocessing, inference, and Grad-CAM visualization
"""

import base64
import io
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageEnhance

from src.models import create_model, load_checkpoint, get_last_conv_layer
from src.utils import get_class_names


logger = logging.getLogger("skin_cancer_inference")


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


class SkinCancerEnsemblePredictor:
    """
    Production-ready ensemble predictor for skin cancer classification.

    Combines ResNet50d and EfficientNet-B0 with validated ensemble weights
    and post-hoc prior adjustment.
    """

    def __init__(
        self,
        resnet50d_checkpoint: str,
        efficientnet_b0_checkpoint: str,
        label_encoder_path: str,
        ensemble_weight: float = 0.5,
        temperature: float = 0.25,
        device: Optional[str] = None,
        image_size: int = 224,
    ):
        """
        Initialize ensemble predictor.

        Args:
            resnet50d_checkpoint: Path to ResNet50d checkpoint
            efficientnet_b0_checkpoint: Path to EfficientNet-B0 checkpoint
            label_encoder_path: Path to label encoder JSON
            ensemble_weight: Weight for ResNet50d (EfficientNet gets 1-weight)
            temperature: Temperature for prior adjustment
            device: Device to run on ('cuda' or 'cpu')
            image_size: Input image size
        """
        self.ensemble_weight = ensemble_weight
        self.temperature = temperature
        self.image_size = image_size

        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # Load class names and priors
        self.class_names = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']
        self.num_classes = len(self.class_names)

        # Load class priors from ensemble report
        self.class_priors = np.array([
            0.035735410094637224,  # AK
            0.12746451104100948,   # BCC
            0.10415023659305994,   # BKL
            0.009118690851735016,  # DF
            0.18202878548895898,   # MEL
            0.5064570189274448,    # NV
            0.025877365930599368,  # SCC
            0.009167981072555205,  # VASC
        ], dtype=np.float32)

        # Load label encoder
        with open(label_encoder_path, 'r') as f:
            label_mapping = json.load(f)
        self.label_to_idx = label_mapping
        self.idx_to_label = {v: k for k, v in label_mapping.items()}

        # Image preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        # Load models
        self._load_models(resnet50d_checkpoint, efficientnet_b0_checkpoint)

        # Setup Grad-CAM for both models
        self._setup_gradcam()

        logger.info("Ensemble predictor initialized successfully")

    def _load_models(self, resnet50d_checkpoint: str, efficientnet_b0_checkpoint: str):
        """Load both model checkpoints."""
        # Build a minimal config with training and data sections for create_model
        base_config = {
            'training': {'dropout': 0.3},
            'data': {'num_classes': self.num_classes}
        }
        
        # Load ResNet50d
        logger.info("Loading ResNet50d...")
        self.resnet_model = create_model(
            config=base_config,
            model_config={'timm_name': 'resnet50d', 'name': 'resnet50d', 'pretrained': False}
        ).to(self.device)
        load_checkpoint(self.resnet_model, resnet50d_checkpoint, self.device)
        self.resnet_model.eval()

        # Load EfficientNet-B0
        logger.info("Loading EfficientNet-B0...")
        self.effnet_model = create_model(
            config=base_config,
            model_config={'timm_name': 'efficientnet_b0', 'name': 'efficientnet_b0', 'pretrained': False}
        ).to(self.device)
        load_checkpoint(self.effnet_model, efficientnet_b0_checkpoint, self.device)
        self.effnet_model.eval()

        logger.info("Both models loaded successfully")

    def _setup_gradcam(self):
        """Setup Grad-CAM for both models."""
        # Get target layers
        resnet_target = get_last_conv_layer(self.resnet_model, 'resnet50d')
        effnet_target = get_last_conv_layer(self.effnet_model, 'efficientnet_b0')

        self.resnet_gradcam = GradCAM(self.resnet_model, resnet_target)
        self.effnet_gradcam = GradCAM(self.effnet_model, effnet_target)

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess PIL image for model input.

        Args:
            image: PIL Image

        Returns:
            Tensor of shape (1, 3, H, W)
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')

        tensor = self.transform(image)
        return tensor.unsqueeze(0).to(self.device)

    def predict(
        self,
        image: Image.Image,
        return_gradcam: bool = False,
        gradcam_target_class: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run ensemble inference on a single image.

        Args:
            image: PIL Image
            return_gradcam: Whether to generate Grad-CAM heatmaps
            gradcam_target_class: Target class for Grad-CAM (None = predicted class)

        Returns:
            Dictionary with predictions, probabilities, and optional Grad-CAM
        """
        # Preprocess
        input_tensor = self.preprocess_image(image)

        # Run inference on both models
        with torch.no_grad():
            resnet_logits = self.resnet_model(input_tensor)
            effnet_logits = self.effnet_model(input_tensor)

        # Ensemble logits
        ensemble_logits = (
            self.ensemble_weight * resnet_logits +
            (1 - self.ensemble_weight) * effnet_logits
        )

        # Apply prior adjustment
        log_priors = torch.log(torch.tensor(self.class_priors, device=self.device) + 1e-12)
        adjusted_logits = ensemble_logits - self.temperature * log_priors

        # Get probabilities and predictions
        probs = F.softmax(adjusted_logits, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        pred_label = self.class_names[pred_idx]
        confidence = probs[0, pred_idx].item()

        # Individual model predictions
        resnet_probs = F.softmax(resnet_logits, dim=1)
        effnet_probs = F.softmax(effnet_logits, dim=1)
        resnet_pred = resnet_probs.argmax(dim=1).item()
        effnet_pred = effnet_probs.argmax(dim=1).item()

        # Build result
        result = {
            'predicted_class': pred_label,
            'predicted_class_idx': pred_idx,
            'confidence': confidence,
            'all_probabilities': {
                name: probs[0, i].item() for i, name in enumerate(self.class_names)
            },
            'individual_models': {
                'resnet50d': {
                    'predicted_class': self.class_names[resnet_pred],
                    'probabilities': {
                        name: resnet_probs[0, i].item() for i, name in enumerate(self.class_names)
                    }
                },
                'efficientnet_b0': {
                    'predicted_class': self.class_names[effnet_pred],
                    'probabilities': {
                        name: effnet_probs[0, i].item() for i, name in enumerate(self.class_names)
                    }
                }
            }
        }

        # Generate Grad-CAM if requested
        if return_gradcam:
            target_class = gradcam_target_class if gradcam_target_class is not None else pred_idx
            result['gradcam'] = self.generate_gradcam(input_tensor, target_class)
            # Generate compact base64 overlay for Swagger/UI (JPEG 384x192, quality=75)
            result['gradcam_overlay_base64'] = self._generate_gradcam_overlay_base64(
                image, result['gradcam'], target_class
            )

        return result

    def generate_gradcam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate Grad-CAM heatmaps from both models.

        Args:
            input_tensor: Input tensor (1, 3, H, W)
            target_class: Target class index

        Returns:
            Dictionary with Grad-CAM heatmaps from both models
        """
        resnet_cam = self.resnet_gradcam(input_tensor, target_class)
        effnet_cam = self.effnet_gradcam(input_tensor, target_class)

        return {
            'resnet50d': resnet_cam,
            'efficientnet_b0': effnet_cam,
            'ensemble': (resnet_cam + effnet_cam) / 2  # Average for visualization
        }

    def _generate_gradcam_overlay_base64(
        self,
        original_image: Image.Image,
        gradcam_data: Dict[str, np.ndarray],
        target_class: Optional[int] = None
    ) -> str:
        """
        Generate a compact base64-encoded Grad-CAM overlay image (JPEG, 384x192, quality=75).

        Creates a side-by-side visualization: original image | Grad-CAM heatmap overlay.
        Quality is reduced adaptively (and size only as a last resort) so the returned
        data URI stays under 25 KB, keeping Swagger UI (/docs) responsive when it
        renders the JSON response of POST /predict.

        Args:
            original_image: Original PIL Image
            gradcam_data: Dictionary with 'resnet50d', 'efficientnet_b0', 'ensemble' heatmaps
            target_class: Target class index (for display)

        Returns:
            Base64-encoded JPEG string (data URI format: data:image/jpeg;base64,...)
        """
        # Use ensemble heatmap for overlay
        cam = np.asarray(gradcam_data['ensemble'], dtype=np.float32)
        if cam.size == 0:
            raise ValueError("Empty Grad-CAM heatmap")
        # Normalize defensively (heatmaps are nominally 0-1)
        cam_max = float(cam.max())
        if cam_max > 0:
            cam = cam / cam_max
        cam = np.clip(cam, 0.0, 1.0)

        # Convert original image to RGB
        if original_image.mode != 'RGB':
            original_image = original_image.convert('RGB')

        # Resize original to 192x192
        img_resized = original_image.resize((192, 192), Image.BILINEAR)

        # Convert CAM to PIL Image (grayscale)
        cam_normalized = (cam * 255).astype(np.uint8)
        cam_img = Image.fromarray(cam_normalized, mode='L')

        # Resize CAM to 192x192
        cam_img_resized = cam_img.resize((192, 192), Image.BILINEAR)

        # Apply colormap (JET-like) to CAM
        cam_colored = self._apply_jet_colormap(cam_img_resized)

        # Create side-by-side composite (384x192 total)
        composite = Image.new('RGB', (384, 192))
        composite.paste(img_resized, (0, 0))
        composite.paste(cam_colored, (192, 0))

        # Encode as JPEG with quality=75 (payload stays well under 25 KB).
        # Fall back to lower quality / smaller size for unusually busy images so
        # Swagger UI never has to render a huge base64 string.
        DATA_URI_PREFIX = "data:image/jpeg;base64,"
        MAX_B64_CHARS = 25_000

        def encode(image: Image.Image, quality: int) -> str:
            buf = io.BytesIO()
            image.save(buf, format='JPEG', quality=quality, optimize=True)
            return base64.b64encode(buf.getvalue()).decode('utf-8')

        b64_str = encode(composite, 75)
        for quality in (65, 55, 45, 35):
            if len(b64_str) + len(DATA_URI_PREFIX) < MAX_B64_CHARS:
                break
            b64_str = encode(composite, quality)

        if len(b64_str) + len(DATA_URI_PREFIX) >= MAX_B64_CHARS:
            # Last-resort downscale (keeps the aspect ratio of the composite)
            for width, height, quality in ((320, 160, 65), (256, 128, 60)):
                b64_str = encode(
                    composite.resize((width, height), Image.BILINEAR),
                    quality,
                )
                if len(b64_str) + len(DATA_URI_PREFIX) < MAX_B64_CHARS:
                    break

        # Return as data URI
        return f"{DATA_URI_PREFIX}{b64_str}"

    def _apply_jet_colormap(self, gray_img: Image.Image) -> Image.Image:
        """
        Apply a JET-like colormap to a grayscale PIL image.

        Args:
            gray_img: Grayscale PIL Image (mode 'L')

        Returns:
            RGB PIL Image with JET colormap applied
        """
        # Convert to numpy array
        arr = np.array(gray_img)

        # Normalize to 0-1
        if arr.max() > 0:
            arr = arr / arr.max()

        # JET colormap approximation
        r = np.clip(1.5 - np.abs(4 * arr - 3), 0, 1)
        g = np.clip(1.5 - np.abs(4 * arr - 2), 0, 1)
        b = np.clip(1.5 - np.abs(4 * arr - 1), 0, 1)

        # Stack into RGB
        rgb = np.stack([r, g, b], axis=-1)
        rgb = (rgb * 255).astype(np.uint8)

        return Image.fromarray(rgb, mode='RGB')

    def predict_batch(
        self,
        images: List[Image.Image],
        return_gradcam: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Run batch inference (processes images sequentially for memory efficiency).

        Args:
            images: List of PIL Images
            return_gradcam: Whether to generate Grad-CAM heatmaps

        Returns:
            List of prediction results
        """
        results = []
        for image in images:
            result = self.predict(image, return_gradcam=return_gradcam)
            results.append(result)
        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded models."""
        return {
            'ensemble_weight': self.ensemble_weight,
            'temperature': self.temperature,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'class_priors': self.class_priors.tolist(),
            'device': str(self.device),
            'image_size': self.image_size,
            'models': {
                'resnet50d': {
                    'architecture': 'resnet50d',
                    'parameters': sum(p.numel() for p in self.resnet_model.parameters())
                },
                'efficientnet_b0': {
                    'architecture': 'efficientnet_b0',
                    'parameters': sum(p.numel() for p in self.effnet_model.parameters())
                }
            }
        }


def create_predictor_from_config(
    config_path: str,
    results_dir: str = "results"
) -> SkinCancerEnsemblePredictor:
    """
    Create predictor from config and results directory.

    Args:
        config_path: Path to config.yaml
        results_dir: Directory containing checkpoints and ensemble report

    Returns:
        Initialized SkinCancerEnsemblePredictor
    """
    import yaml
    from utils import load_config, validate_config

    config = load_config(config_path)
    config = validate_config(config)

    results_dir = Path(results_dir)
    ensemble_report = results_dir / 'metrics' / 'ensemble_report.json'

    if ensemble_report.exists():
        with open(ensemble_report, 'r') as f:
            report = json.load(f)
        ensemble_weight = report['best_params']['ensemble_weight']
        temperature = report['best_params']['temperature']
    else:
        # Fallback to validated defaults
        ensemble_weight = 0.5
        temperature = 0.25

    resnet50d_checkpoint = results_dir / 'checkpoints' / 'resnet50d' / 'best_model.pt'
    efficientnet_b0_checkpoint = results_dir / 'checkpoints' / 'efficientnet_b0' / 'best_model.pt'
    label_encoder_path = results_dir / 'checkpoints' / 'resnet50d' / 'label_encoder.json'

    return SkinCancerEnsemblePredictor(
        resnet50d_checkpoint=str(resnet50d_checkpoint),
        efficientnet_b0_checkpoint=str(efficientnet_b0_checkpoint),
        label_encoder_path=str(label_encoder_path),
        ensemble_weight=ensemble_weight,
        temperature=temperature,
    )


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    # Quick test
    predictor = create_predictor_from_config(
        config_path="configs/config.yaml",
        results_dir="results"
    )

    info = predictor.get_model_info()
    print(f"Predictor initialized on {info['device']}")
    print(f"Models: {list(info['models'].keys())}")
    print(f"Classes: {info['class_names']}")
    print(f"Ensemble weight: {info['ensemble_weight']}")
    print(f"Temperature: {info['temperature']}")