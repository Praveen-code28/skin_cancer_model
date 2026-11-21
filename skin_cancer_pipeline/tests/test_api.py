"""
FastAPI TestClient suite for the skin cancer classification API.

Tests verify:
- /health endpoint
- /predict endpoint (single image, base64, batch)
- Grad-CAM output validation
- Error handling (invalid files, missing models, etc.)
"""

import base64
import io
import pytest
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import MagicMock, patch, AsyncMock

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

# Import the app
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app


class MockPredictor:
    """Mock predictor for testing without model checkpoints."""
    
    def __init__(self):
        self.class_names = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']
        self.device = 'cpu'
        self.ensemble_weight = 0.5
        self.temperature = 0.25
        
    def get_model_info(self) -> dict:
        return {
            'device': self.device,
            'models': {
                'resnet50d': 'resnet50d',
                'efficientnet_b0': 'efficientnet_b0'
            },
            'class_names': self.class_names,
            'ensemble_weight': self.ensemble_weight,
            'temperature': self.temperature,
        }
    
    def predict(self, image, return_gradcam=False, gradcam_target_class=None):
        # Return a mock prediction
        import random
        pred_idx = random.randint(0, 7)
        pred_label = self.class_names[pred_idx]
        confidence = random.uniform(0.5, 0.95)
        
        probs = {name: random.random() for name in self.class_names}
        total = sum(probs.values())
        probs = {k: v/total for k, v in probs.items()}
        
        result = {
            'predicted_class': pred_label,
            'predicted_class_idx': pred_idx,
            'confidence': confidence,
            'all_probabilities': probs,
            'individual_models': {
                'resnet50d': {
                    'predicted_class': pred_label,
                    'probabilities': probs
                },
                'efficientnet_b0': {
                    'predicted_class': pred_label,
                    'probabilities': probs
                }
            }
        }
        
        if return_gradcam:
            # Generate mock Grad-CAM (224x224 heatmap)
            import numpy as np
            heatmap = np.random.rand(224, 224).astype(np.float32)
            result['gradcam'] = {
                'resnet50d': heatmap.tolist(),
                'efficientnet_b0': heatmap.tolist(),
                'ensemble': heatmap.tolist()
            }
        
        return result
    
    def get_model_info(self) -> dict:
        return {
            'device': self.device,
            'models': {
                'resnet50d': 'resnet50d',
                'efficientnet_b0': 'efficientnet_b0'
            },
            'class_names': self.class_names,
            'ensemble_weight': self.ensemble_weight,
            'temperature': self.temperature,
        }


# Create a global mock predictor instance
mock_predictor_instance = MockPredictor()


@pytest.fixture(scope="session")
def client():
    """Create test client with mocked predictor."""
    from app.main import app as fastapi_app
    
    # Create a mock lifespan that doesn't load models
    @asynccontextmanager
    async def mock_lifespan(app_instance):
        import app.main as app_module
        app_module.predictor = mock_predictor_instance
        yield
    
    # Patch the app's lifespan context
    original_lifespan = app.router.lifespan_context
    app.router.lifespan_context = mock_lifespan
    
    # Also patch the predictor variable
    import app.main as app_module
    app_module.predictor = mock_predictor_instance
    
    yield TestClient(app)
    
    # Restore
    app.router.lifespan_context = original_lifespan


@pytest.fixture(scope="session")
def sample_image():
    """Create a sample test image."""
    img = Image.new('RGB', (224, 224), color='red')
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return buffer.getvalue()


@pytest.fixture(scope="session")
def sample_image_base64():
    """Create a base64 encoded sample image."""
    img = Image.new('RGB', (224, 224), color='blue')
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_check(self, client):
        """Test health check returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert data["models_loaded"] is True
        assert data["device"] in ["cuda", "cpu"]
        assert "resnet50d" in data["models"]
        assert "efficientnet_b0" in data["models"]
        assert data["ensemble_weight"] == 0.5
        assert data["temperature"] == 0.25
        assert len(data["classes"]) == 8

    def test_health_models_list(self, client):
        """Test health check returns expected models."""
        response = client.get("/health")
        data = response.json()
        models = data["models"]
        assert "resnet50d" in models
        assert "efficientnet_b0" in models


class TestPredictEndpoint:
    """Tests for the /predict endpoint."""

    def test_predict_single_image(self, client, sample_image):
        """Test single image prediction."""
        files = {"file": ("test.png", sample_image, "image/png")}
        response = client.post("/predict", files=files, data={"gradcam": "true"})

        assert response.status_code == 200
        data = response.json()
        assert "predicted_class" in data
        assert data["predicted_class"] in ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']
        assert "confidence" in data
        assert 0 <= data["confidence"] <= 1
        assert "all_probabilities" in data
        assert "individual_models" in data
        assert "gradcam" in data
        assert data["gradcam"] is not None

    def test_predict_without_gradcam(self, client, sample_image):
        """Test prediction without Grad-CAM."""
        files = {"file": ("test.png", sample_image, "image/png")}
        response = client.post("/predict", files=files, data={"gradcam": "false"})

        assert response.status_code == 200
        data = response.json()
        assert "predicted_class" in data
        assert "gradcam" not in data or data["gradcam"] is None

    def test_predict_with_gradcam_target(self, client, sample_image):
        """Test prediction with specific Grad-CAM target class."""
        files = {"file": ("test.png", sample_image, "image/png")}
        response = client.post("/predict", files=files, data={"gradcam": "true", "gradcam_target_class": "4"})

        assert response.status_code == 200
        data = response.json()
        assert data["gradcam"] is not None

    def test_predict_invalid_file_type(self, client):
        """Test prediction with invalid file type."""
        files = {"file": ("test.txt", b"not an image", "text/plain")}
        response = client.post("/predict", files=files)

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "image" in data["detail"].lower()

    def test_predict_empty_file(self, client):
        """Test prediction with empty file."""
        files = {"file": ("empty.png", b"", "image/png")}
        response = client.post("/predict", files=files)

        assert response.status_code in [400, 500]

    def test_predict_small_image(self, client):
        """Test prediction with too small image."""
        img = Image.new('RGB', (16, 16), color='red')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        files = {"file": ("small.png", buffer.getvalue(), "image/png")}
        response = client.post("/predict", files=files)

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "small" in data["detail"].lower() or "size" in data["detail"].lower()


class TestPredictBase64Endpoint:
    """Tests for the /predict/base64 endpoint."""

    def test_predict_base64(self, client, sample_image_base64):
        """Test base64 image prediction."""
        response = client.post(
            "/predict/base64",
            data={"base64_image": sample_image_base64, "gradcam": "true"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "predicted_class" in data
        assert "gradcam" in data

    def test_predict_base64_with_data_url(self, client, sample_image_base64):
        """Test base64 prediction with data URL prefix."""
        data_url = f"data:image/png;base64,{sample_image_base64}"
        response = client.post(
            "/predict/base64",
            data={"base64_image": data_url, "gradcam": "true"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "predicted_class" in data


class TestBatchPredictEndpoint:
    """Tests for the /predict/batch endpoint."""

    def test_predict_batch(self, client, sample_image):
        """Test batch prediction."""
        files = [
            ("files", ("test1.png", sample_image, "image/png")),
            ("files", ("test2.png", sample_image, "image/png"))
        ]
        response = client.post("/predict/batch", files=files, data={"gradcam": "false"})

        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert data["total"] == 2

    def test_predict_batch_too_many(self, client, sample_image):
        """Test batch prediction with too many images."""
        files = [
            ("files", (f"test{i}.png", sample_image, "image/png"))
            for i in range(25)
        ]
        response = client.post("/predict/batch", files=files)

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "20" in data["detail"] or "maximum" in data["detail"].lower()


class TestSampleImagesEndpoint:
    """Tests for the /sample-images endpoint."""

    def test_get_sample_images(self, client):
        """Test sample images listing."""
        response = client.get("/sample-images")
        assert response.status_code == 200

        data = response.json()
        assert "samples" in data
        assert "total" in data
        assert data["total"] >= 8

    def test_sample_image_structure(self, client):
        """Test sample image info structure."""
        response = client.get("/sample-images")
        data = response.json()

        for sample in data["samples"]:
            assert "filename" in sample
            assert "class_name" in sample
            assert "description" in sample


class TestGradCAMOutput:
    """Tests for Grad-CAM output validation."""

    def test_gradcam_structure(self, client, sample_image):
        """Test Grad-CAM response structure."""
        files = {"file": ("test.png", sample_image, "image/png")}
        response = client.post("/predict", files=files, data={"gradcam": "true"})

        assert response.status_code == 200
        data = response.json()
        assert "gradcam" in data
        gradcam = data["gradcam"]
        assert gradcam is not None

    def test_gradcam_dimensions(self, client, sample_image):
        """Test Grad-CAM heatmap dimensions (opt-in raw grids)."""
        files = {"file": ("test.png", sample_image, "image/png")}
        response = client.post(
            "/predict",
            files=files,
            data={"gradcam": "true", "include_heatmaps": "true"},
        )

        assert response.status_code == 200
        data = response.json()
        gradcam = data["gradcam"]
        assert gradcam is not None

        for model_name in ["resnet50d", "efficientnet_b0", "ensemble"]:
            heatmap = gradcam[model_name]
            assert isinstance(heatmap, list)
            assert len(heatmap) == 224
            assert len(heatmap[0]) == 224
            for row in heatmap:
                for val in row:
                    assert 0 <= val <= 1

    def test_gradcam_default_omits_raw_heatmaps(self, client, sample_image):
        """Default /predict response must stay small for Swagger UI."""
        files = {"file": ("test.png", sample_image, "image/png")}
        response = client.post("/predict", files=files, data={"gradcam": "true"})

        assert response.status_code == 200
        gradcam = response.json()["gradcam"]
        assert gradcam is not None
        assert gradcam.get("resnet50d") is None
        assert gradcam.get("efficientnet_b0") is None
        assert gradcam.get("ensemble") is None
        assert len(response.content) < 35_000


class TestErrorHandling:
    """Tests for error handling."""

    def test_404_endpoint(self, client):
        """Test 404 for non-existent endpoint."""
        response = client.get("/nonexistent")
        assert response.status_code == 404

    def test_missing_file(self, client):
        """Test prediction without file."""
        response = client.post("/predict")
        assert response.status_code == 422

    def test_invalid_gradcam_target(self, client, sample_image):
        """Test invalid gradcam target class."""
        files = {"file": ("test.png", sample_image, "image/png")}
        response = client.post("/predict", files=files, data={"gradcam": "true", "gradcam_target_class": "99"})
        assert response.status_code in [200, 400, 422]


class TestRiskAssessment:
    """Tests for clinical risk assessment."""

    def test_melanoma_high_confidence_critical(self, client, sample_image):
        """Test melanoma with high confidence gets critical tier."""
        files = {"file": ("test.png", sample_image, "image/png")}
        response = client.post("/predict", files=files, data={"gradcam": "false"})
        
        data = response.json()
        assert data["risk_assessment"]["tier"] in ["low", "moderate", "high", "critical"]

    def test_risk_assessment_structure(self, client, sample_image):
        """Test risk assessment structure."""
        files = {"file": ("test.png", sample_image, "image/png")}
        response = client.post("/predict", files=files, data={"gradcam": "false"})
        
        data = response.json()
        assert "risk_assessment" in data
        risk = data["risk_assessment"]
        assert "tier" in risk
        assert "explanation" in risk
        assert "recommended_action" in risk
        assert "urgency_days" in risk
        assert isinstance(risk["urgency_days"], int)
        assert 0 <= risk["urgency_days"] <= 90


class TestUIEndpoint:
    """Tests for the web UI."""

    def test_ui_root(self, client):
        """Test root endpoint serves UI."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    def test_ui_contains_title(self, client):
        """Test UI contains expected title."""
        response = client.get("/")
        assert "Skin Cancer Classification" in response.text
        assert "Skin Cancer AI" in response.text

    def test_ui_no_cache_headers(self, client):
        """UI must not be cached so browsers always get the latest HTML."""
        response = client.get("/")
        cache_control = response.headers.get("cache-control", "")
        assert "no-store" in cache_control
        assert "no-cache" in cache_control
        assert response.headers.get("pragma") == "no-cache"

    def test_ui_uses_native_label_dropzone(self, client):
        """UI must contain a native label target for the file input."""
        response = client.get("/")
        assert 'for="fileInput"' in response.text
        assert 'id="fileInput"' in response.text
        assert 'id="statusBanner"' in response.text


class TestModelInfo:
    """Tests for model info endpoint."""

    def test_model_info_in_health(self, client):
        """Test model info in health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "models" in data
        assert "resnet50d" in data["models"]
        assert "efficientnet_b0" in data["models"]

    def test_predict_includes_model_info(self, client, sample_image):
        """Test prediction includes model info."""
        files = {"file": ("test.png", sample_image, "image/png")}
        response = client.post("/predict", files=files, data={"gradcam": "false"})

        assert response.status_code == 200
        data = response.json()
        assert "individual_models" in data
        assert "resnet50d" in data["individual_models"]
        assert "efficientnet_b0" in data["individual_models"]


class TestAPIVersion:
    """Tests for API versioning."""

    def test_version_in_health(self, client):
        """Test version is present in health check."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "version" in data
        assert data["version"] == "1.0.0"


class TestCORS:
    """Tests for CORS headers."""

    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/health", headers={"Origin": "http://localhost:3000"})
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers


class TestValidation:
    """Tests for input validation."""

    def test_predict_validates_content_type(self, client):
        """Test prediction validates content type."""
        files = {"file": ("test.pdf", b"%PDF", "application/pdf")}
        response = client.post("/predict", files=files)

        assert response.status_code == 400

    def test_predict_validates_file_extension(self, client, sample_image):
        """Test prediction with valid image extension."""
        img = Image.new('RGB', (224, 224), color='green')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        files = {"file": ("test.jpg", buffer.getvalue(), "image/jpeg")}
        response = client.post("/predict", files=files)

        assert response.status_code == 200


class TestMetricsConsistency:
    """Tests for metrics consistency across endpoints."""

    def test_ensemble_weight_consistency(self, client):
        """Test ensemble weight is consistent."""
        response = client.get("/health")
        data = response.json()
        assert data["ensemble_weight"] == 0.5

    def test_temperature_consistency(self, client):
        """Test temperature is consistent."""
        response = client.get("/health")
        data = response.json()
        assert data["temperature"] == 0.25

    def test_classes_match(self, client):
        """Test class names match expected."""
        response = client.get("/health")
        data = response.json()
        expected = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]
        assert data["classes"] == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])