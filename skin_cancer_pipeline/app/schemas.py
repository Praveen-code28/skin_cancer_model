"""
Pydantic v2 request/response models for the skin cancer classification API.

Includes clinical risk tiering for actionable decision support.
"""

from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field, ConfigDict
from typing_extensions import Annotated


class ClassProbability(BaseModel):
    """Probability for a single class."""
    class_name: str = Field(..., description="Class label (e.g., 'MEL', 'NV')")
    probability: float = Field(..., ge=0.0, le=1.0, description="Predicted probability")


class IndividualModelPrediction(BaseModel):
    """Prediction from a single model in the ensemble."""
    predicted_class: str = Field(..., description="Predicted class label")
    probabilities: Dict[str, float] = Field(
        ...,
        description="Per-class probabilities from this model"
    )


class GradCAMResponse(BaseModel):
    """Grad-CAM response.

    Raw heatmap grids are optional: they are large (224x224 floats each) and are
    only returned when explicitly requested via ``include_heatmaps=true``. The
    compact ``overlay_base64`` preview is always returned.
    """
    resnet50d: Optional[List[List[float]]] = Field(
        None,
        description="Grad-CAM heatmap from ResNet50d (HxW grid, values 0-1). Only when include_heatmaps=true."
    )
    efficientnet_b0: Optional[List[List[float]]] = Field(
        None,
        description="Grad-CAM heatmap from EfficientNet-B0 (HxW grid, values 0-1). Only when include_heatmaps=true."
    )
    ensemble: Optional[List[List[float]]] = Field(
        None,
        description="Averaged ensemble Grad-CAM heatmap (HxW grid, values 0-1). Only when include_heatmaps=true."
    )
    overlay_base64: Optional[str] = Field(
        None,
        description="Base64-encoded JPEG overlay image (original + Grad-CAM side-by-side, 384x192, quality=75)"
    )


RiskTier = Literal["low", "moderate", "high", "critical"]
"""Clinical risk tier based on predicted class and confidence."""


class RiskAssessment(BaseModel):
    """Clinical risk assessment for the prediction."""
    tier: RiskTier = Field(..., description="Overall risk tier")
    explanation: str = Field(..., description="Human-readable explanation")
    recommended_action: str = Field(..., description="Suggested clinical action")
    urgency_days: int = Field(..., ge=0, le=90, description="Recommended follow-up timeframe in days")


class PredictionResponse(BaseModel):
    """Complete prediction response for a single image."""
    predicted_class: str = Field(..., description="Final ensemble prediction")
    predicted_class_idx: int = Field(..., ge=0, le=7, description="Class index (0-7)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence (0-1)")
    all_probabilities: Dict[str, float] = Field(
        ...,
        description="Per-class probabilities from ensemble"
    )
    individual_models: Dict[str, IndividualModelPrediction] = Field(
        ...,
        description="Individual model predictions"
    )
    gradcam: Optional[GradCAMResponse] = Field(
        None,
        description="Grad-CAM heatmaps (if requested)"
    )
    risk_assessment: RiskAssessment = Field(
        ...,
        description="Clinical risk assessment"
    )
    model_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Model metadata (ensemble weight, temperature, etc.)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "predicted_class": "MEL",
                "predicted_class_idx": 4,
                "confidence": 0.87,
                "all_probabilities": {
                    "AK": 0.02,
                    "BCC": 0.05,
                    "BKL": 0.03,
                    "DF": 0.01,
                    "MEL": 0.87,
                    "NV": 0.05,
                    "SCC": 0.02,
                    "VASC": 0.01
                },
                "individual_models": {
                    "resnet50d": {
                        "predicted_class": "MEL",
                        "probabilities": {"MEL": 0.85, "NV": 0.08, "BCC": 0.04}
                    },
                    "efficientnet_b0": {
                        "predicted_class": "MEL",
                        "probabilities": {"MEL": 0.89, "NV": 0.06, "BCC": 0.03}
                    }
                },
                "risk_assessment": {
                    "tier": "high",
                    "explanation": "High probability of melanoma (MEL). Requires urgent dermatology referral.",
                    "recommended_action": "Urgent dermatology referral within 2 weeks for biopsy.",
                    "urgency_days": 14
                }
            }
        }
    )


class BatchPredictionResponse(BaseModel):
    """Response for batch prediction endpoint."""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total: int = Field(..., description="Total number of images processed")


class HealthResponse(BaseModel):
    """Health check response."""
    status: Literal["healthy"] = "healthy"
    version: str = Field(..., description="API version")
    models_loaded: bool = Field(..., description="Whether models are loaded")
    device: str = Field(..., description="Inference device (cuda/cpu)")
    models: Dict[str, Any] = Field(..., description="Loaded model details")
    ensemble_weight: float = Field(..., description="Ensemble blend weight (ResNet50d)")
    temperature: float = Field(..., description="Prior adjustment temperature")
    classes: List[str] = Field(..., description="Available class labels")


class SampleImageInfo(BaseModel):
    """Information about a sample image."""
    filename: str = Field(..., description="Sample image filename")
    class_name: str = Field(..., description="True class label")
    description: str = Field(..., description="Brief description")


class SampleImagesResponse(BaseModel):
    """Response for sample images endpoint."""
    samples: List[SampleImageInfo] = Field(..., description="Available sample images")
    total: int = Field(..., description="Total number of samples")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    status_code: int = Field(..., description="HTTP status code")


# Risk tier calculation logic (shared between API and UI)
RISK_TIER_CONFIG = {
    "MEL": {"base_tier": "critical", "min_confidence_high": 0.5},
    "SCC": {"base_tier": "high", "min_confidence_high": 0.4},
    "BCC": {"base_tier": "moderate", "min_confidence_high": 0.5},
    "AK": {"base_tier": "moderate", "min_confidence_high": 0.4},
    "DF": {"base_tier": "low", "min_confidence_high": 0.6},
    "BKL": {"base_tier": "low", "min_confidence_high": 0.6},
    "VASC": {"base_tier": "low", "min_confidence_high": 0.6},
    "NV": {"base_tier": "low", "min_confidence_high": 0.7},
}

RISK_TIER_EXPLANATIONS = {
    "critical": "High probability of melanoma. Requires immediate dermatology evaluation.",
    "high": "Elevated probability of skin cancer. Prompt dermatology referral recommended.",
    "moderate": "Possible pre-cancerous or concerning lesion. Dermatology evaluation within 4-6 weeks.",
    "low": "Low probability of malignancy. Routine monitoring or follow-up as appropriate.",
}

RISK_ACTIONS = {
    "critical": "Urgent dermatology referral within 2 weeks for excisional biopsy.",
    "high": "Dermatology referral within 2-4 weeks for evaluation and possible biopsy.",
    "moderate": "Dermatology evaluation within 4-6 weeks. Consider dermoscopic monitoring.",
    "low": "Routine skin surveillance. Annual full-body skin exam recommended.",
}

RISK_URGENCY = {
    "critical": 14,
    "high": 28,
    "moderate": 42,
    "low": 90,
}


def calculate_risk_assessment(predicted_class: str, confidence: float) -> Dict[str, Any]:
    """
    Calculate clinical risk assessment based on predicted class and confidence.

    Args:
        predicted_class: Predicted class label
        confidence: Prediction confidence (0-1)

    Returns:
        Dictionary with tier, explanation, recommended_action, urgency_days
    """
    config = RISK_TIER_CONFIG.get(predicted_class, {"base_tier": "low", "min_confidence_high": 0.7})
    base_tier = config["base_tier"]
    min_conf_high = config["min_confidence_high"]

    # Adjust tier based on confidence
    if confidence >= min_conf_high:
        tier = base_tier
    elif confidence >= min_conf_high * 0.7:
        # Downgrade one level
        tier_order = ["critical", "high", "moderate", "low"]
        current_idx = tier_order.index(base_tier)
        tier = tier_order[min(current_idx + 1, len(tier_order) - 1)]
    else:
        # Downgrade two levels
        tier_order = ["critical", "high", "moderate", "low"]
        current_idx = tier_order.index(base_tier)
        tier = tier_order[min(current_idx + 2, len(tier_order) - 1)]

    return {
        "tier": tier,
        "explanation": RISK_TIER_EXPLANATIONS.get(tier, "Risk assessment based on model prediction."),
        "recommended_action": RISK_ACTIONS.get(tier, "Consult a dermatologist for evaluation."),
        "urgency_days": RISK_URGENCY.get(tier, 90),
    }