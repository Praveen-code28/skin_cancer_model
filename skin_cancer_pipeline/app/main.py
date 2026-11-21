"""
FastAPI application for skin cancer classification inference.

Provides REST API endpoints for:
- Health checks
- Single image prediction with optional Grad-CAM
- Sample images for testing
- Static web UI
"""

import base64
import io
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import numpy as np

from src.inference import SkinCancerEnsemblePredictor, create_predictor_from_config
from app.schemas import (
    BatchPredictionResponse,
    ErrorResponse,
    GradCAMResponse,
    HealthResponse,
    PredictionResponse,
    RiskAssessment,
    SampleImageInfo,
    SampleImagesResponse,
    calculate_risk_assessment,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("skin_cancer_api")

# Global predictor instance
predictor: Optional["SkinCancerEnsemblePredictor"] = None

# Paths
BASE_DIR = Path(__file__).parent.parent
STATIC_DIR = BASE_DIR / "app" / "static"
RESULTS_DIR = BASE_DIR / "results"
CONFIG_PATH = BASE_DIR / "configs" / "config.yaml"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global predictor

    # Startup
    logger.info("Starting Skin Cancer Classification API...")
    try:
        global predictor
        predictor = SkinCancerEnsemblePredictor(
            resnet50d_checkpoint=str(RESULTS_DIR / "checkpoints" / "resnet50d" / "best_model.pt"),
            efficientnet_b0_checkpoint=str(RESULTS_DIR / "checkpoints" / "efficientnet_b0" / "best_model.pt"),
            label_encoder_path=str(RESULTS_DIR / "checkpoints" / "resnet50d" / "label_encoder.json"),
            ensemble_weight=0.5,
            temperature=0.25,
        )
        logger.info("Models loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down API...")


# Create FastAPI app
app = FastAPI(
    title="Skin Cancer Classification API",
    description="Production-ready ensemble API for skin lesion classification with Grad-CAM explainability",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _pil_to_base64(image: Image.Image) -> str:
    """Convert PIL image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _base64_to_pil(base64_str: str) -> Image.Image:
    """Convert base64 string to PIL Image."""
    if base64_str.startswith("data:image"):
        base64_str = base64_str.split(",")[1]
    image_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_data))


def _array_to_base64(array: np.ndarray) -> str:
    """Convert numpy array (HxW, 0-1) to base64 PNG."""
    # Scale to 0-255
    img_array = (array * 255).astype(np.uint8)
    img = Image.fromarray(img_array, mode='L')  # Grayscale
    return _pil_to_base64(img)


def _to_list(value):
    """Convert numpy arrays (or nested lists) to plain Python lists."""
    return value.tolist() if hasattr(value, "tolist") else value


def _build_gradcam_response(
    gradcam_data: dict,
    overlay_base64: Optional[str],
    include_heatmaps: bool = False,
) -> GradCAMResponse:
    """Build a Grad-CAM response, omitting the heavy raw heatmaps by default."""
    if not include_heatmaps:
        return GradCAMResponse(overlay_base64=overlay_base64)

    return GradCAMResponse(
        resnet50d=_to_list(gradcam_data["resnet50d"]),
        efficientnet_b0=_to_list(gradcam_data["efficientnet_b0"]),
        ensemble=_to_list(gradcam_data["ensemble"]),
        overlay_base64=overlay_base64,
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded"
        )

    info = predictor.get_model_info()
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded=True,
        device=info["device"],
        models=info["models"],
        ensemble_weight=info["ensemble_weight"],
        temperature=info["temperature"],
        classes=info["class_names"],
    )


@app.options("/health", tags=["Health"])
async def health_check_options():
    """OPTIONS handler for CORS preflight requests."""
    return {"message": "OK"}


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(
    file: UploadFile = File(..., description="Skin lesion image (JPEG/PNG)"),
    gradcam: bool = Form(False, description="Whether to return Grad-CAM heatmaps"),
    gradcam_target_class: Optional[int] = Form(None, description="Target class index for Grad-CAM (0-7, optional)"),
    include_heatmaps: bool = Form(
        False,
        description="Include raw 224x224 Grad-CAM grids in the JSON response (large payload; off by default)",
    ),
):
    """
    Predict skin lesion class from uploaded image.

    Args:
        file: Image file (JPEG, PNG)
        gradcam: Whether to return Grad-CAM heatmaps
        gradcam_target_class: Target class index for Grad-CAM (0-7, optional)

    Returns:
        Prediction with probabilities, risk assessment, and optional Grad-CAM
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded"
        )

    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image (JPEG, PNG)"
        )

    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Validate image dimensions
        if image.width < 32 or image.height < 32:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Image too small (minimum 32x32 pixels)"
            )

        # Run prediction
        result = predictor.predict(
            image=image,
            return_gradcam=gradcam,
            gradcam_target_class=gradcam_target_class,
        )

        # Calculate risk assessment
        risk = calculate_risk_assessment(
            result["predicted_class"],
            result["confidence"]
        )

        # Build response
        response = PredictionResponse(
            predicted_class=result["predicted_class"],
            predicted_class_idx=result["predicted_class_idx"],
            confidence=result["confidence"],
            all_probabilities=result["all_probabilities"],
            individual_models=result["individual_models"],
            risk_assessment=RiskAssessment(**risk),
        )

        # Add Grad-CAM if requested
        if gradcam and "gradcam" in result:
            response.gradcam = _build_gradcam_response(
                result["gradcam"],
                result.get("gradcam_overlay_base64"),
                include_heatmaps=include_heatmaps,
            )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
async def predict_batch(
    files: List[UploadFile] = File(..., description="Multiple image files"),
    gradcam: bool = Form(False, description="Whether to return Grad-CAM heatmaps"),
    include_heatmaps: bool = Form(
        False,
        description="Include raw 224x224 Grad-CAM grids (large payload; off by default)",
    ),
):
    """
    Predict skin lesion class for multiple images.

    Args:
        files: List of image files
        gradcam: Whether to return Grad-CAM heatmaps

    Returns:
        List of predictions
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded"
        )

    if len(files) > 20:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 20 images per batch"
        )

    results = []
    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))

            result = predictor.predict(image=image, return_gradcam=gradcam)
            risk = calculate_risk_assessment(result["predicted_class"], result["confidence"])

            pred_response = PredictionResponse(
                predicted_class=result["predicted_class"],
                predicted_class_idx=result["predicted_class_idx"],
                confidence=result["confidence"],
                all_probabilities=result["all_probabilities"],
                individual_models=result["individual_models"],
                risk_assessment=RiskAssessment(**risk),
            )

            if gradcam and "gradcam" in result:
                pred_response.gradcam = _build_gradcam_response(
                    result["gradcam"],
                    result.get("gradcam_overlay_base64"),
                    include_heatmaps=include_heatmaps,
                )

            results.append(pred_response)

        except Exception as e:
            logger.error(f"Batch prediction error for {file.filename}: {e}")
            # Continue with other images
            continue

    return BatchPredictionResponse(predictions=results, total=len(results))


@app.post("/predict/base64", response_model=PredictionResponse, tags=["Inference"])
async def predict_base64(
    base64_image: str = Form(..., description="Base64 encoded image (with or without data URL prefix)"),
    gradcam: bool = Form(False, description="Whether to return Grad-CAM heatmaps"),
    gradcam_target_class: Optional[int] = Form(None, description="Target class index for Grad-CAM (0-7)"),
    include_heatmaps: bool = Form(
        False,
        description="Include raw 224x224 Grad-CAM grids (large payload; off by default)",
    ),
):
    """
    Predict from base64 encoded image.

    Args:
        base64_image: Base64 encoded image string (with or without data URL prefix)
        gradcam: Whether to return Grad-CAM heatmaps
        gradcam_target_class: Target class index for Grad-CAM

    Returns:
        Prediction with probabilities and risk assessment
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded"
        )

    try:
        # Decode base64 image
        image = _base64_to_pil(base64_image)

        # Run prediction
        result = predictor.predict(
            image=image,
            return_gradcam=gradcam,
            gradcam_target_class=gradcam_target_class,
        )

        risk = calculate_risk_assessment(
            result["predicted_class"],
            result["confidence"]
        )

        response = PredictionResponse(
            predicted_class=result["predicted_class"],
            predicted_class_idx=result["predicted_class_idx"],
            confidence=result["confidence"],
            all_probabilities=result["all_probabilities"],
            individual_models=result["individual_models"],
            risk_assessment=RiskAssessment(**risk),
        )

        if gradcam and "gradcam" in result:
            response.gradcam = _build_gradcam_response(
                result["gradcam"],
                result.get("gradcam_overlay_base64"),
                include_heatmaps=include_heatmaps,
            )

        return response

    except Exception as e:
        logger.error(f"Base64 prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/sample-images", response_model=SampleImagesResponse, tags=["Utilities"])
async def get_sample_images():
    """
    Get available sample images for testing.

    Returns list of sample images with metadata.
    """
    samples_dir = RESULTS_DIR / "sample_images"

    if not samples_dir.exists():
        # Create synthetic sample info from test set metadata
        return SampleImagesResponse(
            samples=[
                SampleImageInfo(
                    filename=f"sample_{cls}.png",
                    class_name=cls,
                    description=f"Sample {cls} lesion"
                )
                for cls in ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]
            ],
            total=8
        )

    # If actual sample images exist
    samples = []
    for img_path in samples_dir.glob("*.png"):
        # Extract class from filename if possible
        class_name = img_path.stem.split("_")[0] if "_" in img_path.stem else "Unknown"
        samples.append(SampleImageInfo(
            filename=img_path.name,
            class_name=class_name,
            description=f"Sample {class_name} lesion"
        ))

    return SampleImagesResponse(samples=samples, total=len(samples))


@app.get("/sample-images/{filename}", tags=["Utilities"])
async def get_sample_image(filename: str):
    """
    Download a specific sample image.
    """
    samples_dir = RESULTS_DIR / "sample_images"
    file_path = samples_dir / filename

    if not file_path.exists():
        # Fallback to gradcam visualizations
        for model_dir in (RESULTS_DIR / "gradcam").glob("*"):
            file_path = model_dir / filename
            if file_path.exists():
                break

    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sample image not found: {filename}"
        )

    return FileResponse(
        path=file_path,
        media_type="image/png",
        filename=filename
    )


# Serve the web UI at root
@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def serve_ui():
    """Serve the main web UI (always fresh, never cached by the browser)."""
    ui_path = STATIC_DIR / "index.html"
    if ui_path.exists():
        return FileResponse(
            str(ui_path),
            media_type="text/html",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    # Fallback HTML if static file not found
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Skin Cancer Classification API</title>
        <style>
            body { font-family: system-ui, sans-serif; max-width: 800px; margin: 2rem auto; padding: 0 1rem; }
            .endpoint { background: #f5f5f5; padding: 1rem; margin: 1rem 0; border-radius: 8px; }
            code { background: #eee; padding: 0.2rem 0.4rem; border-radius: 4px; }
        </style>
    </head>
    <body>
        <h1>Skin Cancer Classification API</h1>
        <p>Ensemble model: ResNet50d + EfficientNet-B0 (w=0.50, τ=0.25)</p>

        <h2>API Endpoints</h2>
        <div class="endpoint"><strong>GET /health</strong> - Health check</div>
        <div class="endpoint"><strong>POST /predict</strong> - Single image prediction (multipart/form-data)</div>
        <div class="endpoint"><strong>POST /predict/base64</strong> - Single image prediction (base64)</div>
        <div class="endpoint"><strong>POST /predict/batch</strong> - Batch prediction (up to 20 images)</div>
        <div class="endpoint"><strong>GET /sample-images</strong> - List sample images</div>
        <div class="endpoint"><strong>GET /docs</strong> - Interactive API documentation</div>

        <h2>Quick Test</h2>
        <p>Upload an image to <code>POST /predict</code> with <code>gradcam=true</code> for explainability.</p>
    </body>
    </html>
    """)


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=exc.detail,
            status_code=exc.status_code
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            status_code=500
        ).model_dump()
    )


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )