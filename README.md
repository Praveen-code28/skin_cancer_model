# Skin Cancer Classification

A research and educational project for classifying dermoscopic images into eight
lesion categories. The pipeline includes model training and evaluation, a
ResNet50d/EfficientNet-B0 ensemble, Grad-CAM visualizations, and a FastAPI
inference interface.

> **Important:** This project is not a medical device and is not intended for
> diagnosis or treatment decisions. Predictions require review by qualified
> healthcare professionals.

## Repository contents

- `skin_cancer_pipeline/` - training, evaluation, API, configuration, and tests
- `dataset/README.md` - dataset layout, provenance, and license information

The dataset, trained weights, and generated results are intentionally excluded
from Git because of their size and licensing. See the dataset instructions
before running the pipeline.

## Setup

Use Python 3.10 or later. The dependency file pins CUDA 13.4 PyTorch wheels;
install those first, then install the remaining project dependencies:

```powershell
cd skin_cancer_pipeline
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.15.0+cu134 torchvision==0.29.2+cu134 torchaudio==2.11.1+cu134 --index-url https://download.pytorch.org/whl/cu134
python -m pip install -r requirements.txt
```

Place the dataset in the repository's `dataset/` directory. The supplied
configuration uses `../dataset` when commands are run from
`skin_cancer_pipeline/`.

## Train and evaluate

```powershell
python run_pipeline.py --config configs/config.yaml
```

The pipeline writes checkpoints, metrics, plots, and logs under
`skin_cancer_pipeline/results/`. These generated files are local and are not
tracked by Git.

Run the test suite with:

```powershell
python -m pytest tests/ -v --tb=short
```

## Inference API

The API requires trained model checkpoints under `results/checkpoints/`. After
training, start the server from `skin_cancer_pipeline/`:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/` for the web interface or
`http://127.0.0.1:8000/docs` for the API documentation.

## License

No license is currently specified for the project source code. The dataset has
its own license and attribution requirements; see
[`dataset/README.md`](dataset/README.md) before obtaining or using it.
