# ISIC 2019 Preprocessed + Leakage-Free Split

**25,331 dermoscopy images of skin lesions in 8 classes, cleaned and resized to 224×224, with a lesion mask for every image and a train / val / test split where no lesion appears in two splits.** Ready to train on. 0.3 GB instead of 9.1 GB for the original images.

## Why this version

- **No data leakage.** Many ISIC images are different photos of the same lesion. A random split puts photos of one lesion in both train and test, which makes test scores look better than they really are. Here every photo of a lesion stays in one split: **0 lesions shared** between train, val and test.
- **Already cleaned.** Noise reduced, lighting corrected, contrast enhanced, hair digitally removed.
- **Masks included.** A lesion outline for every image, for segmentation or cropping experiments.
- **Drop-in folders.** One folder per class, so `image_dataset_from_directory` (Keras) and `ImageFolder` (PyTorch) work directly.

## Files

```
images/<split>/<class>/ISIC_xxxxxxx.jpg   224×224 colour image
masks/<split>/<class>/ISIC_xxxxxxx.png    224×224, white = lesion, black = background
labels.csv                                one row per image
sample_grid.png                           one example per class
```

`<split>` is `train`, `val` or `test`. `<class>` is one of the class codes below. Image and mask share the same file name.

**labels.csv columns:** `image` (ISIC ID), `label` (class code), `split`, `image_path`, `mask_path`, `lesion_id` (photos of the same lesion share it), `age_approx` (5-year steps), `sex`, `anatom_site_general` (body location). Blank means unknown.

## Classes

| Code | Diagnosis | Train | Val | Test | Total |
|---|---|---:|---:|---:|---:|
| AK | Actinic keratosis | 725 | 57 | 85 | 867 |
| BCC | Basal cell carcinoma | 2,591 | 339 | 393 | 3,323 |
| BKL | Benign keratosis | 2,115 | 248 | 261 | 2,624 |
| DF | Dermatofibroma | 186 | 31 | 22 | 239 |
| MEL | Melanoma | 3,695 | 386 | 441 | 4,522 |
| NV | Melanocytic nevus (mole) | 10,284 | 1,284 | 1,307 | 12,875 |
| SCC | Squamous cell carcinoma | 526 | 62 | 40 | 628 |
| VASC | Vascular lesion | 186 | 41 | 26 | 253 |
| | **All** | **20,308** | **2,448** | **2,575** | **25,331** |

The classes are very unbalanced (NV is 51% of all images). Val and test keep the real-world mix on purpose: balance only the training data (class weights, oversampling), and report **balanced accuracy** or **macro-F1** rather than plain accuracy.

## Quick start

```python
import pandas as pd, tensorflow as tf
ROOT = "/kaggle/input/isic-2019-preprocessed-leakage-free-split"
labels = pd.read_csv(f"{ROOT}/labels.csv")
train_ds = tf.keras.utils.image_dataset_from_directory(
    f"{ROOT}/images/train", image_size=(224, 224), batch_size=32)
```

PyTorch: `torchvision.datasets.ImageFolder(f"{ROOT}/images/train")`.

## How the images were made

Starting from the original full-size ISIC 2019 training images, with OpenCV:

1. **Denoise:** Non-Local Means (h = 5, template 7, search 21).
2. **Lighting correction:** Shades-of-Gray colour constancy (p = 6).
3. **Contrast:** CLAHE on the L channel of LAB (clip limit 2.0, 8×8 tiles).
4. **Resize:** centre crop to a square, then 224×224 (area interpolation).
5. **Hair removal:** DullRazor (black-hat with a 17×17 cross, threshold 10, 3×3 dilation, Telea inpainting).

**Masks** (made from the hair-removed images): 5×5 Gaussian blur, Otsu threshold, closing then opening (7×7, 2 passes each), keep the largest region. They are automatic, not drawn by doctors, so some are rough.

**Split:** scikit-learn `StratifiedGroupKFold` grouped by `lesion_id` and stratified by class (seed 42). One fold of five (20%) was held out, then divided in half into val and test. Images with no `lesion_id` count as their own lesion.

## Limitations

- Made from the ISIC 2019 **training** images only, so there is no "unknown" (UNK) class.
- The centre crop can cut off the edges of lesions in wide photos.
- Colours were changed on purpose by the preprocessing. To choose your own preprocessing, start from the original images.

## License and credit

**CC BY-NC 4.0**, the same license as the original data: free for **non-commercial** use with credit. The changes made are listed above. Not affiliated with or endorsed by ISIC.

Source: ISIC 2019 Challenge, https://challenge.isic-archive.com/data/#2019. Please cite the original data:

> BCN_20000 Dataset: (c) Department of Dermatology, Hospital Clínic de Barcelona
> HAM10000 Dataset: (c) by ViDIR Group, Department of Dermatology, Medical University of Vienna; https://doi.org/10.1038/sdata.2018.161
> MSK Dataset: (c) Anonymous; https://arxiv.org/abs/1710.05006; https://arxiv.org/abs/1902.03368

1. Tschandl P., Rosendahl C. & Kittler H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Sci. Data* 5, 180161 (2018).
2. Codella N. C. F. et al. Skin Lesion Analysis Toward Melanoma Detection: A Challenge at the 2017 ISBI, Hosted by the ISIC. arXiv:1710.05006 (2017).
3. Hernández-Pérez C. et al. BCN20000: Dermoscopic lesions in the wild. *Sci. Data* 11, 641 (2024).
