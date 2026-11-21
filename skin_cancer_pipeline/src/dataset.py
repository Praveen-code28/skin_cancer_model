"""
Dataset module for skin cancer classification.
Handles data loading, stratified splitting, and Albumentations transforms.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from utils import get_class_names, set_seed


logger = logging.getLogger("skin_cancer_pipeline")


class SkinCancerDataset(Dataset):
    """
    PyTorch Dataset for skin cancer images with Albumentations transforms.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        root_path: str,
        transform: Optional[A.Compose] = None,
        label_encoder: Optional[LabelEncoder] = None,
        is_test: bool = False
    ):
        """
        Initialize dataset.
        
        Args:
            df: DataFrame with columns ['image', 'label', 'image_path', 'lesion_id', ...]
            root_path: Root directory of dataset
            transform: Albumentations transform pipeline
            label_encoder: Fitted LabelEncoder for labels
            is_test: Whether this is test set (no label encoding needed if labels not available)
        """
        self.df = df.reset_index(drop=True)
        self.root_path = Path(root_path)
        self.transform = transform
        self.label_encoder = label_encoder
        self.is_test = is_test
        
        # Verify image paths exist
        self._verify_paths()
        
        # Encode labels if not test
        if not is_test and label_encoder is not None:
            self.targets = label_encoder.transform(self.df['label'].values)
        elif not is_test:
            self.label_encoder = LabelEncoder()
            self.targets = self.label_encoder.fit_transform(self.df['label'].values)
        else:
            self.targets = None
    
    def _verify_paths(self) -> None:
        """Verify that image files exist."""
        missing = []
        for idx, row in self.df.iterrows():
            img_path = self.root_path / row['image_path']
            if not img_path.exists():
                missing.append(str(img_path))
        
        if missing:
            logger.warning(f"Missing {len(missing)} image files (first 5): {missing[:5]}")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        img_path = self.root_path / row['image_path']
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        sample = {
            'image': image,
            'image_id': row['image'],
            'lesion_id': row['lesion_id']
        }
        
        if not self.is_test and self.targets is not None:
            sample['label'] = torch.tensor(self.targets[idx], dtype=torch.long)
            sample['label_name'] = row['label']
        
        return sample
    
    def get_class_distribution(self) -> Counter:
        """Get class distribution in dataset."""
        if self.targets is not None:
            return Counter(self.targets)
        return Counter(self.df['label'].values)


def get_train_transforms(config: Dict[str, Any]) -> A.Compose:
    """
    Get training augmentation pipeline.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Albumentations Compose object
    """
    aug_config = config['data']['train_augmentation']
    image_size = config['data']['image_size']
    
    transforms = []
    
    # Geometric transforms
    if aug_config.get('horizontal_flip', 0) > 0:
        transforms.append(A.HorizontalFlip(p=aug_config['horizontal_flip']))
    if aug_config.get('vertical_flip', 0) > 0:
        transforms.append(A.VerticalFlip(p=aug_config['vertical_flip']))
    if aug_config.get('rotate_limit', 0) > 0:
        transforms.append(A.Rotate(limit=aug_config['rotate_limit'], p=0.5))
    if aug_config.get('shift_scale_rotate', 0) > 0:
        transforms.append(A.Affine(
            translate_percent=0.0625,
            scale=(0.9, 1.1),
            rotate=(-15, 15),
            p=aug_config['shift_scale_rotate'],
            border_mode=cv2.BORDER_REFLECT_101
        ))
    
    # Photometric transforms
    if aug_config.get('brightness_contrast', 0) > 0:
        transforms.append(A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=aug_config['brightness_contrast']
        ))
    if aug_config.get('hue_saturation', 0) > 0:
        transforms.append(A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=10,
            p=aug_config['hue_saturation']
        ))
    if aug_config.get('gauss_noise', 0) > 0:
        transforms.append(A.GaussNoise(std_range=(0.1, 0.3), p=aug_config['gauss_noise']))
    if aug_config.get('cutout', 0) > 0:
        transforms.append(A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(8, 16),
            hole_width_range=(8, 16),
            fill=0,
            p=aug_config['cutout']
        ))
    
    # Resize and normalize
    transforms.extend([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    return A.Compose(transforms)


def get_val_test_transforms(config: Dict[str, Any]) -> A.Compose:
    """
    Get validation/test transform pipeline (no augmentation, only resize + normalize).
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Albumentations Compose object
    """
    image_size = config['data']['image_size']
    
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def compute_class_weights(df: pd.DataFrame, label_encoder: LabelEncoder) -> torch.Tensor:
    """
    Compute class weights for imbalanced dataset.
    
    Args:
        df: DataFrame with 'label' column
        label_encoder: Fitted LabelEncoder
    
    Returns:
        Tensor of class weights
    """
    class_counts = df['label'].value_counts().sort_index()
    total = len(df)
    num_classes = len(class_counts)
    
    # Effective number of samples (Class-Balanced Loss)
    beta = 0.9999
    effective_num = 1.0 - np.power(beta, class_counts.values)
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum() * num_classes
    
    # Alternative: simple inverse frequency
    # weights = total / (num_classes * class_counts.values)
    
    logger.info(f"Class weights: {dict(zip(class_counts.index, weights))}")
    
    return torch.tensor(weights, dtype=torch.float32)


def get_weighted_sampler(df: pd.DataFrame, label_encoder: LabelEncoder) -> WeightedRandomSampler:
    """
    Create WeightedRandomSampler for balanced training.
    
    Args:
        df: Training DataFrame
        label_encoder: Fitted LabelEncoder
    
    Returns:
        WeightedRandomSampler
    """
    targets = label_encoder.transform(df['label'].values)
    class_counts = np.bincount(targets)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[targets]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    logger.info(f"Created WeightedRandomSampler with weights: {class_weights}")
    
    return sampler


def custom_collate_fn(batch):
    """
    Custom collate function that handles string fields properly.
    Keeps string fields as lists instead of trying to tensorize them.
    """
    # Keys that should remain as lists (strings)
    string_keys = {'image_id', 'lesion_id', 'label_name'}
    
    # Separate string fields from tensor fields
    collated = {}
    for key in batch[0].keys():
        if key in string_keys:
            # Keep string fields as lists
            collated[key] = [item[key] for item in batch]
        else:
            # Tensorize numeric fields using default collate
            from torch.utils.data._utils.collate import default_collate
            collated[key] = default_collate([item[key] for item in batch])
    return collated


def create_data_loaders(
    config: Dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_encoder: LabelEncoder
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders.
    
    Args:
        config: Configuration dictionary
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        label_encoder: Fitted LabelEncoder
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    batch_size = config['data']['batch_size']
    num_workers = config['experiment']['num_workers']
    pin_memory = config['experiment']['pin_memory']
    
    # Transforms
    train_transform = get_train_transforms(config)
    val_transform = get_val_test_transforms(config)
    
    # Datasets
    train_dataset = SkinCancerDataset(
        train_df, config['data']['root_path'], train_transform, label_encoder
    )
    val_dataset = SkinCancerDataset(
        val_df, config['data']['root_path'], val_transform, label_encoder
    )
    test_dataset = SkinCancerDataset(
        test_df, config['data']['root_path'], val_transform, label_encoder, is_test=False
    )
    
    # Sampler for training
    if config['data'].get('use_weighted_sampler', False):
        sampler = get_weighted_sampler(train_df, label_encoder)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            collate_fn=custom_collate_fn
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            collate_fn=custom_collate_fn
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate_fn
    )
    
    logger.info(f"Train loader: {len(train_loader)} batches ({len(train_dataset)} samples)")
    logger.info(f"Val loader: {len(val_loader)} batches ({len(val_dataset)} samples)")
    logger.info(f"Test loader: {len(test_loader)} batches ({len(test_dataset)} samples)")
    
    return train_loader, val_loader, test_loader


def load_and_split_data(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, LabelEncoder]:
    """
    Load data from CSV and create train/val/test splits.
    Uses predefined splits from CSV (leakage-free by lesion_id).
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Tuple of (train_df, val_df, test_df, label_encoder)
    """
    root_path = Path(config['data']['root_path'])
    labels_csv = root_path / config['data']['labels_csv']
    
    logger.info(f"Loading labels from {labels_csv}")
    df = pd.read_csv(labels_csv)
    logger.info(f"Loaded {len(df)} samples")
    
    # Check for predefined splits
    if config['data'].get('use_predefined_splits', True) and 'split' in df.columns:
        logger.info("Using predefined splits from CSV (leakage-free by lesion_id)")
        
        train_df = df[df['split'] == 'train'].copy()
        val_df = df[df['split'] == 'val'].copy()
        test_df = df[df['split'] == 'test'].copy()
        
        # Verify no lesion_id overlap (NaN treated as unique per row)
        train_lesions = _get_valid_lesion_ids(train_df)
        val_lesions = _get_valid_lesion_ids(val_df)
        test_lesions = _get_valid_lesion_ids(test_df)
        
        overlap_train_val = train_lesions & val_lesions
        overlap_train_test = train_lesions & test_lesions
        overlap_val_test = val_lesions & test_lesions
        
        if overlap_train_val or overlap_train_test or overlap_val_test:
            logger.warning("WARNING: Lesion ID overlap detected between splits!")
            logger.warning(f"Train-Val overlap: {len(overlap_train_val)}")
            logger.warning(f"Train-Test overlap: {len(overlap_train_test)}")
            logger.warning(f"Val-Test overlap: {len(overlap_val_test)}")
        else:
            logger.info("Verified: No lesion_id overlap between splits (leakage-free)")
    else:
        # Fallback: create stratified group split by lesion_id
        logger.info("Creating new StratifiedGroupKFold splits...")
        set_seed(config['experiment']['seed'])
        
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config['experiment']['seed'])
        groups = df['lesion_id'].values
        y = df['label'].values
        
        # Get first fold as test (20%)
        for train_val_idx, test_idx in sgkf.split(df, y, groups):
            break
        
        train_val_df = df.iloc[train_val_idx].copy()
        test_df = df.iloc[test_idx].copy()
        
        # Split train_val into train/val (80/20 of remaining)
        train_idx, val_idx = train_test_split(
            range(len(train_val_df)),
            test_size=0.2,
            stratify=train_val_df['label'],
            random_state=config['experiment']['seed']
        )
        
        train_df = train_val_df.iloc[train_idx].copy()
        val_df = train_val_df.iloc[val_idx].copy()
    
    # Fit label encoder on training data
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df['label'])
    
    # Verify all classes present in training
    train_classes = set(train_df['label'].unique())
    all_classes = set(df['label'].unique())
    missing_classes = all_classes - train_classes
    
    if missing_classes:
        logger.warning(f"Classes missing from training set: {missing_classes}")
    
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    logger.info(f"Classes: {label_encoder.classes_}")
    logger.info(f"Train class distribution:\n{train_df['label'].value_counts().sort_index()}")
    
    return train_df, val_df, test_df, label_encoder


def save_label_encoder(label_encoder: LabelEncoder, save_path: Path) -> None:
    """Save LabelEncoder to JSON."""
    mapping = {str(cls): int(idx) for idx, cls in enumerate(label_encoder.classes_)}
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    logger.info(f"Saved label encoder to {save_path}")


def load_label_encoder(load_path: Path) -> LabelEncoder:
    """Load LabelEncoder from JSON."""
    with open(load_path, 'r') as f:
        mapping = json.load(f)
    
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array([k for k, v in sorted(mapping.items(), key=lambda x: x[1])])
    return label_encoder


def _get_valid_lesion_ids(df: pd.DataFrame) -> set:
    """
    Get valid lesion IDs, treating NaN as unique per-row identifiers.
    
    Args:
        df: DataFrame with lesion_id column
    
    Returns:
        Set of valid lesion IDs (NaN replaced with unique placeholders)
    """
    lesion_ids = df['lesion_id'].copy()
    # Replace NaN with unique placeholders per row
    nan_mask = lesion_ids.isna()
    if nan_mask.any():
        lesion_ids = lesion_ids.copy()
        lesion_ids[nan_mask] = [f"__NAN_{idx}__" for idx in lesion_ids[nan_mask].index]
    return set(lesion_ids.unique())


def verify_no_data_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
    """
    Verify no data leakage between splits by checking lesion_id overlap.
    NaN lesion_ids are treated as unique per-row (no leakage across NaNs).
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
    
    Returns:
        True if no leakage, False otherwise
    """
    train_lesions = _get_valid_lesion_ids(train_df)
    val_lesions = _get_valid_lesion_ids(val_df)
    test_lesions = _get_valid_lesion_ids(test_df)
    
    leaks = []
    overlap = train_lesions & val_lesions
    if overlap:
        leaks.append(f"Train-Val: {len(overlap)} lesions")
    overlap = train_lesions & test_lesions
    if overlap:
        leaks.append(f"Train-Test: {len(overlap)} lesions")
    overlap = val_lesions & test_lesions
    if overlap:
        leaks.append(f"Val-Test: {len(overlap)} lesions")
    
    if leaks:
        logger.error(f"DATA LEAKAGE DETECTED: {', '.join(leaks)}")
        return False
    
    logger.info("No data leakage detected: All splits have disjoint lesion_ids")
    return True


if __name__ == "__main__":
    # Test dataset loading
    import yaml
    from utils import setup_logging
    
    setup_logging(Path("test_logs"), "INFO", True)
    
    with open("../configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    train_df, val_df, test_df, label_encoder = load_and_split_data(config)
    verify_no_data_leakage(train_df, val_df, test_df)
    
    train_loader, val_loader, test_loader = create_data_loaders(config, train_df, val_df, test_df, label_encoder)
    
    # Test one batch
    batch = next(iter(train_loader))
    print(f"Batch image shape: {batch['image'].shape}")
    print(f"Batch label shape: {batch['label'].shape}")
    print(f"Sample labels: {batch['label'][:5]}")