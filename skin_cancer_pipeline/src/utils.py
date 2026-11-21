"""
Utility functions for the skin cancer classification pipeline.
Includes seeding, logging, hardware checks, and directory setup.
"""

import os
import sys
import random
import json
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

import torch
import numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device(device_config: str = "auto") -> torch.device:
    """
    Get the appropriate device for training/inference.
    
    Args:
        device_config: "cuda", "cpu", or "auto"
    
    Returns:
        torch.device object
    """
    if device_config == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_config)
    
    if device.type == "cuda" and torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("Using CPU")
    
    return device


def check_memory(device: torch.device) -> Dict[str, float]:
    """
    Check available GPU/CPU memory.
    
    Args:
        device: torch.device to check
    
    Returns:
        Dictionary with memory info in GB
    """
    if device.type == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        free = total - reserved
        return {
            "total_gb": total,
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "free_gb": free
        }
    else:
        import psutil
        mem = psutil.virtual_memory()
        return {
            "total_gb": mem.total / 1e9,
            "available_gb": mem.available / 1e9,
            "used_gb": mem.used / 1e9
        }


def setup_logging(log_dir: Path, log_level: str = "INFO", log_to_file: bool = True) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_dir: Directory to save log files
        log_level: Logging level
        log_to_file: Whether to save logs to file
    
    Returns:
        Configured logger
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("skin_cancer_pipeline")
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"pipeline_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")
    
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and coerce numeric types in configuration.
    Ensures all hyperparameters are proper numeric types (float/int) 
    rather than strings that may result from YAML parsing.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Validated and coerced configuration dictionary
    """
    # Training config
    train_cfg = config.get('training', {})
    
    # Coerce training numeric values
    if 'epochs' in train_cfg:
        train_cfg['epochs'] = int(train_cfg['epochs'])
    if 'learning_rate' in train_cfg:
        train_cfg['learning_rate'] = float(train_cfg['learning_rate'])
    if 'weight_decay' in train_cfg:
        train_cfg['weight_decay'] = float(train_cfg['weight_decay'])
    if 'gradient_accumulation_steps' in train_cfg:
        train_cfg['gradient_accumulation_steps'] = int(train_cfg['gradient_accumulation_steps'])
    if 'gradient_clip' in train_cfg:
        train_cfg['gradient_clip'] = float(train_cfg['gradient_clip'])
    
    # Loss config
    loss_cfg = train_cfg.get('loss', {})
    if 'focal_gamma' in loss_cfg:
        loss_cfg['focal_gamma'] = float(loss_cfg['focal_gamma'])
    if 'focal_alpha' in loss_cfg:
        loss_cfg['focal_alpha'] = float(loss_cfg['focal_alpha'])
    if 'label_smoothing' in loss_cfg:
        loss_cfg['label_smoothing'] = float(loss_cfg['label_smoothing'])
    
    # Scheduler config
    scheduler_params = train_cfg.get('scheduler_params', {})
    if 'T_0' in scheduler_params:
        scheduler_params['T_0'] = int(scheduler_params['T_0'])
    if 'T_mult' in scheduler_params:
        scheduler_params['T_mult'] = int(scheduler_params['T_mult'])
    if 'eta_min' in scheduler_params:
        scheduler_params['eta_min'] = float(scheduler_params['eta_min'])
    
    # Early stopping config
    early_stop_cfg = train_cfg.get('early_stopping', {})
    if 'patience' in early_stop_cfg:
        early_stop_cfg['patience'] = int(early_stop_cfg['patience'])
    if 'min_delta' in early_stop_cfg:
        early_stop_cfg['min_delta'] = float(early_stop_cfg['min_delta'])
    
    # AMP
    if 'amp' in train_cfg:
        train_cfg['amp'] = bool(train_cfg['amp'])
    
    # Experiment config
    exp_cfg = config.get('experiment', {})
    if 'seed' in exp_cfg:
        exp_cfg['seed'] = int(exp_cfg['seed'])
    if 'num_workers' in exp_cfg:
        exp_cfg['num_workers'] = int(exp_cfg['num_workers'])
    if 'pin_memory' in exp_cfg:
        exp_cfg['pin_memory'] = bool(exp_cfg['pin_memory'])
    
    # Data config
    data_cfg = config.get('data', {})
    if 'batch_size' in data_cfg:
        data_cfg['batch_size'] = int(data_cfg['batch_size'])
    if 'image_size' in data_cfg:
        data_cfg['image_size'] = int(data_cfg['image_size'])
    if 'use_class_weights' in data_cfg:
        data_cfg['use_class_weights'] = bool(data_cfg['use_class_weights'])
    if 'use_weighted_sampler' in data_cfg:
        data_cfg['use_weighted_sampler'] = bool(data_cfg['use_weighted_sampler'])
    
    # Augmentation config
    train_aug = data_cfg.get('train_augmentation', {})
    for key in ['horizontal_flip', 'vertical_flip', 'rotate_limit', 'shift_scale_rotate', 
                'brightness_contrast', 'hue_saturation', 'gauss_noise', 'cutout']:
        if key in train_aug:
            train_aug[key] = float(train_aug[key])
    
    # Evaluation config
    eval_cfg = config.get('evaluation', {})
    gradcam_cfg = eval_cfg.get('gradcam', {})
    if 'num_samples_per_class' in gradcam_cfg:
        gradcam_cfg['num_samples_per_class'] = int(gradcam_cfg['num_samples_per_class'])
    if 'alpha' in gradcam_cfg:
        gradcam_cfg['alpha'] = float(gradcam_cfg['alpha'])
    
    return config


def save_config(config: Dict[str, Any], save_path: Path) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save config
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def save_json(data: Dict[str, Any], save_path: Path) -> None:
    """
    Save dictionary to JSON file.
    
    Args:
        data: Data to save
        save_path: Path to save JSON
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4, default=str)


def load_json(load_path: Path) -> Dict[str, Any]:
    """
    Load dictionary from JSON file.
    
    Args:
        load_path: Path to load JSON from
    
    Returns:
        Loaded dictionary
    """
    with open(load_path, 'r') as f:
        return json.load(f)


def create_directories(config: Dict[str, Any]) -> Dict[str, Path]:
    """
    Create all output directories from config.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Dictionary of created directory paths
    """
    output_config = config.get('output', {})
    base_dir = Path(output_config.get('results_dir', 'results'))
    
    dirs = {
        'base': base_dir,
        'checkpoints': base_dir / output_config.get('checkpoints_dir', 'checkpoints'),
        'metrics': base_dir / output_config.get('metrics_dir', 'metrics'),
        'plots': base_dir / output_config.get('plots_dir', 'plots'),
        'gradcam': base_dir / output_config.get('gradcam_dir', 'gradcam'),
        'eda': base_dir / output_config.get('eda_dir', 'eda'),
        'test_logs': base_dir / output_config.get('test_logs_dir', 'test_logs'),
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with total, trainable, and non-trainable parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': non_trainable
    }


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {mins}m {secs:.1f}s"


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping to halt training when metric stops improving."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "max",
        monitor: str = "val_loss"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.monitor = monitor
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
        if mode == "max":
            self.monitor_op = lambda a, b: a > b + min_delta
            self.best_score = -np.inf
        else:
            self.monitor_op = lambda a, b: a < b - min_delta
            self.best_score = np.inf
    
    def __call__(self, score: float, epoch: int) -> bool:
        if self.monitor_op(score, self.best_score):
            self.best_score = score
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def get_class_names(config: Dict[str, Any]) -> List[str]:
    """
    Get class names from dataset or config.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        List of class names in order
    """
    # ISIC 2019 class order (alphabetical by code)
    return ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']


def print_model_summary(model: torch.nn.Module, input_size: tuple = (1, 3, 224, 224)) -> None:
    """
    Print model summary including parameter count and estimated memory.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (batch, channels, height, width)
    """
    param_counts = count_parameters(model)
    
    print("\n" + "=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    print(f"Total Parameters:     {param_counts['total']:,}")
    print(f"Trainable Parameters: {param_counts['trainable']:,}")
    print(f"Non-trainable:        {param_counts['non_trainable']:,}")
    
    # Estimate model size in MB (float32 = 4 bytes per param)
    model_size_mb = param_counts['total'] * 4 / 1e6
    print(f"Model Size (FP32):    {model_size_mb:.2f} MB")
    print(f"Model Size (FP16):    {model_size_mb / 2:.2f} MB")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Test utilities
    set_seed(42)
    device = get_device("auto")
    mem = check_memory(device)
    print(f"Memory: {mem}")
    
    logger = setup_logging(Path("test_logs"), "DEBUG", True)
    logger.info("Test log message")
    logger.warning("Test warning")