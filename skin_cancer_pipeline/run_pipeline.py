#!/usr/bin/env python
"""
Main entrypoint for the skin cancer classification pipeline.
Executes: EDA -> Pytest -> Train -> Evaluate -> Summary Report
"""

import argparse
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils import (
    load_config, set_seed, get_device, setup_logging, create_directories,
    save_json, save_config, print_model_summary, validate_config, get_class_names
)
from dataset import (
    load_and_split_data, create_data_loaders, compute_class_weights,
    save_label_encoder, verify_no_data_leakage
)
from models import create_model, get_model_list, save_checkpoint
from train import train_model
from evaluate import evaluate_model, run_eda


def parse_args():
    parser = argparse.ArgumentParser(description="Skin Cancer Classification Pipeline")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--model", type=str, default=None, help="Model to train (default: all)")
    parser.add_argument("--skip-tests", action="store_true", help="Skip pytest validation")
    parser.add_argument("--skip-eda", action="store_true", help="Skip EDA")
    parser.add_argument("--skip-train", action="store_true", help="Skip training")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation")
    parser.add_argument("--skip-trained", action="store_true", help="Skip training for models that already have best_model.pt checkpoint")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--eval-only", type=str, default=None, help="Only evaluate a trained model")
    return parser.parse_args()


def run_tests(config: Dict[str, Any], logger: logging.Logger) -> bool:
    """Run pytest validation tests."""
    if not config['testing'].get('run_tests_before_training', True):
        logger.info("Skipping tests (disabled in config)")
        return True
    
    logger.info("Running validation tests...")
    
    import subprocess
    test_args = config['testing'].get('pytest_args', '-v --tb=short').split()
    cmd = [sys.executable, '-m', 'pytest', 'tests/', *test_args]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        # Save test output
        test_log_dir = Path(config['output']['results_dir']) / config['output']['test_logs_dir']
        test_log_dir.mkdir(parents=True, exist_ok=True)
        with open(test_log_dir / 'test_report.txt', 'w') as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Return code: {result.returncode}\n")
            f.write(f"STDOUT:\n{result.stdout}\n")
            f.write(f"STDERR:\n{result.stderr}\n")
        
        if result.returncode == 0:
            logger.info("All tests passed!")
            return True
        else:
            logger.error("Tests failed!")
            logger.error(result.stdout)
            logger.error(result.stderr)
            return False
    
    except subprocess.TimeoutExpired:
        logger.error("Tests timed out after 5 minutes")
        return False
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        return False


def train_single_model(
    model_config: Dict[str, Any],
    config: Dict[str, Any],
    train_loader,
    val_loader,
    test_loader,
    class_weights,
    label_encoder,
    dirs,
    device,
    logger,
    resume_path: Optional[Path] = None
) -> Dict[str, Any]:
    """Train a single model and evaluate."""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING MODEL: {model_config['name']} ({model_config['timm_name']})")
    logger.info(f"{'='*60}")
    
    # Create model
    model = create_model(config, model_config).to(device)
    
    # Model-specific checkpoint directory
    model_ckpt_dir = dirs['checkpoints'] / model_config['name']
    model_ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Save label encoder
    save_label_encoder(label_encoder, model_ckpt_dir / 'label_encoder.json')
    
    # Save config
    save_config(config, model_ckpt_dir / 'config.yaml')
    
    # Add class_names to config for training
    config['class_names'] = get_class_names(config)
    
    # Train
    start_time = time.time()
    history = train_model(
        model, train_loader, val_loader, config, class_weights,
        checkpoint_dir=model_ckpt_dir,
        resume_from=resume_path,
        model_name=model_config['name']
    )
    train_time = time.time() - start_time
    
    # Load best model for evaluation
    best_ckpt = model_ckpt_dir / 'best_model.pt'
    if best_ckpt.exists():
        from models import load_checkpoint
        load_checkpoint(model, best_ckpt, device)
        logger.info("Loaded best model for evaluation")
    else:
        logger.warning("No best checkpoint found, using last model")
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    class_names = get_class_names(config)
    test_metrics = evaluate_model(
        model, test_loader, config, device, dirs, class_names
    )
    
    # Save training history
    save_json(history, dirs['metrics'] / f'{model_config["name"]}_history.json')
    
    # Save test metrics
    save_json(test_metrics, dirs['metrics'] / f'{model_config["name"]}_test_metrics.json')
    
    # Plot training curves
    from evaluate import plot_training_curves
    plot_training_curves(history, dirs['plots'] / f'{model_config["name"]}_training_curves.png')
    
    logger.info(f"Model {model_config['name']} completed in {train_time:.1f}s")
    logger.info(f"Test Macro F1: {test_metrics['macro_f1']:.4f}")
    logger.info(f"Test Balanced Acc: {test_metrics['balanced_accuracy']:.4f}")
    
    return {
        'model_name': model_config['name'],
        'timm_name': model_config['timm_name'],
        'train_time': train_time,
        'history': history,
        'test_metrics': test_metrics
    }


def generate_summary_report(results: list, config: Dict[str, Any], dirs: Dict[str, Path], logger: logging.Logger):
    """Generate final summary report comparing all models."""
    
    logger.info("\n" + "=" * 60)
    logger.info("FINAL SUMMARY REPORT")
    logger.info("=" * 60)
    
    summary = {
        'experiment': config['experiment']['name'],
        'seed': config['experiment']['seed'],
        'dataset': 'ISIC 2019 Preprocessed',
        'models': []
    }
    
    # Create comparison table
    print("\n{:<20} {:>10} {:>12} {:>12} {:>12} {:>10}".format(
        'Model', 'Time (s)', 'Test Acc', 'Bal Acc', 'Macro F1', 'ROC-AUC'
    ))
    print("-" * 80)
    
    for r in results:
        m = r['test_metrics']
        print("{:<20} {:>10.1f} {:>12.4f} {:>12.4f} {:>12.4f} {:>10.4f}".format(
            r['model_name'],
            r['train_time'],
            m['accuracy'],
            m['balanced_accuracy'],
            m['macro_f1'],
            m['roc_auc_ovr']
        ))
        
        summary['models'].append({
            'name': r['model_name'],
            'timm_name': r['timm_name'],
            'train_time': r['train_time'],
            'test_accuracy': m['accuracy'],
            'test_balanced_accuracy': m['balanced_accuracy'],
            'test_macro_f1': m['macro_f1'],
            'test_roc_auc_ovr': m['roc_auc_ovr'],
            'test_roc_auc_ovo': m['roc_auc_ovo'],
            'test_pr_auc_macro': m['pr_auc_macro'],
            'per_class': m['per_class']
        })
    
    # Save summary
    save_json(summary, dirs['metrics'] / 'summary_report.json')
    
    # Also save as readable text
    with open(dirs['metrics'] / 'summary_report.txt', 'w') as f:
        f.write("SKIN CANCER CLASSIFICATION - SUMMARY REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Experiment: {config['experiment']['name']}\n")
        f.write(f"Seed: {config['experiment']['seed']}\n")
        f.write(f"Dataset: ISIC 2019 Preprocessed (25,331 images, 8 classes)\n\n")
        f.write("{:<20} {:>10} {:>12} {:>12} {:>12} {:>10}\n".format(
            'Model', 'Time (s)', 'Test Acc', 'Bal Acc', 'Macro F1', 'ROC-AUC'
        ))
        f.write("-" * 80 + "\n")
        
        for r in results:
            m = r['test_metrics']
            f.write("{:<20} {:>10.1f} {:>12.4f} {:>12.4f} {:>12.4f} {:>10.4f}\n".format(
                r['model_name'],
                r['train_time'],
                m['accuracy'],
                m['balanced_accuracy'],
                m['macro_f1'],
                m['roc_auc_ovr']
            ))
        
        f.write("\n\nPer-class metrics:\n")
        for r in results:
            f.write(f"\n--- {r['model_name']} ---\n")
            for cls, metrics in r['test_metrics']['per_class'].items():
                f.write(f"  {cls}: Prec={metrics['precision']:.3f}, "
                        f"Rec={metrics['recall_sensitivity']:.3f}, "
                        f"Spec={metrics['specificity']:.3f}, "
                        f"F1={metrics['f1']:.3f}\n")
    
    logger.info(f"Summary report saved to {dirs['metrics']}/summary_report.json")


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Validate and coerce numeric types in config
    config = validate_config(config)
    
    # Setup directories and logging
    dirs = create_directories(config)
    logger = setup_logging(
        dirs['test_logs'],
        config['output']['log_level'],
        config['output']['log_to_file']
    )
    
    logger.info("=" * 60)
    logger.info("SKIN CANCER CLASSIFICATION PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Config: {args.config}")
    logger.info(f"Results directory: {dirs['base']}")
    
    # Set seed
    set_seed(config['experiment']['seed'])
    
    # Get device
    device = get_device(config['experiment']['device'])
    
    # Save effective config
    save_config(config, dirs['base'] / 'effective_config.yaml')
    
    # Step 1: EDA
    if not args.skip_eda:
        logger.info("\n[1/5] Running Exploratory Data Analysis...")
        run_eda(config, dirs['eda'])
    else:
        logger.info("\n[1/5] Skipping EDA")
    
    # Step 2: Run tests
    if not args.skip_tests:
        logger.info("\n[2/5] Running validation tests...")
        if not run_tests(config, logger):
            logger.error("Tests failed! Aborting pipeline.")
            sys.exit(1)
    else:
        logger.info("\n[2/5] Skipping tests")
    
    # Step 3: Load data
    logger.info("\n[3/5] Loading data...")
    train_df, val_df, test_df, label_encoder = load_and_split_data(config)
    
    # Verify no leakage
    if not verify_no_data_leakage(train_df, val_df, test_df):
        logger.error("Data leakage detected! Aborting.")
        sys.exit(1)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        config, train_df, val_df, test_df, label_encoder
    )
    
    # Compute class weights
    class_weights = compute_class_weights(train_df, label_encoder) if config['data']['use_class_weights'] else None
    if class_weights is not None:
        class_weights = class_weights.to(device)
    
    # Step 4: Train models
    models_to_train = get_model_list(config)
    if args.model:
        models_to_train = [m for m in models_to_train if m['name'] == args.model]
        if not models_to_train:
            logger.error(f"Model '{args.model}' not found in config")
            sys.exit(1)
    
    results = []
    
    if not args.skip_train and not args.eval_only:
        logger.info(f"\n[4/5] Training {len(models_to_train)} model(s)...")
        
        for i, model_config in enumerate(models_to_train):
            logger.info(f"\n--- Model {i+1}/{len(models_to_train)} ---")
            
            # Check if --skip-trained and checkpoint exists
            model_ckpt_dir = dirs['checkpoints'] / model_config['name']
            best_ckpt = model_ckpt_dir / 'best_model.pt'
            
            if args.skip_trained and best_ckpt.exists():
                logger.info(f"Checkpoint found at {best_ckpt}, skipping training (--skip-trained)")
                
                # Load the existing model and evaluate
                model = create_model(config, model_config).to(device)
                from models import load_checkpoint
                load_checkpoint(model, best_ckpt, device)
                logger.info("Loaded existing best model for evaluation")
                
                class_names = get_class_names(config)
                test_metrics = evaluate_model(
                    model, test_loader, config, device, dirs, class_names
                )
                
                # Save test metrics
                save_json(test_metrics, dirs['metrics'] / f'{model_config["name"]}_test_metrics.json')
                
                results.append({
                    'model_name': model_config['name'],
                    'timm_name': model_config['timm_name'],
                    'train_time': 0,
                    'history': {},
                    'test_metrics': test_metrics
                })
                
                logger.info(f"Test Macro F1: {test_metrics['macro_f1']:.4f}")
                logger.info(f"Test Balanced Acc: {test_metrics['balanced_accuracy']:.4f}")
                continue
            
            resume_path = Path(args.resume) if args.resume and i == 0 else None
            
            result = train_single_model(
                model_config, config, train_loader, val_loader, test_loader,
                class_weights, label_encoder, dirs, device, logger, resume_path
            )
            results.append(result)
    
    # Step 5: Evaluate (if eval-only or after training)
    if args.eval_only or (not args.skip_eval and not results):
        logger.info("\n[5/5] Running evaluation only...")
        
        for model_config in models_to_train:
            logger.info(f"Evaluating {model_config['name']}...")
            
            model = create_model(config, model_config).to(device)
            
            # Load checkpoint
            ckpt_path = dirs['checkpoints'] / model_config['name'] / 'best_model.pt'
            if args.eval_only:
                ckpt_path = Path(args.eval_only)
            
            if ckpt_path.exists():
                from models import load_checkpoint
                load_checkpoint(model, ckpt_path, device)
            else:
                logger.warning(f"Checkpoint not found: {ckpt_path}")
                continue
            
            class_names = get_class_names(config)
            test_metrics = evaluate_model(
                model, test_loader, config, device, dirs, class_names
            )
            
            results.append({
                'model_name': model_config['name'],
                'timm_name': model_config['timm_name'],
                'train_time': 0,
                'test_metrics': test_metrics
            })
    
    # Generate summary
    if results:
        generate_summary_report(results, config, dirs, logger)
    
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {dirs['base']}")


if __name__ == "__main__":
    main()