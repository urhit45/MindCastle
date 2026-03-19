#!/usr/bin/env python3
"""
TinyNet Training Script
Trains the TinyNet model on JSONL data and saves checkpoints + ONNX export.
"""

import argparse
import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging

# Add the project root to Python path so we can import app modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.ml.tinynet import TinyNet
from app.ml.train_utils import (
    TinyNetDataset, load_labels_config, load_training_data,
    create_label_mappings, compute_class_weights, split_data,
    compute_metrics, aggregate_epoch_predictions, save_checkpoint,
    export_onnx, setup_logging,
)
from app.ml.fairness import compute_group_metrics, parity_check, generate_fairness_report
from app.ml.audit import write_run_manifest, write_model_card


def train_epoch(model: TinyNet, train_loader: DataLoader, optimizer: optim.Optimizer, 
                device: torch.device, category_to_idx: dict, state_to_idx: dict) -> dict:
    """
    Train for one epoch.
    
    Args:
        model: TinyNet model
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        category_to_idx: Category to index mapping
        state_to_idx: State to index mapping
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    for batch_idx, batch in enumerate(train_loader):
        # Move data to device
        x = batch['x'].to(device)
        cat_target = batch['cat_target'].to(device)
        state_target = batch['state_target'].squeeze(1).to(device)  # Remove extra dimension
        nextstep_target = batch['nextstep_target'].squeeze(1).to(device)
        
        # Forward pass
        optimizer.zero_grad()
        hidden, cat_logits, state_logits, nextstep_logits = model.forward(x)
        
        # Compute loss
        losses = model.compute_losses(
            cat_logits, state_logits, nextstep_logits,
            cat_target, state_target, nextstep_target
        )

        # Accumulate epoch loss (was: total_loss = losses['total'] — overwrote each step)
        batch_loss = losses['total']
        batch_loss.backward()
        optimizer.step()
        total_loss += batch_loss.item()

        # Store predictions and targets for epoch-level metrics
        with torch.no_grad():
            predictions = model.predict(x)
            all_predictions.append(predictions)
            all_targets.append({
                'cat_target': cat_target,
                'state_target': state_target,
            })

        if batch_idx % 10 == 0:
            logging.info(f"Batch {batch_idx}/{len(train_loader)}, Loss: {batch_loss.item():.4f}")

    # Aggregate across all batches (was: only last batch)
    agg_preds, agg_tgts = aggregate_epoch_predictions(all_predictions, all_targets)
    train_metrics = compute_metrics(agg_preds, agg_tgts, category_to_idx, state_to_idx)
    train_metrics['train_loss'] = total_loss / max(len(train_loader), 1)

    return train_metrics


def validate_epoch(model: TinyNet, val_loader: DataLoader, device: torch.device,
                   category_to_idx: dict, state_to_idx: dict) -> dict:
    """
    Validate for one epoch.
    
    Args:
        model: TinyNet model
        val_loader: Validation data loader
        device: Device to validate on
        category_to_idx: Category to index mapping
        state_to_idx: State to index mapping
        
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            # Move data to device
            x = batch['x'].to(device)
            cat_target = batch['cat_target'].to(device)
            state_target = batch['state_target'].squeeze(1).to(device)
            nextstep_target = batch['nextstep_target'].squeeze(1).to(device)
            
            # Forward pass
            hidden, cat_logits, state_logits, nextstep_logits = model.forward(x)
            
            # Compute loss
            losses = model.compute_losses(
                cat_logits, state_logits, nextstep_logits,
                cat_target, state_target, nextstep_target
            )
            
            total_loss += losses['total'].item()
            
            # Store predictions and targets for epoch-level metrics
            predictions = model.predict(x)
            all_predictions.append(predictions)
            all_targets.append({
                'cat_target': cat_target,
                'state_target': state_target,
            })

    # Aggregate across all batches (was: only last batch)
    agg_preds, agg_tgts = aggregate_epoch_predictions(all_predictions, all_targets)
    val_metrics = compute_metrics(agg_preds, agg_tgts, category_to_idx, state_to_idx)
    val_metrics['val_loss'] = total_loss / max(len(val_loader), 1)
    
    return val_metrics


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train TinyNet model")
    
    parser.add_argument(
        "--data",
        type=str,
        default="data/train.jsonl",
        help="Path to training data JSONL file"
    )
    
    parser.add_argument(
        "--out",
        type=str,
        default="runs/exp1",
        help="Output directory for checkpoints and exports"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-3,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay"
    )
    
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=3,
        help="Early stopping patience"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="backend/config/labels.yaml",
        help="Path to labels configuration file"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logging.info(f"Using device: {device}")
    
    # Load configuration
    logging.info("Loading configuration...")
    try:
        categories, states, next_step_templates = load_labels_config(args.config)
        logging.info(f"Loaded {len(categories)} categories, {len(states)} states, {len(next_step_templates)} next step templates")
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Load training data
    logging.info("Loading training data...")
    try:
        data = load_training_data(args.data)
        if len(data) == 0:
            logging.error("No training data found")
            sys.exit(1)
    except Exception as e:
        logging.error(f"Failed to load training data: {e}")
        sys.exit(1)
    
    # Create label mappings
    category_to_idx, state_to_idx = create_label_mappings(categories, states)
    
    # Split data
    logging.info("Splitting data...")
    train_data, val_data = split_data(data, train_ratio=0.85, random_state=args.seed)
    
    # Initialize vectorizer
    logging.info("Initializing vectorizer...")
    from app.ml.vectorizer import HashingVectorizer512
    vectorizer = HashingVectorizer512(use_tfidf=False, seed=args.seed)
    
    # Create datasets
    train_dataset = TinyNetDataset(train_data, vectorizer, category_to_idx, state_to_idx)
    val_dataset = TinyNetDataset(val_data, vectorizer, category_to_idx, state_to_idx)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    logging.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Initialize model
    logging.info("Initializing model...")
    model = TinyNet(config_path=args.config)
    model = model.to(device)
    
    # Compute class weights for categories
    logging.info("Computing class weights...")
    category_weights = compute_class_weights(train_data, category_to_idx)
    category_weights = category_weights.to(device)
    
    # Update model's category loss function with weights
    model.categories_loss = nn.BCEWithLogitsLoss(pos_weight=category_weights)
    
    # Setup optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Training loop
    logging.info("Starting training...")
    best_score = 0.0
    patience_counter = 0
    output_dir = Path(args.out)
    
    for epoch in range(args.epochs):
        logging.info(f"Epoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, category_to_idx, state_to_idx
        )
        
        # Validate
        val_metrics = validate_epoch(
            model, val_loader, device, category_to_idx, state_to_idx
        )
        
        # Log metrics
        logging.info(f"Train - Loss: {train_metrics['train_loss']:.4f}, "
                    f"Cat Micro F1: {train_metrics['cat_micro_f1']:.4f}, "
                    f"State Acc: {train_metrics['state_accuracy']:.4f}")
        
        logging.info(f"Val - Loss: {val_metrics['val_loss']:.4f}, "
                    f"Cat Micro F1: {val_metrics['cat_micro_f1']:.4f}, "
                    f"State Acc: {val_metrics['state_accuracy']:.4f}")
        
        # Check if this is the best model
        current_score = val_metrics['combined_score']
        is_best = current_score > best_score
        
        if is_best:
            best_score = current_score
            patience_counter = 0
            logging.info(f"New best score: {best_score:.4f}")
        else:
            patience_counter += 1
            logging.info(f"No improvement for {patience_counter} epochs")
        
        # Save checkpoint
        save_checkpoint(
            model, optimizer, epoch, val_metrics, output_dir, is_best=is_best
        )
        
        # Early stopping
        if patience_counter >= args.early_stop_patience:
            logging.info(f"Early stopping after {patience_counter} epochs without improvement")
            break
    
    # Export ONNX model
    logging.info("Exporting ONNX model...")
    export_onnx(model, output_dir)

    # ── Phase III: fairness + audit artifacts ──────────────────────────────
    run_id = f"run_{int(time.time())}_seed{args.seed}"

    # Fairness: run full-val inference, compute per-group metrics
    logging.info("Running fairness evaluation on validation set...")
    model.eval()
    all_val_preds: list = []
    all_val_tgts:  list = []
    with torch.no_grad():
        for batch in val_loader:
            xb = batch['x'].to(device)
            all_val_preds.append(model.predict(xb))
            all_val_tgts.append({
                'cat_target':   batch['cat_target'],
                'state_target': batch['state_target'].squeeze(1),
            })

    agg_vp, agg_vt = aggregate_epoch_predictions(all_val_preds, all_val_tgts)
    state_preds_np  = agg_vp['state']['predictions'].cpu().numpy()
    cat_preds_np    = agg_vp['categories']['predictions'].cpu().numpy()
    state_tgts_np   = agg_vt['state_target'].cpu().numpy()
    cat_tgts_np     = agg_vt['cat_target'].cpu().numpy()
    # Use 'group' field if present, else 'default' for all samples
    groups = [item.get('group', 'default') for item in val_data]

    group_metrics = compute_group_metrics(
        state_preds_np, state_tgts_np, cat_preds_np, cat_tgts_np, groups,
    )
    parity = parity_check(group_metrics)
    fairness_report = generate_fairness_report(
        group_metrics, parity, run_id,
        output_path=output_dir / "fairness_report.json",
    )
    if not fairness_report["overall_pass"]:
        logging.warning("FAIRNESS ALERT: parity check failed — %s", parity)
    else:
        logging.info("Fairness check passed.")

    # Run manifest
    labels_path = Path(args.config)
    if not labels_path.is_absolute():
        labels_path = project_root / labels_path
    write_run_manifest(
        run_id=run_id,
        data_path=Path(args.data),
        labels_path=labels_path,
        model_version="0.1.0",
        seed=args.seed,
        thresholds={"abstain_state": 0.35, "abstain_cat": 0.25, "defer_state": 0.50},
        metrics={"best_val_score": best_score, **val_metrics},
        output_path=output_dir / "run_manifest.json",
    )

    # Model card (idempotent)
    write_model_card(output_dir / "model_card.json")

    # Final summary
    logging.info("Training completed!")
    logging.info(f"Best validation score: {best_score:.4f}")
    logging.info(f"Checkpoints saved to: {output_dir}")
    logging.info(f"ONNX model saved to: {output_dir / 'tinynet.onnx'}")
    logging.info(f"Run manifest: {output_dir / 'run_manifest.json'}")
    logging.info(f"Fairness report: {output_dir / 'fairness_report.json'}")


if __name__ == "__main__":
    main()
