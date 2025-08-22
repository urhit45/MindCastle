#!/usr/bin/env python3
"""
TinyNet Training Script
Trains the TinyNet model on JSONL data and saves checkpoints + ONNX export.
"""

import argparse
import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import logging

# Add the project root to Python path so we can import app modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.ml.tinynet import TinyNet
from app.ml.train_utils import (
    TinyNetDataset, load_labels_config, load_training_data,
    create_label_mappings, compute_class_weights, split_data,
    compute_metrics, save_checkpoint, export_onnx, setup_logging
)


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
        
        total_loss = losses['total']
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Store predictions and targets for metrics
        with torch.no_grad():
            predictions = model.predict(x)
            all_predictions.append(predictions)
            all_targets.append({
                'cat_target': cat_target,
                'state_target': state_target,
                'nextstep_target': nextstep_target
            })
        
        if batch_idx % 10 == 0:
            logging.info(f"Batch {batch_idx}/{len(train_loader)}, Loss: {total_loss.item():.4f}")
    
    # Compute training metrics
    train_metrics = compute_metrics(
        all_predictions[-1], all_targets[-1], category_to_idx, state_to_idx
    )
    train_metrics['train_loss'] = total_loss.item()
    
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
            
            # Store predictions and targets for metrics
            predictions = model.predict(x)
            all_predictions.append(predictions)
            all_targets.append({
                'cat_target': cat_target,
                'state_target': state_target,
                'nextstep_target': nextstep_target
            })
    
    # Compute validation metrics
    val_metrics = compute_metrics(
        all_predictions[-1], all_targets[-1], category_to_idx, state_to_idx
    )
    val_metrics['val_loss'] = total_loss / len(val_loader)
    
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
    
    # Final summary
    logging.info("Training completed!")
    logging.info(f"Best validation score: {best_score:.4f}")
    logging.info(f"Checkpoints saved to: {output_dir}")
    logging.info(f"ONNX model saved to: {output_dir / 'tinynet.onnx'}")


if __name__ == "__main__":
    main()
