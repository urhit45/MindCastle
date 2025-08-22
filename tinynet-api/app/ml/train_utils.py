"""
Training utilities for TinyNet
"""

import json
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

from .vectorizer import HashingVectorizer512
from .tinynet import TinyNet


class TinyNetDataset(Dataset):
    """Dataset class for TinyNet training data"""
    
    def __init__(self, data: List[Dict], vectorizer: HashingVectorizer512, 
                 category_to_idx: Dict[str, int], state_to_idx: Dict[str, int]):
        """
        Initialize dataset.
        
        Args:
            data: List of data dictionaries
            vectorizer: HashingVectorizer512 instance
            category_to_idx: Mapping from category names to indices
            state_to_idx: Mapping from state names to indices
        """
        self.data = data
        self.vectorizer = vectorizer
        self.category_to_idx = category_to_idx
        self.state_to_idx = state_to_idx
        self.num_categories = len(category_to_idx)
        self.num_states = len(state_to_idx)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Vectorize text
        text = item['text']
        vector = self.vectorizer.encode(text)
        
        # Convert to tensor - ensure consistent shape (batch_size, 512)
        x = torch.FloatTensor(vector)  # (512,) - will be batched by DataLoader
        
        # Process categories (multi-label)
        categories = item.get('categories', [])
        cat_target = torch.zeros(self.num_categories)
        for cat in categories:
            if cat in self.category_to_idx:
                cat_target[self.category_to_idx[cat]] = 1.0
        
        # Process state (single-label) - ensure consistent shape
        state = item.get('state', 'continue')
        state_idx = self.state_to_idx.get(state, 0)  # Default to 0 if unknown
        state_target = torch.LongTensor([state_idx])  # (1,) - will be batched by DataLoader
        
        # For now, use dummy next_step target (label 0) - ensure consistent shape
        nextstep_target = torch.LongTensor([0])  # (1,) - will be batched by DataLoader
        
        # Only return tensor fields that can be properly batched
        return {
            'x': x,
            'cat_target': cat_target,
            'state_target': state_target,
            'nextstep_target': nextstep_target
        }


def load_labels_config(config_path: str = "backend/config/labels.yaml") -> Tuple[List[str], List[str], List[str]]:
    """
    Load labels configuration from YAML file.
    
    Args:
        config_path: Path to labels.yaml
        
    Returns:
        Tuple of (categories, states, next_step_templates)
    """
    try:
        # Try relative path first
        config_file = Path(config_path)
        if not config_file.exists():
            # Try from project root
            config_file = Path(__file__).parent.parent.parent / config_path
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                categories = config.get('categories', [])
                states = config.get('states', [])
                next_step_templates = config.get('next_step_templates', [])
                return categories, states, next_step_templates
        else:
            raise FileNotFoundError(f"Config file not found at {config_path}")
            
    except Exception as e:
        raise RuntimeError(f"Error loading config: {e}")


def load_training_data(data_path: str) -> List[Dict]:
    """
    Load training data from JSONL file.
    
    Args:
        data_path: Path to JSONL file
        
    Returns:
        List of data dictionaries
    """
    data = []
    with open(data_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    data.append(item)
            except json.JSONDecodeError as e:
                logging.warning(f"Invalid JSON at line {line_num}: {e}")
                continue
    
    logging.info(f"Loaded {len(data)} training samples from {data_path}")
    return data


def create_label_mappings(categories: List[str], states: List[str]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Create label to index mappings.
    
    Args:
        categories: List of category names
        states: List of state names
        
    Returns:
        Tuple of (category_to_idx, state_to_idx)
    """
    category_to_idx = {cat: idx for idx, cat in enumerate(categories)}
    state_to_idx = {state: idx for idx, state in enumerate(states)}
    
    logging.info(f"Created mappings: {len(categories)} categories, {len(states)} states")
    return category_to_idx, state_to_idx


def compute_class_weights(data: List[Dict], category_to_idx: Dict[str, int]) -> torch.Tensor:
    """
    Compute positive class weights for categories based on training frequencies.
    
    Args:
        data: Training data
        category_to_idx: Category to index mapping
        
    Returns:
        Tensor of positive weights for BCEWithLogitsLoss
    """
    num_categories = len(category_to_idx)
    category_counts = torch.zeros(num_categories)
    
    for item in data:
        categories = item.get('categories', [])
        for cat in categories:
            if cat in category_to_idx:
                category_counts[category_to_idx[cat]] += 1
    
    # Compute positive weights (inverse frequency)
    total_samples = len(data)
    positive_weights = total_samples / (2 * category_counts + 1e-8)  # Add small epsilon
    
    logging.info(f"Computed positive weights for {num_categories} categories")
    return positive_weights


def split_data(data: List[Dict], train_ratio: float = 0.85, random_state: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """
    Split data into train/validation sets.
    
    Args:
        data: Full dataset
        train_ratio: Ratio for training set
        random_state: Random seed
        
    Returns:
        Tuple of (train_data, val_data)
    """
    # Try to stratify by presence of any category
    try:
        # Create stratification labels (presence of any category)
        stratify_labels = []
        for item in data:
            has_category = len(item.get('categories', [])) > 0
            stratify_labels.append(1 if has_category else 0)
        
        train_data, val_data = train_test_split(
            data, 
            train_size=train_ratio, 
            random_state=random_state,
            stratify=stratify_labels
        )
        logging.info(f"Stratified split: {len(train_data)} train, {len(val_data)} val")
        
    except Exception as e:
        logging.warning(f"Stratified split failed, using random split: {e}")
        train_data, val_data = train_test_split(
            data, 
            train_size=train_ratio, 
            random_state=random_state
        )
        logging.info(f"Random split: {len(train_data)} train, {len(val_data)} val")
    
    return train_data, val_data


def compute_metrics(predictions: Dict[str, Any], targets: Dict[str, torch.Tensor], 
                   category_to_idx: Dict[str, int], state_to_idx: Dict[str, int]) -> Dict[str, float]:
    """
    Compute training metrics.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        category_to_idx: Category to index mapping
        state_to_idx: State to index mapping
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Categories: micro/macro F1
    cat_preds = predictions['categories']['predictions'].cpu().numpy()
    cat_targets = targets['cat_target'].cpu().numpy()
    
    # Flatten for sklearn metrics
    cat_preds_flat = cat_preds.flatten()
    cat_targets_flat = cat_targets.flatten()
    
    # Only compute F1 if we have positive samples
    if np.sum(cat_targets_flat) > 0:
        metrics['cat_micro_f1'] = f1_score(cat_targets_flat, cat_preds_flat, average='micro', zero_division=0)
        metrics['cat_macro_f1'] = f1_score(cat_targets_flat, cat_preds_flat, average='macro', zero_division=0)
    else:
        metrics['cat_micro_f1'] = 0.0
        metrics['cat_macro_f1'] = 0.0
    
    # State: accuracy
    state_preds = predictions['state']['predictions'].cpu().numpy()
    state_targets = targets['state_target'].cpu().numpy()
    
    metrics['state_accuracy'] = accuracy_score(state_targets, state_preds)
    
    # Combined metric for early stopping (micro F1 + state accuracy)
    metrics['combined_score'] = metrics['cat_micro_f1'] + metrics['state_accuracy']
    
    return metrics


def save_checkpoint(model: TinyNet, optimizer: torch.optim.Optimizer, epoch: int, 
                   metrics: Dict[str, float], save_path: Path, is_best: bool = False):
    """
    Save model checkpoint.
    
    Args:
        model: TinyNet model
        optimizer: Optimizer
        epoch: Current epoch
        metrics: Training metrics
        save_path: Directory to save checkpoint
        is_best: Whether this is the best model so far
    """
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save PyTorch checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'model_info': model.get_model_info()
    }
    
    checkpoint_path = save_path / 'checkpoint.pt'
    torch.save(checkpoint, checkpoint_path)
    
    if is_best:
        best_path = save_path / 'best.pt'
        torch.save(checkpoint, best_path)
        logging.info(f"Saved best model to {best_path}")
    
    # Save metrics
    metrics_path = save_path / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logging.info(f"Saved checkpoint to {checkpoint_path}")


def export_onnx(model: TinyNet, save_path: Path, input_shape: Tuple[int, int] = (1, 512)):
    """
    Export model to ONNX format for inference.
    
    Args:
        model: TinyNet model
        save_path: Directory to save ONNX file
        input_shape: Input tensor shape for ONNX export
    """
    try:
        model.eval()
        dummy_input = torch.randn(input_shape)
        
        onnx_path = save_path / 'tinynet.onnx'
        
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['hidden', 'cat_logits', 'state_logits', 'nextstep_logits'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'hidden': {0: 'batch_size'},
                'cat_logits': {0: 'batch_size'},
                'state_logits': {0: 'batch_size'},
                'nextstep_logits': {0: 'batch_size'}
            }
        )
        
        logging.info(f"Exported ONNX model to {onnx_path}")
        
    except Exception as e:
        logging.error(f"Failed to export ONNX: {e}")


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
