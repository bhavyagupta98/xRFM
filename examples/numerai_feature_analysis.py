"""
Numerai dataset training and feature weight analysis for xRFM model.

This script loads the Numerai training data from ~/Desktop/kaggle/numerai_training_data.csv,
trains an xRFM model, and prints feature weights at nodes to understand what's happening
in the model.

Run with e.g.
    python examples/numerai_feature_analysis.py --train-size 50000 --val-size 10000 --test-size 10000
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from xrfm import xRFM


DEFAULT_TOTAL_TRAIN = 50_000
DEFAULT_TOTAL_VAL = 10_000
DEFAULT_TOTAL_TEST = 10_000
DEFAULT_MIN_SUBSET_SIZE = 20_000
DEFAULT_DATA_PATH = os.path.expanduser("~/Desktop/kaggle/numerai_training_data.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="Train xRFM on Numerai dataset and analyze feature weights.")
    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH,
                        help="Path to Numerai training data CSV file.")
    parser.add_argument("--train-size", type=int, default=DEFAULT_TOTAL_TRAIN,
                        help="Number of training samples to use.")
    parser.add_argument("--val-size", type=int, default=DEFAULT_TOTAL_VAL,
                        help="Number of validation samples to use.")
    parser.add_argument("--test-size", type=int, default=DEFAULT_TOTAL_TEST,
                        help="Number of test samples to use.")
    parser.add_argument("--min-subset-size", type=int, default=DEFAULT_MIN_SUBSET_SIZE,
                        help="Minimum subset size passed to xRFM.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for NumPy shuffling.")
    return parser.parse_args()


def _compute_split_sizes(n_available: int, train: int, val: int, test: int):
    desired_total = train + val + test
    if desired_total <= n_available:
        return train, val, test

    print(
        f"Requested {desired_total} total samples but only {n_available} are available after preprocessing. "
        "Using proportional split."
    )

    train_ratio = train / desired_total
    val_ratio = val / desired_total
    test_ratio = test / desired_total

    adjusted_train = max(1, int(round(train_ratio * n_available)))
    adjusted_val = max(1, int(round(val_ratio * n_available)))
    adjusted_test = n_available - adjusted_train - adjusted_val
    if adjusted_test <= 0:
        adjusted_test = 1
        if adjusted_val > 1:
            adjusted_val -= 1
        if adjusted_train + adjusted_val + adjusted_test > n_available and adjusted_train > 1:
            adjusted_train -= 1

    # Ensure the counts sum exactly to the available samples.
    diff = (adjusted_train + adjusted_val + adjusted_test) - n_available
    if diff > 0 and adjusted_train > diff:
        adjusted_train -= diff
    elif diff > 0 and adjusted_val > diff:
        adjusted_val -= diff

    return adjusted_train, adjusted_val, adjusted_test


def prepare_data(args, device: torch.device):
    """Load and prepare Numerai dataset."""
    data_path = Path(args.data_path).expanduser()
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Separate features and target
    feature_cols = [col for col in df.columns if col.startswith('feature')]
    if 'target' not in df.columns:
        raise ValueError("Target column 'target' not found in dataset.")
    
    X = df[feature_cols].values.astype(np.float32)
    y = df['target'].values.astype(np.int64)
    
    print(f"Loaded {len(X)} samples with {len(feature_cols)} features")
    print(f"Target distribution: {np.bincount(y)}")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_array = np.ascontiguousarray(X_scaled, dtype=np.float32)
    
    total_available = len(X_array)
    train_count, val_count, test_count = _compute_split_sizes(
        total_available, args.train_size, args.val_size, args.test_size
    )
    total_required = train_count + val_count + test_count
    
    if total_required > total_available:
        raise ValueError("Unable to allocate requested splits after adjustment.")
    
    rng = np.random.default_rng(seed=args.seed)
    indices = rng.choice(total_available, size=total_required, replace=False)
    X_subset = X_array[indices]
    y_subset = y[indices]
    
    X_train = torch.from_numpy(X_subset[:train_count]).to(device)
    y_train = torch.from_numpy(y_subset[:train_count]).to(device)
    
    val_start = train_count
    val_end = train_count + val_count
    X_val = torch.from_numpy(X_subset[val_start:val_end]).to(device)
    y_val = torch.from_numpy(y_subset[val_start:val_end]).to(device)
    
    X_test = torch.from_numpy(X_subset[val_end:]).to(device)
    y_test = torch.from_numpy(y_subset[val_end:]).to(device)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, train_count, feature_cols


def build_model(device: torch.device, min_subset_size: int):
    """Build xRFM model with default parameters."""
    rfm_params = {
        "model": {
            "kernel": "l2",
            "exponent": 1.0,
            "bandwidth": 10.0,
            "diag": False,
            "bandwidth_mode": "adaptive",
        },
        "fit": {
            "reg": 1e-3,
            "iters": 3,
            "early_stop_rfm": True,
        },
    }

    model = xRFM(
        rfm_params=rfm_params,
        device=device,
        min_subset_size=min_subset_size,
        tuning_metric="accuracy",
        split_method="top_vector_agop_on_subset",
        split_temperature=None,
        overlap_fraction=0.1,
    )
    return model


def print_node_weights(node, depth=0, feature_names=None, node_id=0):
    """
    Recursively print feature weights at each node.
    
    For internal nodes: prints split_direction (feature importance for splitting)
    For leaf nodes: prints model weights and Mahalanobis matrix diagonal (feature importance)
    """
    indent = "  " * depth
    node_type = node['type']
    
    if node_type == 'node':
        split_dir = node.get('split_direction')
        split_point = node.get('split_point', 'N/A')
        
        if split_dir is not None:
            # Convert to numpy if it's a tensor
            if isinstance(split_dir, torch.Tensor):
                split_dir = split_dir.cpu().numpy()
            
            # Get absolute values and sort by importance
            abs_weights = np.abs(split_dir)
            top_indices = np.argsort(abs_weights)[::-1][:10]  # Top 10 features
            
            print(f"{indent}Node {node_id} (depth {depth}): Internal Split Node")
            print(f"{indent}  Split point: {split_point:.4f}")
            print(f"{indent}  Top 10 features by split direction magnitude:")
            for idx in top_indices:
                feat_name = feature_names[idx] if feature_names else f"feature{idx+1}"
                print(f"{indent}    {feat_name}: {split_dir[idx]:.6f} (abs: {abs_weights[idx]:.6f})")
            
            # Recursively process children
            print_node_weights(node['left'], depth + 1, feature_names, node_id * 2 + 1)
            print_node_weights(node['right'], depth + 1, feature_names, node_id * 2 + 2)
        else:
            print(f"{indent}Node {node_id} (depth {depth}): Internal Node (no split_direction)")
            if 'left' in node:
                print_node_weights(node['left'], depth + 1, feature_names, node_id * 2 + 1)
            if 'right' in node:
                print_node_weights(node['right'], depth + 1, feature_names, node_id * 2 + 2)
    
    elif node_type == 'leaf':
        model = node.get('model')
        train_indices = node.get('train_indices', [])
        n_samples = len(train_indices) if hasattr(train_indices, '__len__') else 0
        
        print(f"{indent}Node {node_id} (depth {depth}): Leaf Node")
        print(f"{indent}  Training samples: {n_samples}")
        
        if model is not None:
            # Get model weights (alpha coefficients)
            weights = model.weights
            if weights is not None:
                if isinstance(weights, torch.Tensor):
                    weights = weights.cpu().numpy()
                
                # For classification, weights might be 2D (one per class)
                if weights.ndim == 2:
                    # Average across classes or take first class
                    weights = weights[:, 0] if weights.shape[1] > 0 else weights.mean(axis=1)
                
                print(f"{indent}  Model weights shape: {weights.shape}")
                print(f"{indent}  Model weights stats: min={weights.min():.6f}, max={weights.max():.6f}, mean={weights.mean():.6f}, std={weights.std():.6f}")
            
            # Get Mahalanobis matrix diagonal (feature importance)
            M = model.M
            if M is not None:
                if isinstance(M, torch.Tensor):
                    M = M.cpu().numpy()
                
                # Extract diagonal if it's a matrix
                if M.ndim == 2:
                    M_diag = np.diag(M)
                else:
                    M_diag = M
                
                # Get top features by Mahalanobis diagonal
                abs_M = np.abs(M_diag)
                top_indices = np.argsort(abs_M)[::-1][:10]  # Top 10 features
                
                print(f"{indent}  Mahalanobis matrix diagonal (feature importance):")
                print(f"{indent}    Top 10 features:")
                for idx in top_indices:
                    feat_name = feature_names[idx] if feature_names else f"feature{idx+1}"
                    print(f"{indent}      {feat_name}: {M_diag[idx]:.6f} (abs: {abs_M[idx]:.6f})")
                
                print(f"{indent}    All features: {M_diag}")
            
            # Get bandwidth
            if hasattr(model, 'kernel_obj') and hasattr(model.kernel_obj, 'bandwidth'):
                print(f"{indent}  Kernel bandwidth: {model.kernel_obj.bandwidth:.4f}")
        else:
            print(f"{indent}  No model found in leaf node")


def analyze_model_weights(model, feature_names):
    """Analyze and print feature weights across all nodes in the model."""
    print("\n" + "="*80)
    print("FEATURE WEIGHT ANALYSIS")
    print("="*80)
    
    if model.trees is None or len(model.trees) == 0:
        print("No trees found in model. Model may not be fitted yet.")
        return
    
    print(f"\nNumber of trees: {len(model.trees)}")
    
    for tree_idx, tree in enumerate(model.trees):
        print(f"\n{'='*80}")
        print(f"TREE {tree_idx + 1}")
        print(f"{'='*80}")
        print_node_weights(tree, depth=0, feature_names=feature_names, node_id=0)
    
    # Summary statistics across all leaf nodes
    print(f"\n{'='*80}")
    print("SUMMARY: Leaf Node Statistics")
    print(f"{'='*80}")
    
    all_leaf_nodes = []
    for tree in model.trees:
        leaf_nodes = model._collect_leaf_nodes(tree)
        all_leaf_nodes.extend(leaf_nodes)
    
    print(f"Total number of leaf nodes: {len(all_leaf_nodes)}")
    
    # Collect Mahalanobis diagonals from all leaves
    all_M_diags = []
    for leaf_node in all_leaf_nodes:
        model_obj = leaf_node.get('model')
        if model_obj is not None and model_obj.M is not None:
            M = model_obj.M
            if isinstance(M, torch.Tensor):
                M = M.cpu().numpy()
            if M.ndim == 2:
                M_diag = np.diag(M)
            else:
                M_diag = M
            all_M_diags.append(M_diag)
    
    if all_M_diags:
        # Average feature importance across all leaves
        avg_M_diag = np.mean(all_M_diags, axis=0)
        abs_avg_M = np.abs(avg_M_diag)
        top_indices = np.argsort(abs_avg_M)[::-1]
        
        print(f"\nAverage feature importance (Mahalanobis diagonal) across all {len(all_M_diags)} leaves:")
        print("Top features:")
        for idx in top_indices[:10]:
            feat_name = feature_names[idx] if feature_names else f"feature{idx+1}"
            print(f"  {feat_name}: {avg_M_diag[idx]:.6f} (abs: {abs_avg_M[idx]:.6f})")


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        train_count,
        feature_names,
    ) = prepare_data(args, device)
    y_test_numpy = y_test.cpu().numpy()

    min_subset_size = min(args.min_subset_size, train_count)
    if min_subset_size <= 0:
        raise ValueError("min_subset_size must be positive after adjustment.")

    model = build_model(device, min_subset_size=min_subset_size)

    print(f"\nFitting xRFM on Numerai dataset...")
    print(f"Training samples: {train_count}, Validation samples: {len(X_val)}, Test samples: {len(X_test)}")
    print(f"Features: {len(feature_names)}")
    
    start_time = time.time()
    model.fit(X_train, y_train, X_val, y_val)
    fit_time = time.time() - start_time
    print(f"Training completed in {fit_time:.2f} seconds.")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    start_time = time.time()
    preds = model.predict(X_test)
    predict_time = time.time() - start_time

    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()

    accuracy = (preds == y_test_numpy).mean()
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Prediction time: {predict_time:.2f} seconds")

    # Analyze feature weights
    analyze_model_weights(model, feature_names)


if __name__ == "__main__":
    main()

