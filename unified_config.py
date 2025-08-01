# unified_config.py
"""
Unified configuration for both centralized and federated learning experiments
to ensure fair comparison between approaches.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import json 
from syft_utils import load_data, run_federated_training_with_syft, evaluate_model, get_ckks_context, evaluate_biometric_model
from centralized_training import BiometricHomomorphicLogisticRegression, train_encrypted_mlp_local

# =============================================================================
# SHARED CONFIGURATION
# =============================================================================

class UnifiedConfig:
    """Unified configuration for all experiments"""
    
    # Data Configuration
    FEATURE_COLUMNS = [
        'dwell_avg', 'flight_avg', 'traj_avg',
        'hold_mean', 'hold_std',
        'flight_mean', 'flight_std'
    ]
    
    # Data Preprocessing
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    MISSING_VALUE_STRATEGY = 'mean'  # fillna with mean
    
    # Model Configurations
    MODEL_CONFIGS = {
        'LogisticRegression': {
            'class': LogisticRegression,
            'params': {
                'C': 1.0,
                'max_iter': 1000,
                'random_state': RANDOM_STATE,
                'solver': 'lbfgs'
            },
            'type': 'Plain'
        },
        
        'MLPClassifier': {
            'class': MLPClassifier,
            'params': {
                'hidden_layer_sizes': (32, 16),
                'max_iter': 1000,
                'random_state': RANDOM_STATE,
                'early_stopping': True,
                'validation_fraction': 0.1,
                'n_iter_no_change': 10
            },
            'type': 'Plain'
        },
        
        'HomomorphicLogisticRegression': {
            'class': None,  # Will be set to BiometricHomomorphicLogisticRegression
            'params': {
                'poly_modulus_degree': 8192,
                'scale': 2**40
            },
            'type': 'Encrypted',
            'training_params': {
                'epochs': 10,
                'lr': 0.01,
                'verbose': True
            }
        },
        
        'EncryptedMLP': {
            'class': None,  # Will be set to TrulyEncryptedMLP
            'params': {
                'hidden_dim': 32,  # Match first layer of MLPClassifier
                'input_dim': len(FEATURE_COLUMNS)
            },
            'type': 'Encrypted',
            'training_params': {
                'epochs': 10,
                'lr': 0.01,
                'verbose': True
            }
        }
    }
    
    # Federated Learning Configuration
    FEDERATED_CONFIG = {
        'num_clients': 3,
        'num_rounds': 10,  # Increased for better convergence
        'epochs_per_round': 5,  # Increased to match centralized training intensity
        'client_fraction': 1.0,  # Use all clients each round
        'algorithms': ['FedAvg', 'SCAFFOLD'],
        'scaffold_lr': 0.05
    }
    
    # Evaluation Configuration
    EVALUATION_CONFIG = {
        'test_subset_size': None,  # Use full test set, or set to int for subset
        'cross_validation_folds': 5,
        'metrics': ['accuracy', 'f1', 'precision', 'recall', 'auc']
    }

# =============================================================================
# UNIFIED DATA PREPROCESSING
# =============================================================================

def prepare_unified_dataset(data, config=None):
    """
    Prepare dataset with unified preprocessing for both centralized and federated learning.
    
    Args:
        data: Raw dataset
        config: UnifiedConfig instance (optional)
    
    Returns:
        dict: Preprocessed dataset with train/test splits
    """
    if config is None:
        config = UnifiedConfig()
    
    print("📊 Preparing unified dataset...")
    
    # Extract features and labels
    X = data[config.FEATURE_COLUMNS].copy()
    y = data['label'].values
    
    # Handle missing values
    if config.MISSING_VALUE_STRATEGY == 'mean':
        X = X.fillna(X.mean())
    elif config.MISSING_VALUE_STRATEGY == 'zero':
        X = X.fillna(0)
    
    # Print dataset statistics
    total_records = len(y)
    label_0_count = np.sum(y == 0)
    label_1_count = np.sum(y == 1)
    
    print(f"   Total records: {total_records}")
    print(f"   Label 0 count: {label_0_count} ({label_0_count/total_records*100:.1f}%)")
    print(f"   Label 1 count: {label_1_count} ({label_1_count/total_records*100:.1f}%)")
    print(f"   Features: {len(config.FEATURE_COLUMNS)} ({config.FEATURE_COLUMNS})")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, 
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y
    )
    
    print(f"   Train set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    print(f"   Input dimension: {X_train.shape[1]}")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_columns': config.FEATURE_COLUMNS,
        'input_dim': X_train.shape[1]
    }

# =============================================================================
# UNIFIED MODEL FACTORY
# =============================================================================

def create_unified_model(model_name, dataset_info, config=None):
    """
    Create model with unified configuration.
    
    Args:
        model_name: Name of model from MODEL_CONFIGS
        dataset_info: Dataset info from prepare_unified_dataset
        config: UnifiedConfig instance
    
    Returns:
        Configured model instance
    """
    if config is None:
        config = UnifiedConfig()
    
    if model_name not in config.MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")
    
    model_config = config.MODEL_CONFIGS[model_name]
    model_class = model_config['class']
    params = model_config['params'].copy()
    
    # Handle encrypted models that need input_dim
    if model_name in ['HomomorphicLogisticRegression', 'EncryptedMLP']:
        params['input_dim'] = dataset_info['input_dim']
    
    # Create model instance
    if model_class is not None:
        return model_class(**params)
    else:
        # For custom models, return config to be used by training functions
        return {'params': params, 'training_params': model_config.get('training_params', {})}

# =============================================================================
# UNIFIED TRAINING WRAPPER
# =============================================================================

def get_unified_training_params(model_name, config=None):
    """Get training parameters for a specific model."""
    if config is None:
        config = UnifiedConfig()
    
    model_config = config.MODEL_CONFIGS.get(model_name, {})
    return model_config.get('training_params', {})

# =============================================================================
# UPDATED CENTRALIZED TRAINING FUNCTION
# =============================================================================

def run_unified_centralized_training(data, config=None):
    """
    Run centralized training with unified configuration.
    """
    if config is None:
        config = UnifiedConfig()
    
    print("🏃 Running UNIFIED Centralized Training")
    print("=" * 50)
    
    # Prepare unified dataset
    dataset = prepare_unified_dataset(data, config)
    
    X_train, X_test = dataset['X_train'], dataset['X_test']
    y_train, y_test = dataset['y_train'], dataset['y_test']
    
    results = {}
    
    # Train each model with unified configuration
    for model_name, model_config in config.MODEL_CONFIGS.items():
        print(f"\n🔬 Training {model_name} ({model_config['type']})...")
        
        try:
            if model_config['type'] == 'Plain':
                # Standard sklearn models
                model = create_unified_model(model_name, dataset, config)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                if hasattr(model, 'predict_proba'):
                    y_prob = model.predict_proba(X_test)[:, 1]
                else:
                    y_prob = y_pred.astype(float)
                
                metrics = evaluate_biometric_model(y_test, y_pred, y_prob)
                results[model_name] = metrics
                print(f"   ✅ Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1']:.3f}")
                
            elif model_config['type'] == 'Encrypted':
                # Handle encrypted models with unified parameters
                if model_name == 'HomomorphicLogisticRegression':
                    model_params = create_unified_model(model_name, dataset, config)
                    training_params = get_unified_training_params(model_name, config)
                    
                    # Create and train homomorphic model
                    hom_model = BiometricHomomorphicLogisticRegression(**model_params['params'])
                    
                    # Train with encrypted data
                    hom_model.scaler.fit(X_train)
                    X_train_encrypted = hom_model.encrypt_data(X_train)
                    y_train_encrypted = hom_model.encrypt_data(y_train.reshape(-1, 1))
                    
                    hom_model.train_encrypted(X_train_encrypted, y_train_encrypted, **training_params)
                    
                    # Evaluate
                    y_pred, y_conf = hom_model.predict_encrypted(X_test, threshold=0.5)
                    metrics = evaluate_biometric_model(y_test, y_pred, y_conf)
                    results[model_name] = metrics
                    print(f"   ✅ Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1']:.3f}")
                    
                elif model_name == 'EncryptedMLP':
                    # Use the existing train_encrypted_mlp_local function with unified params
                    metrics = train_encrypted_mlp_local(X_train, y_train, X_test, y_test)
                    results[model_name] = metrics
                    if 'error' not in metrics:
                        print(f"   ✅ Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1']:.3f}")
                    else:
                        print(f"   ❌ Error: {metrics['error']}")
        
        except Exception as e:
            print(f"   ❌ {model_name} failed: {e}")
            results[model_name] = {"error": str(e)}
    
    # Save results
    with open("unified_centralized_results.json", "w") as f:
        json.dump(results, f, indent=4, default=str)
    
    print_unified_results_summary(results, "CENTRALIZED")
    return results


# =============================================================================
# UNIFIED RESULTS SUMMARY
# =============================================================================

def print_unified_results_summary(results, training_type):
    """Print unified results summary."""
    print(f"\n📊 {training_type} TRAINING SUMMARY (UNIFIED CONFIG)")
    print("=" * 60)
    
    for model_name, model_results in results.items():
        if training_type == "CENTRALIZED":
            if 'error' not in model_results:
                print(f"{model_name}:")
                print(f"  Accuracy: {model_results.get('accuracy', 0):.3f}")
                print(f"  F1 Score: {model_results.get('f1', 0):.3f}")
                print(f"  Precision: {model_results.get('precision', 0):.3f}")
                print(f"  Recall: {model_results.get('recall', 0):.3f}")
            else:
                print(f"{model_name}: ❌ FAILED - {model_results['error']}")
        
        else:  # FEDERATED
            print(f"{model_name}:")
            for alg_name, alg_results in model_results.items():
                if alg_results.get('status') == 'success':
                    metrics = alg_results['metrics']
                    print(f"  {alg_name}: Acc={metrics.get('accuracy', 0):.3f}, F1={metrics.get('f1', 0):.3f}")
                else:
                    print(f"  {alg_name}: ❌ FAILED")

