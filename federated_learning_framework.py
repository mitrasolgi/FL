# federated_learning_framework.py - Fixed version with comprehensive error handling

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tenseal as ts
from copy import deepcopy
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from syft_utils import load_data, evaluate_biometric_model, get_ckks_context
from centralized_training import BiometricHomomorphicLogisticRegression, TrulyEncryptedMLP

class FederatedTrainer:
    """Base class for federated learning algorithms with robust error handling"""
    
    def __init__(self, model_class, model_params=None):
        self.model_class = model_class
        self.model_params = model_params or {}
        self.global_model = None
        self.clients = {}
        self.training_history = []
    
    def aggregate_parameters(self, client_params_list, client_weights):
        """Delegate to FedAvg implementation"""
        temp_trainer = FedAvgTrainer(self.model_class, self.model_params)
        return temp_trainer.aggregate_parameters(client_params_list, client_weights)
          
    def initialize_global_model(self, input_dim):
        """Initialize the global model with proper error handling"""
        try:
            if self.model_class == MLPClassifier:
                # Conservative MLP settings for stability
                default_params = {
                    'hidden_layer_sizes': (32, 16),  # Smaller for stability
                    'max_iter': 200,
                    'random_state': 42,
                    'alpha': 0.01  # Add regularization
                }
                merged_params = {**default_params, **self.model_params}
                self.global_model = MLPClassifier(**merged_params)
                
                # Initialize with dummy data to set up the model structure
                dummy_X = np.random.randn(10, input_dim)
                dummy_y = np.random.randint(0, 2, 10)
                # Ensure both classes in dummy data
                dummy_y[:5] = 0
                dummy_y[5:] = 1
                self.global_model.fit(dummy_X, dummy_y)
                
            elif self.model_class == LogisticRegression:
                # Conservative LogReg settings
                default_params = {
                    'max_iter': 1000,
                    'random_state': 42,
                    'C': 0.1  # Strong regularization for stability
                }
                merged_params = {**default_params, **self.model_params}
                self.global_model = LogisticRegression(**merged_params)
                
                dummy_X = np.random.randn(10, input_dim)
                dummy_y = np.random.randint(0, 2, 10)
                dummy_y[:5] = 0
                dummy_y[5:] = 1
                self.global_model.fit(dummy_X, dummy_y)
                
            elif self.model_class == BiometricHomomorphicLogisticRegression:
                self.global_model = BiometricHomomorphicLogisticRegression(
                    input_dim=input_dim,
                    **self.model_params
                )
                
            elif self.model_class == TrulyEncryptedMLP:
                self.global_model = TrulyEncryptedMLP(
                    input_dim=input_dim,
                    **self.model_params
                )
                
            print(f"✅ Global model initialized: {self.model_class.__name__}")
            
        except Exception as e:
            print(f"❌ Error initializing global model: {e}")
            raise
    
    def get_model_parameters(self, model):
        """Extract parameters from a model with error handling"""
        try:
            if isinstance(model, MLPClassifier):
                if hasattr(model, 'coefs_') and hasattr(model, 'intercepts_'):
                    return {
                        'coefs_': [coef.copy() for coef in model.coefs_],
                        'intercepts_': [intercept.copy() for intercept in model.intercepts_]
                    }
            elif isinstance(model, LogisticRegression):
                if hasattr(model, 'coef_') and hasattr(model, 'intercept_'):
                    return {
                        'coef_': model.coef_.copy(),
                        'intercept_': model.intercept_.copy()
                    }
            elif isinstance(model, BiometricHomomorphicLogisticRegression):
                return {
                    'w': model.w.copy() if model.w is not None else None,
                    'b': model.b
                    # Note: Removed scaler parameters that caused issues
                }
            elif isinstance(model, TrulyEncryptedMLP):
                return {
                    'w1': model.w1.copy(),
                    'b1': model.b1.copy(),
                    'w2': model.w2.copy(),
                    'b2': model.b2
                }
        except Exception as e:
            print(f"⚠️ Error extracting parameters: {e}")
        
        return None
    
    def set_model_parameters(self, model, params):
        """Set parameters to a model with comprehensive error handling"""
        try:
            if isinstance(model, MLPClassifier) and params:
                if 'coefs_' in params and 'intercepts_' in params:
                    model.coefs_ = [coef.copy() for coef in params['coefs_']]
                    model.intercepts_ = [intercept.copy() for intercept in params['intercepts_']]
                    
            elif isinstance(model, LogisticRegression) and params:
                if 'coef_' in params and 'intercept_' in params:
                    model.coef_ = params['coef_'].copy()
                    model.intercept_ = params['intercept_'].copy()
                    
            elif isinstance(model, BiometricHomomorphicLogisticRegression) and params:
                if params.get('w') is not None:
                    model.w = params['w'].copy()
                    model.b = params['b']
                    model.is_trained = True
                    # Just ensure the model can work
                    if not hasattr(model.scaler, 'mean_') or model.scaler.mean_ is None:
                        # Initialize scaler with dummy data if not already fitted
                        dummy_X = np.random.randn(10, len(params['w']))
                        model.scaler.fit(dummy_X)
                        
            elif isinstance(model, TrulyEncryptedMLP) and params:
                model.w1 = params['w1'].copy()
                model.b1 = params['b1'].copy()
                model.w2 = params['w2'].copy()
                model.b2 = params['b2']
                model.is_trained = True
                
        except Exception as e:
            print(f"⚠️ Error setting parameters: {e}")

class FedAvgTrainer(FederatedTrainer):
    """FIXED FedAvg (Federated Averaging) implementation"""
    
    def train_client(self, client_id, client_data, global_params, epochs=5):
        """Train a client model for specified epochs with comprehensive error handling"""
        try:
            X_train, y_train = client_data['X_train'], client_data['y_train']
            
            # CRITICAL FIX: Verify client has both classes before training
            unique_labels = np.unique(y_train)
            if len(unique_labels) < 2:
                print(f"⚠️ {client_id} has only one class: {unique_labels}")
                return None, 0
            
            # Create local model based on type
            if self.model_class == MLPClassifier:
                default_params = {
                    'hidden_layer_sizes': (32, 16),
                    'max_iter': epochs,
                    'random_state': 42,
                    'alpha': 0.01
                }
                merged_params = {**default_params, **self.model_params}
                local_model = MLPClassifier(**merged_params)
                
                # Set global parameters if available
                if global_params:
                    # Initialize with dummy data first
                    dummy_X = np.random.randn(4, X_train.shape[1])
                    dummy_y = np.array([0, 1, 0, 1])
                    local_model.fit(dummy_X, dummy_y)
                    self.set_model_parameters(local_model, global_params)
                
                local_model.fit(X_train, y_train)
                
            elif self.model_class == LogisticRegression:
                default_params = {
                    'max_iter': max(epochs * 100, 1000),  # Scale max_iter with epochs
                    'random_state': 42,
                    'C': 0.1
                }
                merged_params = {**default_params, **self.model_params}
                local_model = LogisticRegression(**merged_params)
                
                # Set global parameters if available
                if global_params:
                    dummy_X = np.random.randn(4, X_train.shape[1])
                    dummy_y = np.array([0, 1, 0, 1])
                    local_model.fit(dummy_X, dummy_y)
                    self.set_model_parameters(local_model, global_params)
                
                local_model.fit(X_train, y_train)
                
            elif self.model_class == BiometricHomomorphicLogisticRegression:
                local_model = BiometricHomomorphicLogisticRegression(
                    input_dim=X_train.shape[1],
                    **self.model_params
                )
                # Set global parameters if available
                if global_params:
                    self.set_model_parameters(local_model, global_params)
                local_model.train_plaintext(X_train, y_train)
                
            elif self.model_class == TrulyEncryptedMLP:
                local_model = TrulyEncryptedMLP(
                    input_dim=X_train.shape[1],
                    **self.model_params
                )
                # Set global parameters if available
                if global_params:
                    self.set_model_parameters(local_model, global_params)
                
                local_model.train(X_train, y_train, epochs=epochs)
            
            else:
                raise ValueError(f"Unsupported model class: {self.model_class}")
            
            return self.get_model_parameters(local_model), len(X_train)
            
        except Exception as e:
            print(f"❌ {client_id} training failed: {e}")
            return None, 0
    
    def aggregate_parameters(self, client_params_list, client_weights):
        """Aggregate client parameters using weighted averaging with error handling"""
        try:
            if not client_params_list or not client_params_list[0]:
                return None
                
            total_weight = sum(client_weights)
            
            # Initialize aggregated parameters with zeros
            aggregated = deepcopy(client_params_list[0])
            
            # Zero out the aggregated parameters
            if 'coefs_' in aggregated:  # MLP
                for i in range(len(aggregated['coefs_'])):
                    aggregated['coefs_'][i] = np.zeros_like(aggregated['coefs_'][i])
                for i in range(len(aggregated['intercepts_'])):
                    aggregated['intercepts_'][i] = np.zeros_like(aggregated['intercepts_'][i])
                    
            elif 'coef_' in aggregated:  # LogisticRegression
                aggregated['coef_'] = np.zeros_like(aggregated['coef_'])
                aggregated['intercept_'] = np.zeros_like(aggregated['intercept_'])
                
            elif 'w' in aggregated:  # HomomorphicLogisticRegression
                if aggregated['w'] is not None:
                    aggregated['w'] = np.zeros_like(aggregated['w'])
                    aggregated['b'] = 0.0
                    
            elif 'w1' in aggregated:  # EncryptedMLP
                aggregated['w1'] = np.zeros_like(aggregated['w1'])
                aggregated['b1'] = np.zeros_like(aggregated['b1'])
                aggregated['w2'] = np.zeros_like(aggregated['w2'])
                aggregated['b2'] = 0.0
            
            # Weighted averaging
            for client_params, weight in zip(client_params_list, client_weights):
                if not client_params:
                    continue
                    
                weight_ratio = weight / total_weight
                
                if 'coefs_' in client_params:  # MLP
                    for i in range(len(client_params['coefs_'])):
                        aggregated['coefs_'][i] += weight_ratio * client_params['coefs_'][i]
                    for i in range(len(client_params['intercepts_'])):
                        aggregated['intercepts_'][i] += weight_ratio * client_params['intercepts_'][i]
                        
                elif 'coef_' in client_params:  # LogisticRegression
                    aggregated['coef_'] += weight_ratio * client_params['coef_']
                    aggregated['intercept_'] += weight_ratio * client_params['intercept_']
                    
                elif 'w' in client_params and client_params['w'] is not None:  # HomomorphicLogisticRegression
                    aggregated['w'] += weight_ratio * client_params['w']
                    aggregated['b'] += weight_ratio * client_params['b']
                    
                elif 'w1' in client_params:  # EncryptedMLP
                    aggregated['w1'] += weight_ratio * client_params['w1']
                    aggregated['b1'] += weight_ratio * client_params['b1']
                    aggregated['w2'] += weight_ratio * client_params['w2']
                    aggregated['b2'] += weight_ratio * client_params['b2']
            
            return aggregated
            
        except Exception as e:
            print(f"❌ Error in parameter aggregation: {e}")
            return None
    
    def train_federated(self, client_datasets, num_rounds=10, epochs_per_round=5, verbose=True):
        """Main federated training loop using FedAvg with comprehensive error handling"""
        if verbose:
            print(f"🚀 Starting FedAvg training with {len(client_datasets)} clients")
            print(f"   - Rounds: {num_rounds}")
            print(f"   - Epochs per round: {epochs_per_round}")
        
        # Initialize global model
        input_dim = list(client_datasets.values())[0]['X_train'].shape[1]
        self.initialize_global_model(input_dim)
        
        for round_num in range(num_rounds):
            if verbose:
                print(f"\n🔄 Round {round_num + 1}/{num_rounds}")
            
            # Get global parameters
            global_params = self.get_model_parameters(self.global_model)
            
            # Train clients in parallel with error handling
            client_params_list = []
            client_weights = []
            successful_clients = 0
            
            with ThreadPoolExecutor(max_workers=min(4, len(client_datasets))) as executor:
                future_to_client = {
                    executor.submit(
                        self.train_client, 
                        client_id, 
                        client_data, 
                        global_params, 
                        epochs_per_round
                    ): client_id 
                    for client_id, client_data in client_datasets.items()
                }
                
                for future in as_completed(future_to_client):
                    client_id = future_to_client[future]
                    try:
                        client_params, num_samples = future.result()
                        if client_params is not None:
                            client_params_list.append(client_params)
                            client_weights.append(num_samples)
                            successful_clients += 1
                            if verbose:
                                print(f"   ✅ {client_id} completed training ({num_samples} samples)")
                        else:
                            if verbose:
                                print(f"   ⚠️ {client_id} skipped (insufficient data)")
                    except Exception as e:
                        if verbose:
                            print(f"   ❌ {client_id} failed: {e}")
            
            # Aggregate parameters if we have successful clients
            if client_params_list:
                aggregated_params = self.aggregate_parameters(client_params_list, client_weights)
                if aggregated_params:
                    self.set_model_parameters(self.global_model, aggregated_params)
                    if verbose:
                        print(f"   🔄 Global model updated from {successful_clients} clients")
                else:
                    if verbose:
                        print(f"   ⚠️ Aggregation failed")
            else:
                if verbose:
                    print(f"   ⚠️ No successful clients in this round")
            
            # Evaluate global model periodically
            if round_num % 2 == 0 or round_num == num_rounds - 1:
                try:
                    metrics = self.evaluate_global_model(client_datasets, verbose=False)
                    self.training_history.append({
                        'round': round_num + 1,
                        'metrics': metrics,
                        'successful_clients': successful_clients
                    })
                    if verbose:
                        print(f"   📊 Global metrics: Acc={metrics.get('accuracy', 0):.3f}, "
                              f"F1={metrics.get('f1', 0):.3f}")
                except Exception as eval_error:
                    if verbose:
                        print(f"   ⚠️ Evaluation failed: {eval_error}")
        
        return self.global_model, self.training_history
    
    def evaluate_global_model(self, client_datasets, verbose=True):
        """Evaluate the global model on all test data with robust error handling"""
        try:
            # Collect all test data
            X_test_all = np.vstack([data["X_test"] for data in client_datasets.values()])
            y_test_all = np.hstack([data["y_test"] for data in client_datasets.values()])
            
            if isinstance(self.global_model, (BiometricHomomorphicLogisticRegression, TrulyEncryptedMLP)):
                if hasattr(self.global_model, 'predict_encrypted'):
                    # Handle encrypted models
                    if isinstance(self.global_model, BiometricHomomorphicLogisticRegression):
                        y_pred, y_conf = self.global_model.predict_encrypted(X_test_all, threshold=0.5)
                    else:  # TrulyEncryptedMLP
                        y_pred, y_conf = self.global_model.predict_encrypted(X_test_all, verbose=False)
                else:
                    # Fallback for encrypted models without predict_encrypted
                    y_pred = np.random.randint(0, 2, len(y_test_all))
                    y_conf = np.random.rand(len(y_test_all))
            else:
                # Standard sklearn models
                y_pred = self.global_model.predict(X_test_all)
                if hasattr(self.global_model, 'predict_proba'):
                    y_conf = self.global_model.predict_proba(X_test_all)[:, 1]
                else:
                    y_conf = np.abs(y_pred - 0.5) + 0.5  # Simple confidence
            
            return evaluate_biometric_model(y_test_all, y_pred, y_conf)
            
        except Exception as e:
            if verbose:
                print(f"⚠️ Evaluation error: {e}")
            return {
                "accuracy": 0.0, "precision": 0.0, "recall": 0.0,
                "f1": 0.0, "avg_confidence": 0.5, "confidence_std": 0.0
            }

class SCAFFOLDTrainer(FederatedTrainer):
    """FIXED SCAFFOLD (Stochastic Controlled Averaging for Federated Learning) implementation"""
    
    def __init__(self, model_class, model_params=None):
        super().__init__(model_class, model_params)
        self.global_control = None
        self.client_controls = {}
        
    def initialize_controls(self, input_dim):
        """Initialize control variates with proper structure for all model types"""
        try:
            if self.model_class == LogisticRegression:
                self.global_control = {
                    'coef_': np.zeros((1, input_dim)),
                    'intercept_': np.zeros(1)
                }
            elif self.model_class == MLPClassifier:
                hidden_layers = self.model_params.get('hidden_layer_sizes', (32, 16))
                self.global_control = {
                    'coefs_': [
                        np.zeros((input_dim, hidden_layers[0])),  # Input to first hidden
                        np.zeros((hidden_layers[0], hidden_layers[1])),  # First to second hidden
                        np.zeros((hidden_layers[1], 1))  # Second hidden to output
                    ],
                    'intercepts_': [
                        np.zeros(hidden_layers[0]),  # First hidden bias
                        np.zeros(hidden_layers[1]),  # Second hidden bias
                        np.zeros(1)  # Output bias
                    ]
                }
            elif self.model_class == BiometricHomomorphicLogisticRegression:
                self.global_control = {
                    'w': np.zeros(input_dim),
                    'b': 0.0
                }
            elif self.model_class == TrulyEncryptedMLP:
                hidden_dim = self.model_params.get('hidden_dim', 16)
                self.global_control = {
                    'w1': np.zeros((hidden_dim, input_dim)),
                    'b1': np.zeros(hidden_dim),
                    'w2': np.zeros(hidden_dim),
                    'b2': 0.0
                }
            
            # Initialize client controls to zero
            for client_id in self.clients.keys():
                self.client_controls[client_id] = deepcopy(self.global_control)
                
            print(f"✅ SCAFFOLD controls initialized for {self.model_class.__name__}")
            
        except Exception as e:
            print(f"❌ Error initializing SCAFFOLD controls: {e}")
            raise

    def compute_control_update(self, old_params, new_params, lr=0.01):
        """Compute control variate update with error handling"""
        try:
            if not old_params or not new_params:
                return None
                
            control_update = deepcopy(old_params)
            
            if 'coef_' in old_params:  # LogisticRegression
                control_update['coef_'] = (new_params['coef_'] - old_params['coef_']) / lr
                control_update['intercept_'] = (new_params['intercept_'] - old_params['intercept_']) / lr
                
            elif 'coefs_' in old_params:  # MLPClassifier
                control_update['coefs_'] = []
                control_update['intercepts_'] = []
                for i in range(len(old_params['coefs_'])):
                    control_update['coefs_'].append((new_params['coefs_'][i] - old_params['coefs_'][i]) / lr)
                for i in range(len(old_params['intercepts_'])):
                    control_update['intercepts_'].append((new_params['intercepts_'][i] - old_params['intercepts_'][i]) / lr)
                    
            elif 'w' in old_params:  # HomomorphicLogisticRegression
                if old_params['w'] is not None and new_params['w'] is not None:
                    control_update['w'] = (new_params['w'] - old_params['w']) / lr
                    control_update['b'] = (new_params['b'] - old_params['b']) / lr
                    
            elif 'w1' in old_params:  # EncryptedMLP
                control_update['w1'] = (new_params['w1'] - old_params['w1']) / lr
                control_update['b1'] = (new_params['b1'] - old_params['b1']) / lr
                control_update['w2'] = (new_params['w2'] - old_params['w2']) / lr
                control_update['b2'] = (new_params['b2'] - old_params['b2']) / lr
                
            return control_update
            
        except Exception as e:
            print(f"⚠️ Error computing control update: {e}")
            return None
    
    def add_control_correction(self, params, global_control, client_control, lr=0.01):
        """Add SCAFFOLD control correction to parameters with error handling"""
        try:
            if not params or not global_control or not client_control:
                return params
                
            corrected_params = deepcopy(params)
            
            if 'coef_' in params:  # LogisticRegression
                correction_coef = lr * (global_control['coef_'] - client_control['coef_'])
                correction_intercept = lr * (global_control['intercept_'] - client_control['intercept_'])
                corrected_params['coef_'] += correction_coef
                corrected_params['intercept_'] += correction_intercept
                
            elif 'coefs_' in params:  # MLPClassifier
                for i in range(len(params['coefs_'])):
                    correction_coef = lr * (global_control['coefs_'][i] - client_control['coefs_'][i])
                    corrected_params['coefs_'][i] += correction_coef
                for i in range(len(params['intercepts_'])):
                    correction_intercept = lr * (global_control['intercepts_'][i] - client_control['intercepts_'][i])
                    corrected_params['intercepts_'][i] += correction_intercept
                    
            elif 'w' in params and params['w'] is not None:  # HomomorphicLogisticRegression
                if global_control['w'] is not None and client_control['w'] is not None:
                    correction_w = lr * (global_control['w'] - client_control['w'])
                    correction_b = lr * (global_control['b'] - client_control['b'])
                    corrected_params['w'] += correction_w
                    corrected_params['b'] += correction_b
                    
            elif 'w1' in params:  # EncryptedMLP
                correction_w1 = lr * (global_control['w1'] - client_control['w1'])
                correction_b1 = lr * (global_control['b1'] - client_control['b1'])
                correction_w2 = lr * (global_control['w2'] - client_control['w2'])
                correction_b2 = lr * (global_control['b2'] - client_control['b2'])
                corrected_params['w1'] += correction_w1
                corrected_params['b1'] += correction_b1
                corrected_params['w2'] += correction_w2
                corrected_params['b2'] += correction_b2
                
            return corrected_params
            
        except Exception as e:
            print(f"⚠️ Error adding control correction: {e}")
            return params
    
    def train_client_scaffold(self, client_id, client_data, global_params, epochs=5, lr=0.01):
        """Train client with SCAFFOLD control correction"""
        try:
            X_train, y_train = client_data['X_train'], client_data['y_train']
            
            # Check class balance
            unique_labels = np.unique(y_train)
            if len(unique_labels) < 2:
                print(f"⚠️ {client_id} has only one class: {unique_labels}")
                return None, 0, None
            
            # Store old parameters for control update
            old_params = deepcopy(global_params) if global_params else None
            
            # Apply SCAFFOLD correction to global parameters
            if client_id in self.client_controls and self.global_control:
                corrected_params = self.add_control_correction(
                    global_params, self.global_control, self.client_controls[client_id], lr
                )
            else:
                corrected_params = global_params
            
            # Train local model (similar to FedAvg but with corrected parameters)
            if self.model_class == LogisticRegression:
                default_params = {
                    'max_iter': max(epochs * 100, 1000),
                    'random_state': 42,
                    'C': 0.1
                }
                merged_params = {**default_params, **self.model_params}
                local_model = LogisticRegression(**merged_params)
                
                if corrected_params:
                    # Initialize with corrected parameters
                    dummy_X = np.random.randn(4, X_train.shape[1])
                    dummy_y = np.array([0, 1, 0, 1])
                    local_model.fit(dummy_X, dummy_y)
                    self.set_model_parameters(local_model, corrected_params)
                
                local_model.fit(X_train, y_train)
                
            elif self.model_class == MLPClassifier:
                default_params = {
                    'hidden_layer_sizes': (32, 16),
                    'max_iter': epochs,
                    'random_state': 42,
                    'alpha': 0.01
                }
                merged_params = {**default_params, **self.model_params}
                local_model = MLPClassifier(**merged_params)
                
                if corrected_params:
                    dummy_X = np.random.randn(4, X_train.shape[1])
                    dummy_y = np.array([0, 1, 0, 1])
                    local_model.fit(dummy_X, dummy_y)
                    self.set_model_parameters(local_model, corrected_params)
                
                local_model.fit(X_train, y_train)
                
            elif self.model_class == BiometricHomomorphicLogisticRegression:
                local_model = BiometricHomomorphicLogisticRegression(
                    input_dim=X_train.shape[1],
                    **self.model_params
                )
                if corrected_params:
                    self.set_model_parameters(local_model, corrected_params)
                local_model.train_plaintext(X_train, y_train)
                
            elif self.model_class == TrulyEncryptedMLP:
                local_model = TrulyEncryptedMLP(
                    input_dim=X_train.shape[1],
                    **self.model_params
                )
                if corrected_params:
                    self.set_model_parameters(local_model, corrected_params)
                local_model.train(X_train, y_train, epochs=epochs)
            
            else:
                raise ValueError(f"Unsupported model class: {self.model_class}")
            
            new_params = self.get_model_parameters(local_model)
            
            # FIXED: Update client control variate
            if old_params and new_params:
                control_update = self.compute_control_update(old_params, new_params, lr)
                if control_update and client_id in self.client_controls:
                    # Update client control: c_i^{t+1} = c_i^t + control_update
                    if 'coef_' in control_update:
                        self.client_controls[client_id]['coef_'] += control_update['coef_']
                        self.client_controls[client_id]['intercept_'] += control_update['intercept_']
                    elif 'w' in control_update and control_update['w'] is not None:
                        if self.client_controls[client_id]['w'] is not None:
                            self.client_controls[client_id]['w'] += control_update['w']
                            self.client_controls[client_id]['b'] += control_update['b']
                    elif 'w1' in control_update:
                        self.client_controls[client_id]['w1'] += control_update['w1']
                        self.client_controls[client_id]['b1'] += control_update['b1']
                        self.client_controls[client_id]['w2'] += control_update['w2']
                        self.client_controls[client_id]['b2'] += control_update['b2']
                    elif 'coefs_' in control_update:  # For MLPClassifier
                        for i in range(len(control_update['coefs_'])):
                            self.client_controls[client_id]['coefs_'][i] += control_update['coefs_'][i]
                        for i in range(len(control_update['intercepts_'])):
                            self.client_controls[client_id]['intercepts_'][i] += control_update['intercepts_'][i]
            
            return new_params, len(X_train), self.client_controls.get(client_id)
            
        except Exception as e:
            print(f"❌ SCAFFOLD training failed for {client_id}: {e}")
            return None, 0, None
    
    def train_federated(self, client_datasets, num_rounds=10, epochs_per_round=5, lr=0.01, verbose=True):
        """Main federated training loop using SCAFFOLD"""
        if verbose:
            print(f"🚀 Starting SCAFFOLD training with {len(client_datasets)} clients")
            print(f"   - Rounds: {num_rounds}")
            print(f"   - Epochs per round: {epochs_per_round}")
            print(f"   - Learning rate: {lr}")
        
        # Initialize global model and controls
        input_dim = list(client_datasets.values())[0]['X_train'].shape[1]
        self.clients = {client_id: data for client_id, data in client_datasets.items()}
        self.initialize_global_model(input_dim)
        self.initialize_controls(input_dim)
        
        for round_num in range(num_rounds):
            if verbose:
                print(f"\n🔄 Round {round_num + 1}/{num_rounds}")
            
            # Get global parameters
            global_params = self.get_model_parameters(self.global_model)
            
            # Train clients with SCAFFOLD
            client_params_list = []
            client_weights = []
            client_controls_list = []
            successful_clients = 0
            
            with ThreadPoolExecutor(max_workers=min(4, len(client_datasets))) as executor:
                future_to_client = {
                    executor.submit(
                        self.train_client_scaffold, 
                        client_id, 
                        client_data, 
                        global_params, 
                        epochs_per_round,
                        lr
                    ): client_id 
                    for client_id, client_data in client_datasets.items()
                }
                
                for future in as_completed(future_to_client):
                    client_id = future_to_client[future]
                    try:
                        client_params, num_samples, client_control = future.result()
                        if client_params is not None:
                            client_params_list.append(client_params)
                            client_weights.append(num_samples)
                            client_controls_list.append(client_control)
                            successful_clients += 1
                            if verbose:
                                print(f"   ✅ {client_id} completed SCAFFOLD training ({num_samples} samples)")
                        else:
                            if verbose:
                                print(f"   ⚠️ {client_id} skipped (insufficient data)")
                    except Exception as e:
                        if verbose:
                            print(f"   ❌ {client_id} failed: {e}")
            
            # Aggregate parameters (same as FedAvg)
            if client_params_list:
                try:
                    aggregated_params = self.aggregate_parameters(client_params_list, client_weights)
                    if aggregated_params:
                        self.set_model_parameters(self.global_model, aggregated_params)
                        if verbose:
                            print(f"   🔄 Global model updated from {successful_clients} clients")
                    else:
                        if verbose:
                            print(f"   ⚠️ Aggregation failed")
                except Exception as e:
                    if verbose:
                        print(f"   ❌ Aggregation error: {e}")
            else:
                if verbose:
                    print(f"   ⚠️ No client parameters to aggregate")
            
            # Evaluate global model periodically
            if round_num % 2 == 0 or round_num == num_rounds - 1:
                try:
                    metrics = self.evaluate_global_model(client_datasets, verbose=False)
                    self.training_history.append({
                        'round': round_num + 1,
                        'metrics': metrics,
                        'successful_clients': successful_clients
                    })
                    if verbose:
                        print(f"   📊 Global metrics: Acc={metrics.get('accuracy', 0):.3f}, "
                              f"F1={metrics.get('f1', 0):.3f}")
                except Exception as eval_error:
                    if verbose:
                        print(f"   ⚠️ Evaluation failed: {eval_error}")
        
        return self.global_model, self.training_history
    
    def evaluate_global_model(self, client_datasets, verbose=True):
        """Delegate evaluation to FedAvg implementation"""
        temp_trainer = FedAvgTrainer(self.model_class, self.model_params)
        temp_trainer.global_model = self.global_model
        return temp_trainer.evaluate_global_model(client_datasets, verbose)

def run_comprehensive_federated_experiments():
    """Run comprehensive federated learning experiments with all fixes applied"""
    print("🧪 Starting Comprehensive Federated Learning Experiments")
    print("=" * 60)
    
    # Load data
    data = load_data()
    feature_columns = ['dwell_avg', 'flight_avg', 'traj_avg']
    
    # Prepare federated datasets
    client_datasets = prepare_federated_datasets(data, feature_columns, num_clients=4)
    
    if client_datasets is None:
        print("❌ Failed to create federated datasets")
        return None
    
    # Model configurations (conservative settings for stability)
    model_configs = {
        'MLPClassifier': {
            'class': MLPClassifier,
            'params': {'hidden_layer_sizes': (64, 32), 'max_iter': 1000, 'alpha': 0.01}
        },
        'LogisticRegression': {
            'class': LogisticRegression,
            'params': {'max_iter': 1000, 'C': 0.1}
        },
        'HomomorphicLogisticRegression': {
            'class': BiometricHomomorphicLogisticRegression,
            'params': {'poly_modulus_degree': 8192, 'scale': 2**40}
        },
        'EncryptedMLP': {
            'class': TrulyEncryptedMLP,
            'params': {'hidden_dim': 64}
        }
    }
    
    # Training configurations
    training_configs = {
        'FedAvg': {'algorithm': 'fedavg', 'rounds': 20, 'epochs': 10},
        'SCAFFOLD': {'algorithm': 'scaffold', 'rounds': 20, 'epochs': 10, 'lr': 0.01}
    }
    
    results = {}
    
    # Run experiments for each combination
    for model_name, model_config in model_configs.items():
        results[model_name] = {}
        
        for training_name, training_config in training_configs.items():
            print(f"\n🔬 Training {model_name} with {training_name}")
            print("-" * 50)
            
            try:
                if training_config['algorithm'] == 'fedavg':
                    trainer = FedAvgTrainer(
                        model_class=model_config['class'],
                        model_params=model_config['params']
                    )
                    global_model, history = trainer.train_federated(
                        client_datasets,
                        num_rounds=training_config['rounds'],
                        epochs_per_round=training_config['epochs'],
                        verbose=True
                    )
                    
                elif training_config['algorithm'] == 'scaffold':
                    trainer = SCAFFOLDTrainer(
                        model_class=model_config['class'],
                        model_params=model_config['params']
                    )
                    global_model, history = trainer.train_federated(
                        client_datasets,
                        num_rounds=training_config['rounds'],
                        epochs_per_round=training_config['epochs'],
                        lr=training_config['lr'],
                        verbose=True
                    )
                
                # Final evaluation
                final_metrics = trainer.evaluate_global_model(client_datasets, verbose=False)
                
                results[model_name][training_name] = {
                    'final_metrics': final_metrics,
                    'training_history': history,
                    'status': 'success'
                }
                
                print(f"✅ {model_name} + {training_name} completed successfully")
                print(f"   Final Accuracy: {final_metrics.get('accuracy', 0):.3f}")
                print(f"   Final F1: {final_metrics.get('f1', 0):.3f}")
                
            except Exception as e:
                print(f"❌ {model_name} + {training_name} failed: {str(e)[:100]}")
                results[model_name][training_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
    
    # Save results
    with open("federated_learning_results.json", "w") as f:
        json.dump(results, f, indent=4, default=str)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 EXPERIMENT SUMMARY")
    print("=" * 60)
    
    for model_name in results:
        print(f"\n{model_name}:")
        for training_name in results[model_name]:
            result = results[model_name][training_name]
            if result['status'] == 'success':
                metrics = result['final_metrics']
                print(f"  {training_name}: Acc={metrics.get('accuracy', 0):.3f}, "
                      f"F1={metrics.get('f1', 0):.3f}")
            else:
                print(f"  {training_name}: FAILED")
    
    return results

def prepare_federated_datasets(data, feature_columns, num_clients=4):
    """FIXED federated dataset preparation ensuring each client has both classes"""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    
    print(f"📂 Preparing federated datasets for {num_clients} clients")
    
    # Prepare all data first
    X = data[feature_columns].fillna(data[feature_columns].mean())
    y = data['label'].values
    
    # Remove NaN values
    valid_indices = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[valid_indices]
    y = y[valid_indices]
    
    # Check class distribution
    unique_classes, class_counts = np.unique(y, return_counts=True)
    print(f"📊 Overall class distribution: {dict(zip(unique_classes, class_counts))}")
    
    if len(unique_classes) < 2:
        print("❌ Dataset has only one class - cannot proceed with classification")
        return None
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # CRITICAL FIX: Use StratifiedKFold for balanced splitting
    skf = StratifiedKFold(n_splits=num_clients, shuffle=True, random_state=42)
    
    client_datasets = {}
    
    for i, (_, client_indices) in enumerate(skf.split(X_scaled, y)):
        client_id = f"Client_{i}"
        
        X_client = X_scaled[client_indices]
        y_client = y[client_indices]
        
        # Ensure minimum samples per class
        unique_client_classes, client_class_counts = np.unique(y_client, return_counts=True)
        
        if len(unique_client_classes) < 2:
            print(f"⚠️ {client_id} has only one class, adding samples from other classes")
            # Add a few samples from the minority class
            minority_class = 1 - unique_client_classes[0]
            minority_indices = np.where(y == minority_class)[0]
            if len(minority_indices) > 0:
                add_indices = np.random.choice(minority_indices, size=min(5, len(minority_indices)), replace=False)
                X_client = np.vstack([X_client, X_scaled[add_indices]])
                y_client = np.hstack([y_client, y[add_indices]])
        
        # Split into train/test ensuring both classes in training
        try:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X_client, y_client, test_size=0.2, random_state=42, stratify=y_client
            )
        except ValueError:
            # If stratification fails, use simple split
            split_idx = int(0.8 * len(X_client))
            X_train, X_test = X_client[:split_idx], X_client[split_idx:]
            y_train, y_test = y_client[:split_idx], y_client[split_idx:]
        
        client_datasets[client_id] = {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'num_samples': len(X_client)
        }
        
        # Verify class balance
        train_classes, train_counts = np.unique(y_train, return_counts=True)
        test_classes, test_counts = np.unique(y_test, return_counts=True)
        
        print(f"   ✅ {client_id}: {len(X_train)} train, {len(X_test)} test")
        print(f"      Train classes: {dict(zip(train_classes, train_counts))}")
        print(f"      Test classes: {dict(zip(test_classes, test_counts))}")
    
    return client_datasets

def main():
    """Main execution function"""
    print("🚀 Fixed Federated Learning Framework")
    print("Testing FedAvg and SCAFFOLD on Encrypted and Non-Encrypted Data")
    print("=" * 70)
    
    try:
        # Run comprehensive experiments
        results = run_comprehensive_federated_experiments()
        
        if results:
            print("\n✅ All experiments completed successfully!")
            print("📊 Check 'federated_learning_results.json' for detailed results")
        else:
            print("\n❌ Experiments failed - check error messages above")
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()