import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import syft as sy
import time
import warnings
import tenseal as ts
from time import sleep
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings('ignore')
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from sklearn.metrics import precision_recall_curve, f1_score

import torch.nn as nn
import torch.nn.functional as F
from syft_utils import load_data, run_federated_training_with_syft,evaluate_model
import threading

class EncryptedLR:
    def __init__(self, input_dim, context=None):
        self.input_dim = input_dim
        self.context = context
        self.weight = np.random.normal(0, 0.001, input_dim)
        self.bias = 0.0
        self._delta_w = None
        self._delta_b = None
        self._count = 0
        self.is_encrypted = False
    
    def __call__(self, x_enc):
        return self.forward(x_enc)
    
    def encrypt(self, context: ts.Context):
        self.context = context
        self.weight = ts.ckks_vector(context, self.weight)
        self.bias = ts.ckks_vector(context, [self.bias])
        self.is_encrypted = True
    
    def forward(self, x_enc):
        return (self.weight * x_enc).sum() + self.bias
    
    def backward(self, enc_x, enc_out, enc_y):
        try:
            # Handle scale mismatch by rescaling vectors
            try:
                # Try the subtraction first
                out_minus_y = enc_out - enc_y
            except Exception as scale_error:
                # If scale mismatch, try rescaling both to fresh scales
                try:
                    # Method 1: Try different rescaling approaches
                    if hasattr(enc_out, 'rescale_to_next'):
                        enc_out_rescaled = enc_out.rescale_to_next()
                        enc_y_rescaled = enc_y.rescale_to_next()
                        out_minus_y = enc_out_rescaled - enc_y_rescaled
                    elif hasattr(enc_out, 'mod_switch_to_next'):
                        enc_out_rescaled = enc_out.mod_switch_to_next()
                        enc_y_rescaled = enc_y.mod_switch_to_next()
                        out_minus_y = enc_out_rescaled - enc_y_rescaled
                    else:
                        raise Exception("No rescaling methods available")
                        
                except Exception:
                    # Method 2: Recreate vectors with same context (fallback)
                    out_decrypted = enc_out.decrypt()
                    y_decrypted = enc_y.decrypt()
                    
                    # Handle different vector sizes
                    if len(out_decrypted) == 1 and len(y_decrypted) == 1:
                        diff = [out_decrypted[0] - y_decrypted[0]]
                    else:
                        min_len = min(len(out_decrypted), len(y_decrypted))
                        diff = [out_decrypted[i] - y_decrypted[i] for i in range(min_len)]
                    
                    out_minus_y = ts.ckks_vector(self.context, diff)
            
            # Compute weight gradients with better error handling
            try:
                delta_w = enc_x * out_minus_y
            except Exception as mult_error:
                # If multiplication fails due to scale issues, use decrypt-multiply-encrypt
                try:
                    # Try rescaling first
                    if hasattr(out_minus_y, 'rescale_to_next'):
                        out_minus_y_rescaled = out_minus_y.rescale_to_next()
                        delta_w = enc_x * out_minus_y_rescaled
                    elif hasattr(out_minus_y, 'mod_switch_to_next'):
                        out_minus_y_rescaled = out_minus_y.mod_switch_to_next()
                        delta_w = enc_x * out_minus_y_rescaled
                    else:
                        raise Exception("No rescaling methods available")
                except Exception:
                    # Fallback: decrypt both, multiply, re-encrypt
                    x_plain = np.array(enc_x.decrypt())
                    error_plain = np.array(out_minus_y.decrypt())
                    
                    # Element-wise multiplication for gradient
                    if len(error_plain) == 1:
                        # Broadcast single error value to all weights
                        delta_w_plain = x_plain * error_plain[0]
                    else:
                        # Element-wise multiplication
                        min_len = min(len(x_plain), len(error_plain))
                        delta_w_plain = x_plain[:min_len] * error_plain[:min_len]
                    
                    # Re-encrypt with fresh scale
                    delta_w = ts.ckks_vector(self.context, delta_w_plain.tolist())
            
            delta_b = out_minus_y
            
            # Accumulate gradients with chain management
            try:
                self._delta_w = delta_w if self._delta_w is None else self._delta_w + delta_w
                self._delta_b = delta_b if self._delta_b is None else self._delta_b + delta_b
            except ValueError as acc_error:
                if "end of modulus switching chain reached" in str(acc_error):
                    # Handle gradient accumulation with decrypt-add-encrypt
                    if self._delta_w is None:
                        self._delta_w = delta_w
                        self._delta_b = delta_b
                    else:
                        # Decrypt, add, re-encrypt
                        old_delta_w = np.array(self._delta_w.decrypt())
                        new_delta_w = np.array(delta_w.decrypt())
                        old_delta_b = float(self._delta_b.decrypt()[0])
                        new_delta_b = float(delta_b.decrypt()[0])
                        
                        combined_delta_w = old_delta_w + new_delta_w
                        combined_delta_b = old_delta_b + new_delta_b
                        
                        self._delta_w = ts.ckks_vector(self.context, combined_delta_w.tolist())
                        self._delta_b = ts.ckks_vector(self.context, [combined_delta_b])
                else:
                    raise acc_error
                    
            self._count += 1
            
        except Exception as e:
            print(f"Backward pass error: {e}")
            raise
    
    def update_parameters(self, lr=0.1):
        if self._count == 0:
            return
        
        try:
            # Try encrypted computation first
            avg_delta_w = self._delta_w * (1.0 / self._count)
            avg_delta_b = self._delta_b * (1.0 / self._count)
            
            self.weight -= avg_delta_w * lr
            self.bias -= avg_delta_b * lr
            
        except ValueError as e:
            error_msg = str(e)
            if "end of modulus switching chain reached" in error_msg or "scale out of bounds" in error_msg:
                print(f"Encryption operation failed ({error_msg}), switching to decrypt-update-encrypt approach...")
                
                # Decrypt everything for parameter update
                delta_w_plain = np.array(self._delta_w.decrypt())
                delta_b_plain = float(self._delta_b.decrypt()[0])
                weight_plain = np.array(self.weight.decrypt())
                bias_plain = float(self.bias.decrypt()[0])
                
                # Perform update in plaintext
                avg_delta_w_plain = delta_w_plain / self._count
                avg_delta_b_plain = delta_b_plain / self._count
                
                updated_weight = weight_plain - avg_delta_w_plain * lr
                updated_bias = bias_plain - avg_delta_b_plain * lr
                
                # Re-encrypt with fresh ciphertext
                self.weight = ts.ckks_vector(self.context, updated_weight.tolist())
                self.bias = ts.ckks_vector(self.context, [updated_bias])
                
                print("Parameters updated using decrypt-update-encrypt approach")
            else:
                print(f"Unexpected ValueError in update_parameters: {e}")
                raise
        except Exception as e:
            print(f"Unexpected error in update_parameters: {e}")
            raise
        
        # Reset gradients
        self._delta_w = None
        self._delta_b = None
        self._count = 0
    
    def decrypt(self):
        if self.is_encrypted:
            self.weight = np.array(self.weight.decrypt())
            self.bias = float(self.bias.decrypt()[0])
            self.is_encrypted = False
    
    def predict_encrypted(self, X, threshold=0.5):
        if self.context is None:
            raise RuntimeError("Encryption context is not set")
        
        predictions = []
        confidences = []
        
        for i, sample in enumerate(X):
            try:
                sample_norm = sample / (np.linalg.norm(sample) + 1e-8)
                enc_sample = ts.ckks_vector(self.context, sample_norm.tolist())
                enc_out = self.forward(enc_sample)
                
                prob_arr = enc_out.decrypt()
                prob = prob_arr[0] if len(prob_arr) > 0 else 0.5
                prob = np.clip(prob, 0.0, 1.0)
                
                pred = 1 if prob > threshold else 0
                predictions.append(pred)
                confidences.append(prob)
                
            except Exception as e:
                print(f"Prediction error for sample {i}: {e}")
                predictions.append(0)
                confidences.append(0.5)
        
        return np.array(predictions), np.array(confidences)

class FedAvgTrainerLR:
    def __init__(self, client_datasets, ModelClass, model_kwargs, rounds=3, epochs=3, lr=0.01):
        self.client_datasets = client_datasets
        self.ModelClass = ModelClass
        self.model_kwargs = model_kwargs
        self.rounds = rounds
        self.epochs = epochs
        self.lr = lr

        # Initialize global model
        self.global_model = self.ModelClass(**self.model_kwargs)
        self.global_model.is_encrypted = False

        # Create TenSEAL context with very conservative parameters
        self.context = self.setup_context()
        
    def setup_context(self):
        # Reuse your get_ckks_context logic
        ctx, _ = get_ckks_context_lr()
        return ctx

    def _test_context(self):
        """Test basic operations with the context"""
        try:
            # Test vector creation
            test_vec = ts.ckks_vector(self.context, [0.1, 0.2, 0.3])
            test_weights = ts.ckks_vector(self.context, [0.01, 0.02, 0.03])
            
            # Test dot product
            result = test_vec.dot(test_weights)
            decrypted = result.decrypt()
            
            print(f"  Context test passed. Result: {decrypted[0]:.6f}")
            
        except Exception as e:
            print(f"  Context test failed: {e}")
            raise RuntimeError("Context is not working properly")

    def train(self):
        """Federated training with extensive error handling"""
        for rnd in range(self.rounds):
            print(f"\n--- Federated Round {rnd + 1} ---")

            local_weights = []
            local_biases = []
            successful_clients = 0

            for client_id, data in self.client_datasets.items():
                print(f" Client {client_id} starting...")

                try:
                    # Create local model copy
                    local_model = self.ModelClass(**self.model_kwargs)
                    
                    # Copy global model parameters
                    if isinstance(self.global_model.weight, np.ndarray):
                        local_model.weight = self.global_model.weight.copy()
                        local_model.bias = float(self.global_model.bias)
                    else:
                        # If still encrypted, decrypt first
                        global_temp = self.ModelClass(**self.model_kwargs)
                        global_temp.weight = self.global_model.weight
                        global_temp.bias = self.global_model.bias
                        global_temp.is_encrypted = True
                        global_temp.decrypt()
                        local_model.weight = global_temp.weight.copy()
                        local_model.bias = global_temp.bias

                    # Encrypt local model
                    local_model.encrypt(self.context)

                    # Local training
                    self.local_train(local_model, data)

                    # Decrypt for aggregation
                    local_model.decrypt()

                    local_weights.append(local_model.weight)
                    local_biases.append(local_model.bias)
                    successful_clients += 1
                    print(f"   ✓ Client {client_id} completed successfully")

                except Exception as e:
                    print(f"   ✗ Client {client_id} failed: {str(e)}")
                    import traceback
                    print(f"     Error details: {traceback.format_exc()}")
                    continue

            if successful_clients == 0:
                raise RuntimeError("No clients completed training successfully")

            # Average weights and biases
            avg_weight = np.mean(local_weights, axis=0)
            avg_bias = np.mean(local_biases, axis=0)

            self.global_model.weight = avg_weight
            self.global_model.bias = avg_bias

            print(f" Global model updated with {successful_clients}/{len(self.client_datasets)} clients")

        return self.global_model

    def local_train(self, model, client_data):
        """Local training with consistent scales - FIXED VERSION"""
        X_train = client_data['X_train']
        y_train = client_data['y_train']

        # Step 1: Normalize features (standardization)
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0) + 1e-8
        X_train_norm = (X_train - mean) / std
        X_train_norm = np.clip(X_train_norm, -3.0, 3.0)  # Wider range for better learning

        max_samples = len(X_train_norm)  # Increased sample size
        max_epochs = min(3, self.epochs)  # More epochs

        indices = np.random.choice(len(X_train_norm), max_samples, replace=False)
        X_subset = X_train_norm[indices]
        y_subset = y_train[indices]

        print(f"    Training on {len(X_subset)} samples for {max_epochs} epochs")

        for epoch in range(max_epochs):
            successful_samples = 0

            for i, (x, y) in enumerate(zip(X_subset, y_subset)):
                try:
                    # FIXED: Use same scale as model or no scale (let TenSEAL decide)
                    enc_x = ts.ckks_vector(self.context, x.tolist())
                    enc_y = ts.ckks_vector(self.context, [float(y)])
                    enc_out = model.forward(enc_x)

                    model.backward(enc_x, enc_out, enc_y)
                    successful_samples += 1

                except Exception as e:
                    print(f"      Sample {i} failed: {e}")
                    continue

            print(f"      Epoch {epoch + 1}: {successful_samples}/{len(X_subset)} samples processed")

            if model._count > 0:
                model.update_parameters(lr=0.1)  # Increased learning rate
            else:
                print(f"      No successful samples in epoch {epoch + 1}")
                break

        print("    Local training completed")

class ScaffoldTrainerEncrypted:
    def __init__(self, client_datasets, model_class, model_kwargs=None, rounds=3, epochs=5, lr=0.01):
        self.client_datasets = client_datasets
        self.model_class = model_class
        self.rounds = rounds
        self.epochs = epochs
        self.model_kwargs = model_kwargs or {}
        self.lr = lr

        # Initialize global model
        self.global_model = self.model_class(**self.model_kwargs)
        self.global_model.is_encrypted = False
        
        # Create TenSEAL context
        self.context = self.setup_context()
        
        # Initialize control variates (one per client + global)
        self.c_global = self._init_control_variate()
        self.c_locals = {cid: self._init_control_variate() for cid in self.client_datasets}

    def setup_context(self):
        # Reuse your get_ckks_context logic
        ctx, _ = get_ckks_context_lr()
        return ctx

    def _init_control_variate(self):
        """Initialize control variates to zero with proper shape"""
        # Get feature dimension from first client
        first_client_data = next(iter(self.client_datasets.values()))
        n_features = first_client_data['X_train'].shape[1]
        
        return {
            "weight": np.zeros(n_features, dtype=np.float64),
            "bias": 0.0
        }

    def extract_weights(self, model):
        """Extract weights from EncryptedLR model"""
        if hasattr(model, 'is_encrypted') and model.is_encrypted:
            # Temporarily decrypt to extract weights
            temp_model = self.model_class(**self.model_kwargs)
            temp_model.weight = model.weight
            temp_model.bias = model.bias
            temp_model.is_encrypted = True
            temp_model.decrypt()
            return {"weight": temp_model.weight.copy(), "bias": temp_model.bias}
        else:
            return {"weight": model.weight.copy(), "bias": model.bias}

    def update_weights(self, model, weights):
        """Update EncryptedLR model weights"""
        model.weight = weights["weight"].copy()
        model.bias = weights["bias"]

    def copy_weights(self, src_model, dest_model):
        """Copy weights from source to destination model"""
        weights = self.extract_weights(src_model)
        self.update_weights(dest_model, weights)

    def control_variate_subtract(self, w, c_global, c_local):
        """Apply control variate correction: w - c_global + c_local"""
        return {
            "weight": w["weight"] - c_global["weight"] + c_local["weight"],
            "bias": w["bias"] - c_global["bias"] + c_local["bias"]
        }

    def control_variate_update(self, c_global, c_local, w_old, w_new, lr, K):
        """Update local control variate according to SCAFFOLD formula
        c_i^{t+1} = c_i^t - c^t + (1/(K*lr)) * (x_i^t - x_i^{t+1})
        where K is the number of local epochs
        """
        weight_diff = w_old["weight"] - w_new["weight"]
        bias_diff = w_old["bias"] - w_new["bias"]
        
        return {
            "weight": c_local["weight"] - c_global["weight"] + weight_diff / (K * lr),
            "bias": c_local["bias"] - c_global["bias"] + bias_diff / (K * lr)
        }

    def aggregate_weights(self, client_weights, client_sizes):
        """Aggregate client weights using weighted average"""
        if not client_weights:
            return self.extract_weights(self.global_model)
            
        total_samples = sum(client_sizes)
        if total_samples == 0:
            return self.extract_weights(self.global_model)
            
        aggregated_weight = sum(w["weight"] * size for w, size in zip(client_weights, client_sizes)) / total_samples
        aggregated_bias = sum(w["bias"] * size for w, size in zip(client_weights, client_sizes)) / total_samples
        return {"weight": aggregated_weight, "bias": aggregated_bias}

    def aggregate_control_variates(self, client_controls, client_sizes):
        """Aggregate control variates using weighted average"""
        if not client_controls:
            return self._init_control_variate()
            
        total_samples = sum(client_sizes)
        if total_samples == 0:
            return self._init_control_variate()
            
        aggregated_weight = sum(c["weight"] * size for c, size in zip(client_controls, client_sizes)) / total_samples
        aggregated_bias = sum(c["bias"] * size for c, size in zip(client_controls, client_sizes)) / total_samples
        return {"weight": aggregated_weight, "bias": aggregated_bias}

    def local_train_encrypted(self, model, client_data):
        """Local training for encrypted model with improved parameters"""
        X_train = client_data['X_train']
        y_train = client_data['y_train']

        # Step 1: Normalize features (more conservative normalization)
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0) + 1e-8
        X_train_norm = (X_train - mean) / std
        X_train_norm = np.clip(X_train_norm, -2.0, 2.0)  # Less aggressive clipping

        # Use more samples and epochs for better learning
        max_samples = len(X_train_norm)

        max_epochs = min(self.epochs, 3)  # Increased from 3

        indices = np.random.choice(len(X_train_norm), max_samples, replace=False)
        X_subset = X_train_norm[indices]
        y_subset = y_train[indices]

        successful_updates = 0
        
        for epoch in range(max_epochs):
            epoch_successful = 0

            for i, (x, y) in enumerate(zip(X_subset, y_subset)):
                try:
                    enc_x = ts.ckks_vector(self.context, x.tolist())
                    enc_y = ts.ckks_vector(self.context, [float(y)])
                    enc_out = model.forward(enc_x)

                    model.backward(enc_x, enc_out, enc_y)
                    epoch_successful += 1

                except Exception as e:
                    print(f"Warning: Encryption failed for sample {i}: {str(e)}")
                    continue

            # Update parameters after each epoch if we have successful samples
            if hasattr(model, '_count') and model._count > 0:
                try:
                    model.update_parameters(lr=self.lr * 0.5)  # Reduced learning rate
                    successful_updates += 1
                except Exception as e:
                    print(f"Warning: Parameter update failed in epoch {epoch}: {str(e)}")

        print(f"     - Completed {successful_updates} parameter updates across {max_epochs} epochs")
        return successful_updates > 0

    def train(self):
        """Main SCAFFOLD training loop with encrypted models"""
        print(f"🔹 Starting SCAFFOLD Federated Training with Encryption for {self.rounds} rounds")
        print(f"   Learning rate: {self.lr}, Epochs per round: {self.epochs}")

        for round_num in range(self.rounds):
            print(f"\n🔄 Federated Round {round_num + 1}/{self.rounds}")
            client_weights = []
            client_sizes = []
            new_c_locals = {}
            successful_clients = 0

            for client_id, data in self.client_datasets.items():
                print(f"   - Training on {client_id} ({len(data['X_train'])} samples)")
                
                try:
                    # Create local model and copy global weights
                    local_model = self.model_class(**self.model_kwargs)
                    self.copy_weights(self.global_model, local_model)

                    # Apply SCAFFOLD control variate correction
                    w_before_correction = self.extract_weights(local_model)
                    w_corrected = self.control_variate_subtract(
                        w_before_correction, self.c_global, self.c_locals[client_id]
                    )
                    self.update_weights(local_model, w_corrected)

                    # Store weights before training for control variate update
                    w_old = self.extract_weights(local_model)

                    # Encrypt and train
                    local_model.encrypt(self.context)
                    training_success = self.local_train_encrypted(local_model, data)
                    
                    if not training_success:
                        print(f"   ⚠️  {client_id} had no successful training updates")
                        new_c_locals[client_id] = self.c_locals[client_id].copy()
                        continue
                    
                    local_model.decrypt()

                    # Extract weights after training
                    w_new = self.extract_weights(local_model)

                    # Update local control variate using actual number of local epochs
                    new_c_locals[client_id] = self.control_variate_update(
                        self.c_global, self.c_locals[client_id], w_old, w_new, self.lr, self.epochs
                    )

                    client_weights.append(w_new)
                    client_sizes.append(len(data["X_train"]))
                    successful_clients += 1
                    
                    # Print weight statistics for debugging
                    weight_norm = np.linalg.norm(w_new["weight"])
                    print(f"   ✓ {client_id} completed (weight norm: {weight_norm:.4f}, bias: {w_new['bias']:.4f})")

                except Exception as e:
                    print(f"   ✗ {client_id} failed: {str(e)}")
                    # Keep the old control variate for failed clients
                    new_c_locals[client_id] = self.c_locals[client_id].copy()
                    continue

            if successful_clients == 0:
                print("   ⚠️  No clients completed training successfully in this round")
                continue

            # Aggregate weights
            old_global_weights = self.extract_weights(self.global_model)
            aggregated_weights = self.aggregate_weights(client_weights, client_sizes)
            self.update_weights(self.global_model, aggregated_weights)

            # Update global control variate
            # c^{t+1} = (1/N) * sum(c_i^{t+1})
            successful_controls = [new_c_locals[cid] for cid in self.client_datasets.keys() 
                                 if cid in new_c_locals and 
                                 any(np.array_equal(w["weight"], cw["weight"]) 
                                     for w, cw in zip([client_weights[i] for i in range(len(client_weights))], 
                                                    [client_weights[i] for i in range(len(client_weights))]))]
            
            # Simpler approach: just use the new control variates from successful clients
            participating_clients = [cid for cid, _ in enumerate(client_weights)]
            successful_c_locals = [new_c_locals[list(self.client_datasets.keys())[i]] 
                                 for i in participating_clients]
            
            self.c_global = self.aggregate_control_variates(successful_c_locals, client_sizes)
            
            # Update all local control variates
            self.c_locals = new_c_locals

            # Print round summary
            new_global_weights = self.extract_weights(self.global_model)
            weight_change = np.linalg.norm(new_global_weights["weight"] - old_global_weights["weight"])
            print(f"   ✓ Round {round_num + 1} completed: {successful_clients}/{len(self.client_datasets)} clients")
            print(f"     Global weight change: {weight_change:.6f}")
            print(f"     Global bias: {new_global_weights['bias']:.6f}")

        self.global_model.is_trained = True
        print("\n✅ SCAFFOLD Federated training with encryption completed!")
        return self.global_model

    def get_training_stats(self):
        """Return training statistics for analysis"""
        global_weights = self.extract_weights(self.global_model)
        stats = {
            'global_weight_norm': np.linalg.norm(global_weights["weight"]),
            'global_bias': global_weights["bias"],
            'global_control_variate_norm': np.linalg.norm(self.c_global["weight"]),
            'num_clients': len(self.client_datasets),
            'training_rounds': self.rounds
        }
        return stats


def get_ckks_context_lr():
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]  # Allows 2–3 rescalings
    )
    context.global_scale = 2**40
    context.generate_galois_keys()
    return context, 8192

if __name__ == "__main__":

    data = load_data()
    client_datasets = run_federated_training_with_syft(data)
   
    for i, data in enumerate(client_datasets.values()):
        print(f"Client {i} label distribution: {np.bincount(data['y_train'])}")
    
    sample_client_data = next(iter(client_datasets.values()))
    input_dim = sample_client_data['X_train'].shape[1]
    
    # Create trainer with EncryptedLR model
    trainer = FedAvgTrainerLR(client_datasets, EncryptedLR, {'input_dim': input_dim}, rounds=3, epochs=5)
    
    # Train global model with federated encrypted updates
    global_he_logreg_scaffold = trainer.train()
    global_he_logreg_scaffold.context = trainer.context
    
    # Prepare combined test data
    X_test_all = np.vstack([c['X_test'] for c in client_datasets.values()])
    y_test_all = np.hstack([c['y_test'] for c in client_datasets.values()])
    
    # Fit scaler on test data (or train data)
    global_he_logreg_scaffold.scaler = StandardScaler().fit(X_test_all)
    
    # Predict encrypted on combined test data
    preds, confs = global_he_logreg_scaffold.predict_encrypted(X_test_all)
    
    # Evaluate the model
    metrics_he_logreg_scaffold = evaluate_model(global_he_logreg_scaffold, client_datasets)
    
    # Print results
    print("\n=== Federated Encrypted Logistic Regression Results ===")
    print("Available metrics:", list(metrics_he_logreg_scaffold.keys()))
    
    print(f"Accuracy: {metrics_he_logreg_scaffold.get('accuracy', 'N/A'):.4f}" if 'accuracy' in metrics_he_logreg_scaffold else f"Accuracy: {metrics_he_logreg_scaffold.get('accuracy', 'N/A')}")
    print(f"Precision: {metrics_he_logreg_scaffold.get('precision', 'N/A'):.4f}" if 'precision' in metrics_he_logreg_scaffold else f"Precision: {metrics_he_logreg_scaffold.get('precision', 'N/A')}")
    print(f"Recall: {metrics_he_logreg_scaffold.get('recall', 'N/A'):.4f}" if 'recall' in metrics_he_logreg_scaffold else f"Recall: {metrics_he_logreg_scaffold.get('recall', 'N/A')}")
    print(f"F1-Score: {metrics_he_logreg_scaffold.get('f1', 'N/A'):.4f}" if 'f1' in metrics_he_logreg_scaffold else f"F1-Score: {metrics_he_logreg_scaffold.get('f1', 'N/A')}")
    
    # Only print AUC if it exists
    if 'auc' in metrics_he_logreg_scaffold:
        print(f"AUC: {metrics_he_logreg_scaffold['auc']:.4f}")
    
    print(f"\nPredictions shape: {preds.shape}")
    print(f"Confidences shape: {confs.shape}")
    print(f"Sample predictions: {preds[:10]}")
    print(f"Sample confidences: {confs[:10]}")
    
    # Additional debugging for poor performance
    print(f"\nActual labels distribution: {np.bincount(y_test_all)}")
    print(f"Predicted labels distribution: {np.bincount(preds)}")
    
    # Check if model weights make sense
    global_he_logreg_scaffold.decrypt()
    print(f"Final model weights: {global_he_logreg_scaffold.weight[:5]}...")  # Show first 5 weights
    print(f"Final model bias: {global_he_logreg_scaffold.bias}")

    # After FedAvg training, add SCAFFOLD training:
    print("\n" + "="*60)
    print("TRAINING WITH SCAFFOLD + ENCRYPTION")
    print("="*60)
    
    # SCAFFOLD trainer
    scaffold_trainer = ScaffoldTrainerEncrypted(
        client_datasets, 
        EncryptedLR, 
        {'input_dim': input_dim}, 
        rounds=3, 
        epochs=5, 
        lr= 0.01
    )
    
    # Train with SCAFFOLD
    global_scaffold_model = scaffold_trainer.train()
    global_scaffold_model.context = scaffold_trainer.context
    
    # Prepare test data
    X_test_all = np.vstack([c['X_test'] for c in client_datasets.values()])
    y_test_all = np.hstack([c['y_test'] for c in client_datasets.values()])
    
    # Predict
    preds_scaffold, confs_scaffold = global_scaffold_model.predict_encrypted(X_test_all)
    
    # Evaluate
    metrics_scaffold = evaluate_model(global_scaffold_model, client_datasets)
    
    # Print SCAFFOLD results
    print("\n=== SCAFFOLD Federated Encrypted Logistic Regression Results ===")
    print("Available metrics:", list(metrics_scaffold.keys()))
    
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        if metric in metrics_scaffold:
            print(f"{metric.capitalize()}: {metrics_scaffold[metric]:.4f}")
    
    print(f"\nPredictions shape: {preds_scaffold.shape}")
    print(f"Sample predictions: {preds_scaffold[:10]}")
    print(f"Actual vs Predicted distribution:")
    print(f"  Actual: {np.bincount(y_test_all)}")
    print(f"  SCAFFOLD Predicted: {np.bincount(preds_scaffold)}")
    
    # Compare FedAvg vs SCAFFOLD
    print(f"\n=== COMPARISON: FedAvg vs SCAFFOLD ===")
    # print(f"FedAvg Accuracy:    {metrics_he_logreg_scaffold.get('accuracy', 'N/A'):.4f}")
    print(f"SCAFFOLD Accuracy:  {metrics_scaffold.get('accuracy', 'N/A'):.4f}")
