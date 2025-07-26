from sklearn.neural_network import MLPClassifier
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
from syft_utils import load_data, run_federated_training_with_syft,evaluate_model,get_ckks_context,evaluate_biometric_model
import threading
from sklearn.ensemble import RandomForestClassifier

class BiometricHomomorphicLogisticRegression:
    def __init__(self,input_dim, poly_modulus_degree=8192, scale=2**40):
        """
        Initialize TenSEAL CKKS context following best practices from official tutorials.
        Based on TenSEAL Tutorial 1 - Logistic Regression on Encrypted Data
        """
        # Create context with proper coefficient modulus for multiple multiplications
        self.HE = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_modulus_degree,
            coeff_mod_bit_sizes=[60, 40, 40, 60]  # Standard config for 2-3 multiplications
        )
        self.HE.global_scale = scale
        self.HE.generate_galois_keys()
        self.HE.generate_relin_keys()
        
        self.is_trained = False
        self.scaler = StandardScaler()
        self.w = None
        self.b = None
        self.scale = scale
        self.input_dim = input_dim
        self.init_weights()
        
        print(f"🔐 Initialized Biometric Homomorphic Logistic Regression")
        print(f"   - Polynomial modulus degree: {poly_modulus_degree}")
        print(f"   - Scale: 2^{int(np.log2(scale))}")
        print(f"   - Coefficient modulus: [60, 40, 40, 60] bits")
    
    def init_weights(self):
        """Initialize weights with small values instead of leaving them None"""
        self.w = np.zeros(self.input_dim)
        self.b = 0.0    
    def train_plaintext(self, X, y):
        """Train plaintext logistic regression and store weights."""
        # Normalize features to prevent overflow in HE operations
        X_scaled = self.scaler.fit_transform(X)
        
        # Use regularization to keep weights small
        logreg = LogisticRegression(max_iter=1000, random_state=42, C=10.0)
        logreg.fit(X_scaled, y)

        self.w = logreg.coef_.flatten()
        self.b = logreg.intercept_[0]

        # STRONGER clamp on max weight to protect HE inference
        max_weight = np.max(np.abs(self.w))
        if max_weight > 0.5:
            scale_factor = 0.5 / max_weight
            self.w *= scale_factor
            self.b *= scale_factor
            print(f"⚠️  Scaled down weights by {scale_factor:.3f} to prevent HE overflow")
        
        self.is_trained = True
        print("✅ Plaintext logistic regression trained")
        print(f"   - Weight range: [{np.min(self.w):.4f}, {np.max(self.w):.4f}]")
        print(f"   - Bias: {self.b:.4f}")

    def polynomial_sigmoid(self, enc_x):
        """
        Polynomial approximation of sigmoid function.
        Using degree-3 polynomial: σ(x) ≈ 0.5 + 0.197*x - 0.004*x^3
        
        This is based on TenSEAL tutorial approach for sigmoid approximation.
        """
        # Coefficient for linear term
        coeff_1 = 0.197
        # Coefficient for cubic term  
        coeff_3 = 0.004
        
        # Compute x^3 (this uses 2 multiplication levels)
        enc_x_squared = enc_x * enc_x
        enc_x_cubed = enc_x_squared * enc_x
        
        # Polynomial: 0.5 + 0.197*x - 0.004*x^3
        result = enc_x * coeff_1 - enc_x_cubed * coeff_3
        result += 0.5
        
        return result

    def simple_sigmoid(self, enc_x):
        """
        Simple linear approximation: σ(x) ≈ 0.5 + 0.25*x
        Uses only 1 multiplication level, safer for scale management.
        """
        return enc_x * 0.25 + 0.5
    def train(self, X, y, epochs=1, lr=0.01, verbose=False):
        """
        Compatibility method for FedAvgTrainer.
        Trains a plaintext logistic regression model and stores the weights.
        """
        if verbose:
            print("🧪 Using plaintext training within homomorphic model wrapper.")
        self.train_plaintext(X, y)
    

    def predict_encrypted(self, X, threshold=0.5, use_polynomial=False):
        """
        Perform encrypted inference following TenSEAL patterns.
        
        Args:
            X: Input features
            threshold: Classification threshold  
            use_polynomial: If True, use degree-3 polynomial sigmoid, else linear
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        
        # Scale inputs using the fitted scaler
        X_scaled = self.scaler.transform(X)
        
        predictions = []
        confidences = []
        
        print(f"🔐 Running encrypted inference on {len(X_scaled)} samples...")
        print(f"   - Using {'polynomial' if use_polynomial else 'linear'} sigmoid approximation")
        
        for i, sample in enumerate(X_scaled):
            try:
                # Step 1: Encrypt the input sample
                enc_sample = ts.ckks_vector(self.HE, sample.tolist())
                
                # Step 2: Compute encrypted dot product w·x
                enc_dot_product = enc_sample.dot(self.w.tolist())
                
                # Step 3: Add bias term
                enc_logits = enc_dot_product + self.b
                
                # Step 4: Apply sigmoid approximation
                if use_polynomial:
                    enc_probability = self.polynomial_sigmoid(enc_logits)
                else:
                    enc_probability = self.simple_sigmoid(enc_logits)
                
                # Step 5: Decrypt the result
                prob_decrypted = enc_probability.decrypt()
                probability = prob_decrypted[0] if len(prob_decrypted) > 0 else 0.5
                
                # Clamp probability to valid range [0, 1]
                probability = np.clip(probability, 0.0, 1.0)
                
                # Make binary prediction
                prediction = 1 if probability > threshold else 0
                
                predictions.append(prediction)
                confidences.append(probability)
                
                # Progress reporting
                if (i + 1) % 100 == 0 or i == 0:
                    print(f"   - Processed {i + 1}/{len(X_scaled)} samples")
                    print(f"   - Sample {i}: logit≈{enc_dot_product.decrypt()[0] + self.b:.3f}, prob≈{probability:.3f}")
                    
            except Exception as e:
                print(f"⚠️  Error processing sample {i}: {str(e)[:100]}")
                # Fallback to neutral prediction
                predictions.append(0)
                confidences.append(0.5)
        
        predictions = np.array(predictions)
        confidences = np.array(confidences)
        
        print(f"✅ Encrypted inference completed")
        print(f"   - Predictions: {np.sum(predictions)} positive out of {len(predictions)}")
        print(f"   - Confidence range: [{np.min(confidences):.3f}, {np.max(confidences):.3f}]")
        
        return predictions, confidences

    def compare_with_plaintext(self, X_test, y_test=None):
        """
        Compare encrypted vs plaintext predictions for validation.
        """
        if not self.is_trained:
            print("❌ Model not trained")
            return
            
        print("\n🔍 Comparing Encrypted vs Plaintext Predictions...")
        
        # Get a small sample for comparison
        n_samples = min(10, len(X_test))
        X_sample = X_test[:n_samples]
        
        # Plaintext predictions
        X_scaled = self.scaler.transform(X_sample)
        plaintext_logits = X_scaled @ self.w + self.b
        plaintext_probs = 1 / (1 + np.exp(-np.clip(plaintext_logits, -500, 500)))
        plaintext_preds = (plaintext_probs > 0.5).astype(int)
        
        # Encrypted predictions (using simple sigmoid for stability)
        enc_preds, enc_probs = self.predict_encrypted(X_sample, use_polynomial=False)
        
        print("\nComparison Results:")
        print("Sample | Plaintext Prob | Encrypted Prob | Diff | Plain Pred | Enc Pred")
        print("-" * 70)
        
        for i in range(n_samples):
            diff = abs(plaintext_probs[i] - enc_probs[i])
            print(f"{i:6d} | {plaintext_probs[i]:13.4f} | {enc_probs[i]:13.4f} | {diff:.4f} | "
                  f"{plaintext_preds[i]:9d} | {enc_preds[i]:7d}")
        
        # Summary statistics
        prob_mae = np.mean(np.abs(plaintext_probs - enc_probs))
        pred_accuracy = np.mean(plaintext_preds == enc_preds)
        
        print(f"\nSummary:")
        print(f"   - Mean Absolute Error (probabilities): {prob_mae:.4f}")
        print(f"   - Prediction Agreement: {pred_accuracy:.1%}")
        
        if prob_mae > 0.1:
            print("⚠️  Large probability differences detected - check HE parameters")
        else:
            print("✅ Good agreement between plaintext and encrypted predictions")

    def test_he_operations(self):
        """
        Test basic HE operations to validate the setup.
        """
        print("\n🧪 Testing Homomorphic Encryption Operations...")
        
        try:
            # Test basic vector operations
            test_vec = [1.0, 2.0, 3.0]
            enc_vec = ts.ckks_vector(self.HE, test_vec)
            
            # Test encryption/decryption
            decrypted = enc_vec.decrypt()
            print(f"   ✅ Encryption/Decryption: {test_vec} -> {[round(x, 3) for x in decrypted]}")
            
            # Test scalar multiplication
            enc_scaled = enc_vec * 2.0
            scaled_result = enc_scaled.decrypt()
            print(f"   ✅ Scalar multiplication: {[round(x, 3) for x in scaled_result]}")
            
            # Test addition
            enc_sum = enc_vec + 1.0
            sum_result = enc_sum.decrypt()
            print(f"   ✅ Scalar addition: {[round(x, 3) for x in sum_result]}")
            
            # Test polynomial evaluation (if we have weights)
            if self.is_trained and len(self.w) >= 3:
                test_weights = self.w[:3].tolist()
                dot_result = enc_vec.dot(test_weights)
                dot_decrypted = dot_result.decrypt()[0]
                expected_dot = np.dot(test_vec, test_weights)
                print(f"   ✅ Dot product: computed={dot_decrypted:.4f}, expected={expected_dot:.4f}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ HE operation failed: {e}")
            return False



class SimpleEncryptedMLP:
    def __init__(self, input_dim, hidden_dim=32):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Normal MLP weights
        self.w1 = np.random.randn(hidden_dim, input_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.w2 = np.random.randn(hidden_dim) * 0.1
        self.b2 = 0.0
        
        self.is_trained = False
        
        # Setup CKKS context for data encryption only
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        self.context.global_scale = 2**40
        self.context.generate_galois_keys()
    
    def encrypt_data(self, data):
        """Simply encrypt data using CKKS"""
        if isinstance(data, (list, np.ndarray)):
            if len(data.shape) == 1:
                # Single sample
                return ts.ckks_vector(self.context, data.tolist())
            else:
                # Multiple samples
                return [ts.ckks_vector(self.context, sample.tolist()) for sample in data]
        else:
            return ts.ckks_vector(self.context, [data])
    
    def decrypt_data(self, encrypted_data):
        """Decrypt CKKS encrypted data"""
        if isinstance(encrypted_data, list):
            return [np.array(enc.decrypt()) for enc in encrypted_data]
        else:
            return np.array(encrypted_data.decrypt())
    
    def forward(self, x):
        """Standard MLP forward pass"""
        z1 = np.dot(self.w1, x) + self.b1
        a1 = np.maximum(0, z1)  # ReLU
        output = np.dot(self.w2, a1) + self.b2
        return output
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def train_with_encrypted_data(self, X, y, epochs=30, lr=0.01, verbose=True):
        """Train MLP - encrypt data, decrypt for computation, re-encrypt if needed"""
        print(f"🔒 Encrypting training data...")
        
        # Step 1: Encrypt the data
        encrypted_X = self.encrypt_data(X)
        encrypted_y = self.encrypt_data(y)
        
        print(f"✅ Data encrypted with CKKS")
        print(f"🏋️ Training MLP for {epochs} epochs...")
        
        # Step 2: Decrypt for training (since we need gradients)
        X_decrypted = self.decrypt_data(encrypted_X)
        y_decrypted = self.decrypt_data(encrypted_y)
        
        # Step 3: Normal training on decrypted data
        n_samples = min(len(X_decrypted), len(X_decrypted)) 
        indices = np.random.choice(len(X_decrypted), n_samples, replace=False)
        
        batch_size = 32
        n_batches = len(indices) // batch_size
        
        for epoch in range(epochs):
            total_loss = 0
            
            # Shuffle data each epoch
            np.random.shuffle(indices)
            
            for batch_idx in range(n_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, len(indices))
                batch_indices = indices[batch_start:batch_end]
                
                batch_loss = 0
                
                for i in batch_indices:
                    x_sample = X_decrypted[i] if isinstance(X_decrypted, list) else X_decrypted[i]
                    y_sample = y_decrypted[i] if isinstance(y_decrypted, list) else y_decrypted[i]
                    
                    if isinstance(y_sample, np.ndarray):
                        y_sample = y_sample[0]
                    
                    # Forward pass
                    z1 = np.dot(self.w1, x_sample) + self.b1
                    a1 = np.maximum(0, z1)
                    output = np.dot(self.w2, a1) + self.b2
                    
                    # FIX 3: Use same loss function as sklearn (sigmoid + cross-entropy)
                    prob = 1 / (1 + np.exp(-output))  # Sigmoid
                    loss = -(y_sample * np.log(prob + 1e-15) + (1-y_sample) * np.log(1-prob + 1e-15))
                    batch_loss += loss
                    
                    # FIX 4: Proper gradients for sigmoid + cross-entropy
                    grad_output = prob - y_sample
                    grad_w2 = grad_output * a1
                    grad_b2 = grad_output
                    
                    grad_a1 = grad_output * self.w2
                    grad_z1 = grad_a1 * (z1 > 0)
                    
                    grad_w1 = np.outer(grad_z1, x_sample)
                    grad_b1 = grad_z1
                    
                    # Update with proper learning rate
                    self.w1 -= lr * grad_w1
                    self.b1 -= lr * grad_b1
                    self.w2 -= lr * grad_w2
                    self.b2 -= lr * grad_b2
                
                total_loss += batch_loss
            
            if verbose and epoch % 10 == 0:
                avg_loss = total_loss / len(indices)
                print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
        
        self.is_trained = True
        if verbose:
            print("✅ Training completed!")
    
    def predict_with_encrypted_data(self, X, threshold=0.5, verbose=True):
        """Improved prediction to match plain MLP"""
        if not self.is_trained:
            print("❌ Model not trained!")
            return None, None
        
        if verbose:
            print(f"🔒 Encrypting test data...")
        
        # Encrypt input data
        encrypted_X = self.encrypt_data(X)
        if verbose:
            print(f"✅ Test data encrypted with CKKS")
        
        # Decrypt for prediction
        X_decrypted = self.decrypt_data(encrypted_X)
        
        predictions = []
        confidences = []
        
        if verbose:
            print(f"🔍 Making predictions on {len(X_decrypted)} samples...")
        
        for i, sample in enumerate(X_decrypted):
            if isinstance(sample, np.ndarray) and len(sample.shape) > 1:
                sample = sample.flatten()
            
            # Forward pass (same as training)
            z1 = np.dot(self.w1, sample) + self.b1
            a1 = np.maximum(0, z1)
            output = np.dot(self.w2, a1) + self.b2
            
            # Proper sigmoid probability
            confidence = 1 / (1 + np.exp(-output))
            pred = 1 if confidence > threshold else 0
            
            predictions.append(pred)
            confidences.append(confidence)
        
        if verbose:
            print("✅ Predictions completed!")
        return np.array(predictions), np.array(confidences)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

def run_centralized_training():
    print("🧠 Centralized Training - Essential Methods")
    print("=" * 50)
    
    # Load data
    data = load_data()
    print(f"\n📊 Loaded {len(data)} records from {data['user_id'].nunique()} users")
    
    feature_columns = ['dwell_avg', 'flight_avg', 'traj_avg']
    
    # ===========================
    # Method 1: Per-User Individual Models (MLP & Logistic Regression)
    # ===========================
    print("\n👤 Per-User Individual Models:")

    # Lists to store per-user metrics for MLP and Logistic Regression separately
    mlp_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    logreg_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    
    for user in data['user_id'].unique():
        user_data = data[data['user_id'] == user].copy()
        
        X = user_data[feature_columns].fillna(user_data[feature_columns].mean())
        y = user_data['label'].values
        
        if len(np.unique(y)) < 2:
            continue
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # --- MLP ---
        mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
        mlp.fit(X_train, y_train)
        y_pred_mlp = mlp.predict(X_test)
        mlp_metrics['accuracy'].append(accuracy_score(y_test, y_pred_mlp))
        mlp_metrics['precision'].append(precision_score(y_test, y_pred_mlp, zero_division=0))
        mlp_metrics['recall'].append(recall_score(y_test, y_pred_mlp, zero_division=0))
        mlp_metrics['f1'].append(f1_score(y_test, y_pred_mlp, zero_division=0))
        
        # --- Logistic Regression ---
        logreg = LogisticRegression(max_iter=1000, random_state=42)
        logreg.fit(X_train, y_train)
        y_pred_logreg = logreg.predict(X_test)
        logreg_metrics['accuracy'].append(accuracy_score(y_test, y_pred_logreg))
        logreg_metrics['precision'].append(precision_score(y_test, y_pred_logreg, zero_division=0))
        logreg_metrics['recall'].append(recall_score(y_test, y_pred_logreg, zero_division=0))
        logreg_metrics['f1'].append(f1_score(y_test, y_pred_logreg, zero_division=0))
    
    # Average per-user results
    def avg_metrics(metrics_dict):
        return {k: np.mean(v) for k, v in metrics_dict.items()}
    
    per_user_mlp_results = avg_metrics(mlp_metrics)
    per_user_logreg_results = avg_metrics(logreg_metrics)
    
    # ===========================
    # Method 2: True Centralized MLP
    # ===========================
    print("\n🌐 True Centralized MLP:")
    
    X_all = data[feature_columns].fillna(data[feature_columns].mean())
    y_all = data['label'].values
    
    scaler_all = StandardScaler()
    X_all_scaled = scaler_all.fit_transform(X_all)
    X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
        X_all_scaled, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    
    mlp_centralized = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
    mlp_centralized.fit(X_train_all, y_train_all)
    
    y_pred_centralized = mlp_centralized.predict(X_test_all)
    y_prob_centralized = mlp_centralized.predict_proba(X_test_all)[:, 1]
    centralized_results = evaluate_biometric_model(y_test_all, y_pred_centralized, y_prob_centralized)
    
    # ===========================
    # Results Summary
    # ===========================
    print("\n📈 Results Summary")
    print("=" * 30)
    
    print(f"\n👤 Per-User MLP (Average across {len(mlp_metrics['accuracy'])} users):")
    for k, v in per_user_mlp_results.items():
        print(f"   {k}: {v:.4f}")
    
    print(f"\n👤 Per-User Logistic Regression (Average across {len(logreg_metrics['accuracy'])} users):")
    for k, v in per_user_logreg_results.items():
        print(f"   {k}: {v:.4f}")
    
    print(f"\n🌐 Centralized MLP:")
    for k, v in centralized_results.items():
        print(f"   {k}: {v:.4f}")
    
    # Performance gap MLP
    gap = per_user_mlp_results['f1'] - centralized_results['f1']
    print(f"\n📊 Performance Gap (Per-User MLP vs Centralized MLP):")
    print(f"   Per-User MLP F1: {per_user_mlp_results['f1']:.3f}")
    print(f"   Centralized MLP F1: {centralized_results['f1']:.3f}")
    print(f"   Gap: {gap:.3f} ({gap/centralized_results['f1']*100:.1f}% difference)")
    
    print(f"\n✅ Centralized training completed!")
    
    return {
        'per_user_mlp': per_user_mlp_results,
        'per_user_logreg': per_user_logreg_results,
        'centralized_mlp': centralized_results,
        'performance_gap': gap
    }

if __name__ == "__main__":
    run_centralized_training()
