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

def run_centralized_training():
    print("🧠 Centralized Training with Encrypted & Plain Biometric MLP + Logistic Regression")
    print("=" * 70)

    # Load and preprocess data
    data = load_data()
    print(f"\n📊 Loaded {len(data)} biometric records from {data['user_id'].nunique()} users")

    # ===========================
    # Preprocess data (shared) - DO THIS FIRST
    # ===========================
    print("📊 Preprocessing biometric data...")
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    feature_data = data[numeric_columns]
    feature_data = feature_data.fillna(feature_data.mean())

    users = data['user_id'].unique()
    target_users = users[:len(users)//2]

    y = data['user_id'].isin(target_users).astype(int)

    # Scale the features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(feature_data)

    print(f"   - Features shape: {X.shape}")
    print(f"   - Authentic samples: {np.sum(y)}")
    print(f"   - Impostor samples: {len(y) - np.sum(y)}")

    # Create train/test split and convert to numpy arrays
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Ensure everything is numpy arrays (not pandas)
    X_train = np.array(X_train)
    X_test = np.array(X_test) 
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print(f"   - Training samples: {len(X_train)}")
    print(f"   - Test samples: {len(X_test)}")

    # ===========================
    # 🔐 Setup Encrypted MLP - NOW we can use X_train.shape[1]
    # ===========================

    print("\n🔐 Training Encrypted MLP...")
    encrypted_mlp = SimpleEncryptedMLP(input_dim=X_train.shape[1], hidden_dim=32)
    encrypted_mlp.train_with_encrypted_data(X_train, y_train, epochs=30, lr=0.01, verbose=False)
    mlp_pred_enc, mlp_conf_enc = encrypted_mlp.predict_with_encrypted_data(X_test, threshold=0.55, verbose=False)
    mlp_enc_metrics = evaluate_biometric_model(y_test, mlp_pred_enc, mlp_conf_enc)
        # ===========================
    # 🧠 Plain MLP Training
    # ===========================
    plain_mlp = MLPClassifier(hidden_layer_sizes=(X_train.shape[1],), max_iter=1000, random_state=42)
    plain_mlp.fit(X_train, y_train)
    mlp_conf_plain = plain_mlp.predict_proba(X_test)[:, 1]
    mlp_pred_plain = (mlp_conf_plain > 0.5).astype(int)
    mlp_plain_metrics = evaluate_biometric_model(y_test, mlp_pred_plain, mlp_conf_plain)

    # ===========================
    # 🟩 Logistic Regression (Plaintext)
    # ===========================
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)
    logreg_conf = logreg.predict_proba(X_test)[:, 1]
    logreg_pred = (logreg_conf > 0.5).astype(int)
    logreg_metrics = evaluate_biometric_model(y_test, logreg_pred, logreg_conf)

    # ===========================
    # 🔐 Logistic Regression (Encrypted)
    # ===========================
    he_logreg = BiometricHomomorphicLogisticRegression(input_dim=X_train.shape[1])
    he_logreg.train_plaintext(X_train, y_train)
    preds_enc, confs_enc =  he_logreg.predict_encrypted(X_test, use_polynomial=False)
    logreg_enc_metrics = evaluate_biometric_model(y_test, preds_enc, confs_enc)

    # ===========================
    # 📊 Print Results
    # ===========================
    print("\n📈 Evaluation Results (Centralized Setup)")
    
    print("\n🔐 BiometricHomomorphicMLP (Encrypted):")
    for k, v in mlp_enc_metrics.items():
        print(f"   {k}: {v:.4f}")

    print("\n🧠 MLPClassifier (Plain):")
    for k, v in mlp_plain_metrics.items():
        print(f"   {k}: {v:.4f}")

    print("\n🟩 Logistic Regression (Plain):")
    for k, v in logreg_metrics.items():
        print(f"   {k}: {v:.4f}")

    print("\n🔐 Logistic Regression (Encrypted):")
    for k, v in logreg_enc_metrics.items():
        print(f"   {k}: {v:.4f}")

    print("\n✅ Centralized Training Completed")
    print("=" * 70)



if __name__ == "__main__":
    run_centralized_training()