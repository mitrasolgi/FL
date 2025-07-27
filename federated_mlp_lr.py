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
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import torch.nn as nn
import torch.nn.functional as F
from syft_utils import load_data, run_federated_training_with_syft,evaluate_model,get_ckks_context
import threading
import json
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
        

class PlainLogisticRegression:
    def __init__(self, input_dim):
        self.w = np.zeros(input_dim)
        self.b = 0.0
        self.is_trained = False

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self, X, y, epochs=100, lr=0.01, verbose=False, control_variate=None):
        n = len(X)
        for epoch in range(epochs):
            dw = np.zeros_like(self.w)
            db = 0.0
            for xi, yi in zip(X, y):
                z = np.dot(self.w, xi) + self.b
                pred = self.sigmoid(z)
                error = pred - yi
                dw += error * xi
                db += error

            # Subtract control variate from gradients if provided (SCAFFOLD)
            if control_variate is not None:
                dw -= control_variate.get("w", 0)
                db -= control_variate.get("b", 0)

            self.w -= lr * dw / n
            self.b -= lr * db / n

            if verbose and (epoch + 1) % 25 == 0:
                loss = -np.mean(y * np.log(pred + 1e-15) + (1 - y) * np.log(1 - pred + 1e-15))
                print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

        self.is_trained = True

    def predict(self, X):
        preds = []
        confs = []
        for x in X:
            prob = self.sigmoid(np.dot(self.w, x) + self.b)
            confs.append(prob)
            preds.append(1 if prob > 0.5 else 0)
        return np.array(preds), np.array(confs)
    
class BiometricHomomorphicMLP:
    def __init__(self, input_dim, hidden_dim, poly_modulus_degree=8192, scale=40):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.is_trained = False

        # Initialize TenSEAL CKKS context with better parameters
        self.HE, self.poly_modulus_degree = get_ckks_context()

        # self.HE.global_scale = 2 ** scale
        # self.HE.generate_galois_keys()
        # self.HE.generate_relin_keys()

        # Initialize model parameters
        self.init_weights()

        # Data preprocessing
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

        print(f"🔐 Initialized Biometric CKKS MLP (TenSEAL)")
        print(f"   - Input dimension: {input_dim}")
        print(f"   - Hidden dimension: {hidden_dim}")
        print(f"   - Polynomial modulus degree: {poly_modulus_degree}")
        print(f"   - Scale: 2^{scale}")

    def init_weights(self):
        """Initialize weights for biometric classification"""
        self.w1 = np.random.normal(0, np.sqrt(1.0 / self.input_dim), 
                                   (self.hidden_dim, self.input_dim))
        self.b1 = np.zeros(self.hidden_dim)
        self.w2 = np.random.normal(0, 0.01, self.hidden_dim)
        self.b2 = 0.0

    def poly_activation(self, x):
        """Simple polynomial activation to minimize multiplicative depth"""
        return x * (0.5 + 0.125 * x)

    def layer_norm(self, x, epsilon=1e-5):
        """Layer normalization for a 1D numpy array"""
        mean = np.mean(x)
        std = np.std(x)
        return (x - mean) / (std + epsilon)

    def train_encrypted_with_decrypted_gradients(self, X, y, epochs=20, lr=0.005):
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        if hasattr(y, "to_numpy"):
            y = y.to_numpy()

        print(f"🏋️ Encrypted Training (with decrypted gradients) for {epochs} epochs...")
        n = len(X)

        batch_size = min(100, n)
        indices = np.random.choice(n, batch_size, replace=False)
        X_batch = X[indices]
        y_batch = y[indices]

        print(f"   - Using batch size: {batch_size}")

        for epoch in range(epochs):
            total_loss = 0.0
            start_time = time.time()

            for i in range(len(X_batch)):
                try:
                    enc_x = ts.ckks_vector(self.HE, X_batch[i].tolist())
                    enc_pred = self.secure_forward_pass(enc_x)
                    pred = enc_pred.decrypt()[0]

                    loss = (pred - y_batch[i]) ** 2
                    total_loss += loss

                    z1 = np.dot(self.w1, X_batch[i]) + self.b1
                    a1 = np.maximum(0, z1)  # ReLU

                    grad_output = 2 * (pred - y_batch[i])
                    grad_w2 = grad_output * a1
                    grad_b2 = grad_output

                    relu_deriv = (z1 > 0).astype(float)
                    grad_hidden = grad_output * self.w2 * relu_deriv

                    grad_w1 = np.outer(grad_hidden, X_batch[i])
                    grad_b1 = grad_hidden

                    self.w1 -= lr * grad_w1
                    self.b1 -= lr * grad_b1
                    self.w2 -= lr * grad_w2
                    self.b2 -= lr * grad_b2

                except Exception as e:
                    print(f"   ❌ Error in sample {i}: {e}")
                    continue

            avg_loss = total_loss / len(X_batch)
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, Time: {epoch_time:.2f}s")

        self.is_trained = True
        print("✅ Encrypted training with decrypted gradients completed!")
    def forward(self, X):
        """Plain (non-encrypted) forward pass"""
        z1 = np.dot(X, self.w1.T) + self.b1  # shape: (batch, hidden_dim)
        a1 = np.maximum(0, z1)               # ReLU
        logits = np.dot(a1, self.w2) + self.b2  # shape: (batch,)
        return logits
    
    def predict(self, X):
        """Plain inference, assumes weights are decrypted for evaluation"""
        logits = self.forward(X)  # assuming you have a forward pass method
        probs = self.sigmoid(logits)  # or softmax, depending on your task
        preds = (probs > 0.5).astype(int)  # or use np.argmax for multi-class
        return preds, probs
    def preprocess_biometric_data(self, data):
        print("📊 Preprocessing biometric data...")
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        feature_data = data[numeric_columns]
        feature_data = feature_data.fillna(feature_data.mean())

        users = data['user_id'].unique()
        target_users = users[:len(users)//2]

        y = data['user_id'].isin(target_users).astype(int)
        X = self.scaler.fit_transform(feature_data)

        print(f"   - Features shape: {X.shape}")
        print(f"   - Authentic samples: {np.sum(y)}")
        print(f"   - Impostor samples: {len(y) - np.sum(y)}")

        return X, y

    def encrypt_sample(self, sample):
        return ts.ckks_vector(self.HE, sample.tolist())

    def encrypt_biometric_batch(self, X_batch, batch_size=50):
        print(f"🔒 Encrypting {len(X_batch)} biometric samples...")
        encrypted_data = []
        for i, sample in enumerate(X_batch):
            try:
                encrypted_sample = self.encrypt_sample(sample)
                encrypted_data.append(encrypted_sample)
                if i % batch_size == 0:
                    print(f"   - Encrypted sample {i + 1}/{len(X_batch)}")
            except Exception as e:
                print(f"   ❌ Encryption failed for sample {i}: {e}")
                continue
        print("✅ Biometric encryption completed!")
        return encrypted_data

    def secure_forward_pass(self, enc_x):
        """
        Returns encrypted final output and decrypted layer 1 pre-activation vector
        """
        try:
            # Encrypted layer 1 pre-activation: enc_z1 = W1*x + b1
            enc_z1 = enc_x.mm(self.w1.T.tolist())  # encrypted vector of size hidden_dim
            enc_z1 = enc_z1 + self.b1.tolist()

            # Decrypt pre-activation for layer norm
            decrypted_z1 = enc_z1.decrypt()

            # Apply layer normalization on decrypted hidden pre-activation
            norm_z1 = self.layer_norm(np.array(decrypted_z1))

            # Re-encrypt normalized values to continue encrypted computation
            enc_norm_z1 = ts.ckks_vector(self.HE, norm_z1.tolist())

            # Simple activation polynomial (e.g. linear scaling)
            enc_a1 = enc_norm_z1 * 0.5

            # Encrypted output layer: W2 * a1 + b2 (dot product)
            enc_output = enc_a1.dot(self.w2.tolist()) + self.b2

            return enc_output

        except Exception as e:
            print(f"❌ Forward pass error: {e}")
            raise

    @classmethod
    def from_weights(cls, w1, b1, w2, b2, **kwargs):
        model = cls(input_dim=w1.shape[1], hidden_dim=w1.shape[0], **kwargs)
        model.w1 = w1
        model.b1 = b1
        model.w2 = w2
        model.b2 = b2
        model.is_trained = True
        return model

    def authenticate_encrypted(self, X_test, threshold=0.55):
        if not self.is_trained:
            print("⚠️ Model not trained yet!")
            return None, None

        preds = []
        confs = []

        for i in range(len(X_test)):
            try:
                enc_x = self.encrypt_sample(X_test[i])
                enc_pred = self.secure_forward_pass(enc_x)
                decrypted = enc_pred.decrypt()[0]

                confidence = self.sigmoid(decrypted)
                label = 1 if confidence > threshold else 0
                preds.append(label)
                confs.append(confidence)

                # if i % 5 == 0:
                #     print(f"   - Sample {i+1}: decrypted={decrypted:.4f}, confidence={confidence:.4f}")

            except Exception as e:
                print(f"   ❌ Error processing sample {i}: {e}")
                continue

        print(f"✅ Encrypted inference complete: {len(preds)} predictions made.")
        return np.array(preds).astype(int), np.array(confs)

    def train(self, X, y, epochs=5, lr=0.005, verbose=False):
        self.train_encrypted_with_decrypted_gradients(X, y, epochs=epochs, lr=lr)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train_on_biometric_data(self, X, y, epochs=5, lr=0.001):
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        if hasattr(y, "to_numpy"):
            y = y.to_numpy()

        print(f"🏋️ Training biometric model for {epochs} epochs...")
        print(f"   - Samples: {len(X)}")
        print(f"   - Features: {X.shape[1]}")
        print(f"   - Learning rate: {lr}")

        train_size = min(30, len(X))
        indices = np.random.choice(len(X), train_size, replace=False)
        X_train = X[indices]
        y_train = y[indices]

        print(f"   - Training on {len(X_train)} samples")

        print("🔒 Encrypting training data...")
        enc_X_train = self.encrypt_biometric_batch(X_train)

        for epoch in range(epochs):
            print(f"\n🔄 Epoch {epoch + 1}/{epochs}")
            epoch_start = time.time()

            total_loss = 0.0
            successful_samples = 0

            for i, (enc_x, yi) in enumerate(zip(enc_X_train, y_train)):
                try:
                    enc_pred = self.secure_forward_pass(enc_x)
                    pred_val = enc_pred.decrypt()[0]
                    loss = (pred_val - yi) ** 2
                    total_loss += loss
                    successful_samples += 1

                    if i % 10 == 0:
                        print(f"   - Sample {i+1}: pred={pred_val:.4f}, target={yi}, loss={loss:.4f}")

                except Exception as e:
                    print(f"   - Warning: Sample {i} failed: {e}")
                    continue

            if successful_samples > 0:
                avg_loss = total_loss / successful_samples
                epoch_time = time.time() - epoch_start
                print(f"   - Average loss: {avg_loss:.6f} ({successful_samples}/{len(enc_X_train)} samples)")
                print(f"   - Epoch time: {epoch_time:.2f}s")
            else:
                print("   - No successful samples in this epoch")

        self.is_trained = True
        print("✅ Biometric model training completed!")


class PlainBiometricMLP:
    """
    Plain (non-encrypted) MLP for biometric authentication
    Used for comparison with homomorphic version
    """
    
    def __init__(self, input_dim, hidden_dim=32):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.is_trained = False
        
        # Initialize weights
        self.w1 = np.random.normal(0, np.sqrt(1.0 / input_dim), (hidden_dim, input_dim))
        self.b1 = np.zeros(hidden_dim)
        self.w2 = np.random.normal(0, np.sqrt(1.0 / hidden_dim), hidden_dim)
        self.b2 = 0.0
        
        # Data preprocessing
        self.scaler = StandardScaler()
        
        # print(f"📊 Initialized Plain Biometric MLP")
        # print(f"   - Input dimension: {input_dim}")
        # print(f"   - Hidden dimension: {hidden_dim}")
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, x):
        """Forward pass"""
        z1 = np.dot(self.w1, x) + self.b1
        a1 = self.relu(z1)
        z2 = np.dot(self.w2, a1) + self.b2
        return z2, a1
    
    def train(self, X, y, epochs=100, lr=0.01, verbose=True):
        """Train the plain MLP"""
        if verbose:
            print(f"🏋️ Training Plain MLP for {epochs} epochs...")
        
        n = len(X)
        
        for epoch in range(epochs):
            total_loss = 0
            
            # Initialize gradients
            grad_w1 = np.zeros_like(self.w1)
            grad_b1 = np.zeros_like(self.b1)
            grad_w2 = np.zeros_like(self.w2)
            grad_b2 = 0.0
            
            for xi, yi in zip(X, y):
                # Forward pass
                z2, a1 = self.forward(xi)
                pred = self.sigmoid(z2)
                
                # Binary cross-entropy loss
                loss = -(yi * np.log(pred + 1e-15) + (1 - yi) * np.log(1 - pred + 1e-15))
                total_loss += loss
                
                # Backward pass
                error = pred - yi
                
                # Output layer gradients
                grad_w2 += error * a1
                grad_b2 += error
                
                # Hidden layer gradients
                z1 = np.dot(self.w1, xi) + self.b1
                d_hidden = error * self.w2 * self.relu_derivative(z1)
                
                # Input layer gradients
                grad_w1 += np.outer(d_hidden, xi)
                grad_b1 += d_hidden
            
            # Update weights
            self.w1 -= lr * grad_w1 / n
            self.b1 -= lr * grad_b1 / n
            self.w2 -= lr * grad_w2 / n
            self.b2 -= lr * grad_b2 / n
            
            if verbose and (epoch + 1) % 25 == 0:
                avg_loss = total_loss / n
                print(f"   Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        self.is_trained = True
        if verbose:
            print("✅ Plain MLP training completed!")
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            print("⚠️ Model not trained yet!")
            return None, None
        
        predictions = []
        confidence_scores = []
        
        for x in X:
            z2, _ = self.forward(x)
            prob = self.sigmoid(z2)
            confidence_scores.append(prob)
            predictions.append(1 if prob > 0.5 else 0)
        
        return np.array(predictions), np.array(confidence_scores)

class ScaffoldTrainer:
    def __init__(self, client_datasets, model_class,model_kwargs=None, rounds=10, epochs=5, lr=0.01):
        self.client_datasets = client_datasets
        self.model_class = model_class
        self.rounds = rounds
        self.epochs = epochs
        self.model_kwargs = model_kwargs or {}
        self.lr = lr

        input_dim = list(client_datasets.values())[0]["X_train"].shape[1]
        self.global_model = self.model_class(**self.model_kwargs)
        self.c = self._init_control_variate()

    def _init_control_variate(self):
        w = self.extract_weights(self.global_model)
        zero_control = {k: np.zeros_like(v) for k, v in w.items()}
        return zero_control

    def extract_weights(self, model):
        if isinstance(model, (PlainBiometricMLP, BiometricHomomorphicMLP)):
            return {
                "w1": model.w1.copy(),
                "b1": model.b1.copy(),
                "w2": model.w2.copy(),
                "b2": model.b2
            }
        elif isinstance(model, PlainLogisticRegression):
            return {"w": model.w.copy(), "b": model.b}
        else:
            raise NotImplementedError(f"Unsupported model type: {type(model)}")

    def update_weights(self, model, weights):
        if isinstance(model, (PlainBiometricMLP, BiometricHomomorphicMLP)):
            model.w1 = weights["w1"]
            model.b1 = weights["b1"]
            model.w2 = weights["w2"]
            model.b2 = weights["b2"]
        elif isinstance(model, PlainLogisticRegression):
            model.w = weights["w"]
            model.b = weights["b"]
        else:
            raise NotImplementedError(f"Unsupported model type: {type(model)}")


    def copy_weights(self, src_model, dest_model):
        weights = self.extract_weights(src_model)
        self.update_weights(dest_model, weights)

    def control_variate_subtract(self, w, c_global, c_local):
        return {key: w[key] - c_global[key] + c_local[key] for key in w}

    def control_variate_update(self, c_global, c_local, w_old, w_new, lr, epochs):
        c_local_new = {}
        for key in c_global:
            c_local_new[key] = c_local[key] - c_global[key] + (w_old[key] - w_new[key]) / (lr * epochs)
        return c_local_new

    def aggregate_weights(self, client_weights, client_sizes):
        total_samples = sum(client_sizes)
        aggregated = {}
        keys = client_weights[0].keys()
        for key in keys:
            weighted_sum = sum(w[key] * size for w, size in zip(client_weights, client_sizes))
            aggregated[key] = weighted_sum / total_samples
        return aggregated

    def aggregate_control_variates(self, client_controls, client_sizes):
        total_samples = sum(client_sizes)
        aggregated = {}
        keys = client_controls[0].keys()
        for key in keys:
            weighted_sum = sum(c[key] * size for c, size in zip(client_controls, client_sizes))
            aggregated[key] = weighted_sum / total_samples
        return aggregated

    def train(self):
        print(f"🔹 Starting Federated Training with SCAFFOLD for {self.rounds} rounds")

        c_locals = {cid: self._init_control_variate() for cid in self.client_datasets}

        for round_num in range(self.rounds):
            print(f"\n🔄 Federated Round {round_num + 1}/{self.rounds}")
            client_weights = []
            client_sizes = []
            new_c_locals = {}

            for client_id, data in self.client_datasets.items():
                print(f"   - Training on {client_id} ({len(data['X_train'])} samples)")
                local_model = self.model_class(**self.model_kwargs)
                self.copy_weights(self.global_model, local_model)

                # adjusted_weights = self.control_variate_subtract(
                #     self.extract_weights(local_model), self.c, c_locals[client_id]
                # )
                # self.update_weights(local_model, adjusted_weights)

                local_model.train(data["X_train"], data["y_train"], epochs=self.epochs, lr=self.lr, verbose=False)

                w_new = self.extract_weights(local_model)
                w_old = self.extract_weights(self.global_model)

                new_c_locals[client_id] = self.control_variate_update(
                    self.c, c_locals[client_id], w_old, w_new, self.lr, self.epochs
                )

                client_weights.append(w_new)
                client_sizes.append(len(data["X_train"]))

            aggregated_weights = self.aggregate_weights(client_weights, client_sizes)
            self.update_weights(self.global_model, aggregated_weights)

            self.c = self.aggregate_control_variates(list(new_c_locals.values()), client_sizes)
            c_locals = new_c_locals

            print(f"   ✓ Completed Round {round_num + 1}")

        self.global_model.is_trained = True
        print("\n✅ Federated training (SCAFFOLD) completed!")
        return self.global_model


class FedAvgTrainer:
    def __init__(self, client_datasets, model_class, model_kwargs=None,rounds=10, epochs=5, lr=0.01):
        self.client_datasets = client_datasets
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}
        self.rounds = rounds
        self.epochs = epochs
        self.lr = lr
        self.global_model = self.model_class(**self.model_kwargs)

    def extract_weights(self, model):
        if isinstance(model, PlainBiometricMLP):
            return {
                "w1": model.w1.copy(),
                "b1": model.b1.copy(),
                "w2": model.w2.copy(),
                "b2": model.b2
            }
        elif isinstance(model, PlainLogisticRegression):
            return {"w": model.w.copy(), "b": model.b}
        elif isinstance(model, BiometricHomomorphicLogisticRegression):
            return {"w": model.w.copy(), "b": model.b}
        
        elif isinstance(model, BiometricHomomorphicMLP):  # 🔧 Add this block
            return {
                "w1": model.w1.copy(),
                "b1": model.b1.copy(),
                "w2": model.w2.copy(),
                "b2": model.b2
            }
        else:
            raise NotImplementedError("Unsupported model type")


    def update_weights(self, model, weights):
        if isinstance(model, PlainBiometricMLP):
            model.w1 = weights["w1"]
            model.b1 = weights["b1"]
            model.w2 = weights["w2"]
            model.b2 = weights["b2"]
        elif isinstance(model, PlainLogisticRegression):
            model.w = weights["w"]
            model.b = weights["b"]
        elif isinstance(model, BiometricHomomorphicMLP):  # 🔧 Add this block
            model.w1 = weights["w1"]
            model.b1 = weights["b1"]
            model.w2 = weights["w2"]
            model.b2 = weights["b2"]
        elif isinstance(model, BiometricHomomorphicLogisticRegression):  # 🔧 Add this!
            model.w = weights["w"]
            model.b = weights["b"]

        else:
            raise NotImplementedError("Unsupported model type")


    def copy_weights(self, src_model, dest_model):
        weights = self.extract_weights(src_model)
        self.update_weights(dest_model, weights)

    def fedavg_aggregate(self, client_weights, client_sizes):
        total_samples = sum(client_sizes)
        aggregated = {}
        keys = client_weights[0].keys()
        for key in keys:
            weighted_sum = sum(w[key] * size for w, size in zip(client_weights, client_sizes))
            aggregated[key] = weighted_sum / total_samples
        return aggregated

    def train(self):
        print(f"🔹 Starting Federated Averaging for {self.rounds} rounds")

        for round_num in range(self.rounds):
            print(f"\n🔄 Federated Round {round_num + 1}/{self.rounds}")
            client_weights = []
            client_sizes = []

            for client_id, data in self.client_datasets.items():
                print(f"   - Training on {client_id} ({len(data['X_train'])} samples)")

                local_model = self.model_class(**self.model_kwargs)

                self.copy_weights(self.global_model, local_model)

                local_model.train(data["X_train"], data["y_train"], epochs=self.epochs, lr=self.lr, verbose=False)

                client_weights.append(self.extract_weights(local_model))
                client_sizes.append(len(data["X_train"]))

            aggregated_weights = self.fedavg_aggregate(client_weights, client_sizes)
            self.update_weights(self.global_model, aggregated_weights)

            print(f"   ✓ Completed Round {round_num + 1}")

        self.global_model.is_trained = True
        print("\n✅ Federated training (FedAvg) completed!")
        return self.global_model


if __name__ == "__main__":

    data = load_data()

    client_datasets = run_federated_training_with_syft(data)

    
    for i, data in enumerate(client_datasets.values()):
        print(f"Client {i} label distribution: {np.bincount(data['y_train'])}")


    sample_client_data = next(iter(client_datasets.values()))
    input_dim = sample_client_data['X_train'].shape[1]

    # print("\n=== FedAvg with MLP ===")
    fedavg_mlp = FedAvgTrainer(client_datasets, PlainBiometricMLP, model_kwargs={'input_dim': input_dim}, rounds=20, epochs=10, lr=0.01)
    global_mlp = fedavg_mlp.train()
    metrics_mlp_fedavg = evaluate_model(global_mlp, client_datasets)
    # print("FedAvg MLP Metrics:", metrics_mlp_fedavg)

    print("\n=== Scaffold with MLP ===")
    scaffold_mlp = ScaffoldTrainer(client_datasets, PlainBiometricMLP, model_kwargs={'input_dim': input_dim}, rounds=20, epochs=10, lr=0.01)
    global_mlp_scaffold = scaffold_mlp.train()
    metrics_mlp_scaffold = evaluate_model(global_mlp_scaffold, client_datasets)
    # print("Scaffold MLP Metrics:", metrics_mlp_scaffold)


    # print("\n=== FedAvg with Logistic Regression ===")
    fedavg_logreg = FedAvgTrainer(client_datasets, PlainLogisticRegression, model_kwargs={'input_dim': input_dim}, rounds=10, epochs=5, lr=0.01)
    global_logreg = fedavg_logreg.train()
    metrics_logreg_fedavg = evaluate_model(global_logreg, client_datasets)
    # print("FedAvg Logistic Regression Metrics:", metrics_logreg_fedavg)

    # print("\n=== Scaffold with Logistic Regression ===")
    scaffold_logreg = ScaffoldTrainer(client_datasets, PlainLogisticRegression, model_kwargs={'input_dim': input_dim}, rounds=10, epochs=5, lr=0.01)
    global_logreg_scaffold = scaffold_logreg.train()
    metrics_logreg_scaffold = evaluate_model(global_logreg_scaffold, client_datasets)
    print("\n🔐=== Federated Learning with Homomorphic Encryption ===")
    
    # Test 1: FedAvg with Homomorphic Logistic Regression
    print("\n=== FedAvg with Homomorphic Logistic Regression ===")
    try:
        fedavg_he_logreg = FedAvgTrainer(
            client_datasets, 
            BiometricHomomorphicLogisticRegression, 
            model_kwargs={'input_dim': input_dim}, 
            rounds=5,  # Reduced rounds for HE (computationally expensive)
            epochs=3,  # Reduced epochs for HE
            lr=0.01
        )
        global_he_logreg = fedavg_he_logreg.train()
        metrics_he_logreg_fedavg = evaluate_model(global_he_logreg, client_datasets)
        print("✅ FedAvg HE Logistic Regression Metrics:", metrics_he_logreg_fedavg)
    except Exception as e:
        print(f"❌ FedAvg HE Logistic Regression failed: {e}")
        metrics_he_logreg_fedavg = {"error": str(e)}
    
    # Test 2: Scaffold with Homomorphic Logistic Regression
    print("\n=== Scaffold with Homomorphic Logistic Regression ===")
    try:
        scaffold_he_logreg = ScaffoldTrainer(
            client_datasets, 
            BiometricHomomorphicLogisticRegression, 
            model_kwargs={'input_dim': input_dim}, 
            rounds=5, 
            epochs=3, 
            lr=0.01
        )
        global_he_logreg_scaffold = scaffold_he_logreg.train()
        metrics_he_logreg_scaffold = evaluate_model(global_he_logreg_scaffold, client_datasets)
        print("✅ Scaffold HE Logistic Regression Metrics:", metrics_he_logreg_scaffold)
    except Exception as e:
        print(f"❌ Scaffold HE Logistic Regression failed: {e}")
        metrics_he_logreg_scaffold = {"error": str(e)}
    
    # Test 3: FedAvg with Homomorphic MLP
    print("\n=== FedAvg with Homomorphic MLP ===")
    try:
        fedavg_he_mlp = FedAvgTrainer(
            client_datasets, 
            BiometricHomomorphicMLP, 
            model_kwargs={'input_dim': input_dim, 'hidden_dim': 16},  # Smaller hidden dim for HE
            rounds=3,  # Even fewer rounds for MLP HE
            epochs=2, 
            lr=0.005
        )
        global_he_mlp = fedavg_he_mlp.train()
        metrics_he_mlp_fedavg = evaluate_model(global_he_mlp, client_datasets)
        print("✅ FedAvg HE MLP Metrics:", metrics_he_mlp_fedavg)
    except Exception as e:
        print(f"❌ FedAvg HE MLP failed: {e}")
        metrics_he_mlp_fedavg = {"error": str(e)}
    
    # Test 4: Scaffold with Homomorphic MLP
    print("\n=== Scaffold with Homomorphic MLP ===")
    try:
        scaffold_he_mlp = ScaffoldTrainer(
            client_datasets, 
            BiometricHomomorphicMLP, 
            model_kwargs={'input_dim': input_dim, 'hidden_dim': 16}, 
            rounds=3, 
            epochs=2, 
            lr=0.005
        )
        global_he_mlp_scaffold = scaffold_he_mlp.train()
        metrics_he_mlp_scaffold = evaluate_model(global_he_mlp_scaffold, client_datasets)
        print("✅ Scaffold HE MLP Metrics:", metrics_he_mlp_scaffold)
    except Exception as e:
        print(f"❌ Scaffold HE MLP failed: {e}")
        metrics_he_mlp_scaffold = {"error": str(e)}
    
    # Test 5: Demonstration of encrypted inference
    print("\n🔍=== Encrypted Inference Demonstration ===")
    try:
        # Use one of the trained HE models for encrypted inference demo
        if 'global_he_logreg' in locals() and global_he_logreg.is_trained:
            print("Testing encrypted inference with HE Logistic Regression...")
            
            # Get a small test sample
            test_client = list(client_datasets.keys())[0]
            X_test_sample = client_datasets[test_client]['X_test'][:5]  # Just 5 samples
            y_test_sample = client_datasets[test_client]['y_test'][:5]
            
            # Compare plaintext vs encrypted predictions
            global_he_logreg.compare_with_plaintext(X_test_sample, y_test_sample)
            
            # Run encrypted predictions
            enc_preds, enc_confs = global_he_logreg.predict_encrypted(X_test_sample, use_polynomial=False)
            print(f"Encrypted predictions: {enc_preds}")
            print(f"Encrypted confidences: {enc_confs}")
            
    except Exception as e:
        print(f"❌ Encrypted inference demo failed: {e}")
    
    # Compile all results including HE
    all_results = {
        "FedAvg_MLP": metrics_mlp_fedavg,
        "Scaffold_MLP": metrics_mlp_scaffold,
        "FedAvg_LogisticRegression": metrics_logreg_fedavg,
        "Scaffold_LogisticRegression": metrics_logreg_scaffold,
        "FedAvg_HE_LogisticRegression": metrics_he_logreg_fedavg,
        "Scaffold_HE_LogisticRegression": metrics_he_logreg_scaffold,
        "FedAvg_HE_MLP": metrics_he_mlp_fedavg,
        "Scaffold_HE_MLP": metrics_he_mlp_scaffold,
    }
    
    # Save comprehensive results
    with open("comprehensive_federated_learning_results.json", "w") as f:
        json.dump(all_results, f, indent=4)

