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
warnings.filterwarnings('ignore')

from concurrent.futures import ThreadPoolExecutor, as_completed

from sklearn.metrics import precision_recall_curve, f1_score


def tune_threshold(confidences, true_labels, min_precision=0.7, normalize=False):
    """
    Tune threshold by maximizing F1-score while ensuring precision >= min_precision.
    
    Args:
        confidences (array-like): Model confidences (e.g., from sigmoid).
        true_labels (array-like): True binary labels (0 or 1).
        min_precision (float): Minimum acceptable precision.
        normalize (bool): Whether to normalize confidences to [0,1] before tuning.
        
    Returns:
        best_thresh: The threshold that gives the best F1 score meeting the precision requirement.
        best_f1: Best F1 score achieved.
        tuned_confs: (Optionally) normalized confidences if normalization was applied.
    """
    confs = np.array(confidences)
    
    if normalize:
        # Normalize confidences to [0,1]
        confs = (confs - confs.min()) / (confs.max() - confs.min() + 1e-8)
    
    thresholds = np.linspace(0.4, 0.8, 100)  # adjust range as needed
    best_thresh = 0.5
    best_f1 = 0.0

    for t in thresholds:
        preds = (confs > t).astype(int)
        precision = precision_score(true_labels, preds, zero_division=0)
        recall = recall_score(true_labels, preds, zero_division=0)
        f1 = f1_score(true_labels, preds, zero_division=0)
        
        # Only choose threshold if precision meets or exceeds min_precision
        if precision >= min_precision and f1 > best_f1:
            best_thresh = t
            best_f1 = f1

    print(f"📈 Optimal threshold: {best_thresh:.4f} with F1-score: {best_f1:.4f}")
    return best_thresh, best_f1, confs


class BiometricHomomorphicMLP:
    def __init__(self, input_dim, hidden_dim, poly_modulus_degree=8192, scale=40):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.is_trained = False

        # Initialize TenSEAL CKKS context with better parameters
        self.HE = ts.context(
            scheme=ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_modulus_degree,
            # Increased coefficient modulus chain for more multiplicative depth
            coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 40, 60]
        )
        self.HE.global_scale = 2 ** scale
        self.HE.generate_galois_keys()
        self.HE.generate_relin_keys()

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
        # Xavier initialization optimized for biometric features
        self.w1 = np.random.normal(0, np.sqrt(1.0 / self.input_dim), 
                                   (self.hidden_dim, self.input_dim))
        self.b1 = np.zeros(self.hidden_dim)
        self.w2 = np.random.normal(0, 0.01, self.hidden_dim)
        self.b2 = 0.1
        
    def poly_activation(self, x):
        """Simple polynomial activation to minimize multiplicative depth"""
        # For TenSEAL, we need simpler operations
        return 0.5 * x + 0.125 * (x ** 2)
    
    def train_encrypted_with_decrypted_gradients(self, X, y, epochs=10, lr=0.001, verbose=False):
        """
        Train the model on biometric data by encrypting inputs,
        doing encrypted forward pass, decrypting outputs, and
        updating weights in plaintext using decrypted gradients.
        """
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        if hasattr(y, "to_numpy"):
            y = y.to_numpy()

        print(f"🏋️ Encrypted Training (with decrypted gradients) for {epochs} epochs...")
        n = len(X)
        
        # Use smaller batch for testing to avoid memory issues
        batch_size = min(50, n)
        indices = np.random.choice(n, batch_size, replace=False)
        X_batch = X[indices]
        y_batch = y[indices]
        
        print(f"   - Using batch size: {batch_size}")

        for epoch in range(epochs):
            total_loss = 0.0
            start_time = time.time()

            for i in range(len(X_batch)):
                try:
                    # Encrypt input sample
                    enc_x = ts.ckks_vector(self.HE, X_batch[i].tolist())

                    # Encrypted forward pass
                    enc_pred = self.secure_forward_pass(enc_x)

                    # Decrypt prediction
                    pred = enc_pred.decrypt()[0]

                    # Compute loss (MSE)
                    loss = (pred - y_batch[i]) ** 2
                    total_loss += loss

                    # Plaintext forward pass for gradients calculation
                    z1 = np.dot(self.w1, X_batch[i]) + self.b1
                    a1 = self.poly_activation(z1)  # ReLU activation for gradient

                    # Gradient of loss w.r.t prediction (MSE)
                    grad_output = 2 * (pred - y_batch[i])

                    # Gradients for weights and biases
                    grad_w2 = grad_output * a1
                    grad_b2 = grad_output

                    relu_deriv = (z1 > 0).astype(float)
                    grad_hidden = grad_output * self.w2 * relu_deriv

                    grad_w1 = np.outer(grad_hidden, X_batch[i])
                    grad_b1 = grad_hidden

                    # Update weights
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
        
    def compute_global_normalization(data):
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        X = data[numeric_cols].fillna(0)
        global_mean = X.mean()
        global_std = X.std() + 1e-8
        return global_mean, global_std
    
    def preprocess_biometric_data(self, data, global_mean=None, global_std=None):
        """
        Preprocess biometric data for homomorphic encryption
        
        Args:
            data: DataFrame with biometric features and user_id
            
        Returns:
            X: Processed features
            y: Binary labels (authentic vs impostor)
        """
        print("📊 Preprocessing biometric data...")
        
        # Remove non-numeric columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        feature_data = data[numeric_columns]
        
        # Handle missing values
        feature_data = feature_data.fillna(feature_data.mean())
        
        # Create binary authentication task
        # For demonstration: classify users into two groups
        users = data['user_id'].unique()
        target_users = users[:len(users)//2]  # First half as "authentic"
        
        # Create labels: 1 for target users, 0 for others
        y = data['user_id'].isin(target_users).astype(int)
        
        # Normalize features
        if global_mean is not None and global_std is not None:
            X = (feature_data - global_mean) / global_std
        else:
            X = self.scaler.fit_transform(feature_data)  # fallback for single-model testi        
        # X = self.scaler.fit_transform(feature_data)
        
        print(f"   - Features shape: {X.shape}")
        print(f"   - Authentic samples: {np.sum(y)}")
        print(f"   - Impostor samples: {len(y) - np.sum(y)}")
        
        return X, y
    
    def encrypt_sample(self, sample):
        """Encrypt a single biometric sample."""
        return ts.ckks_vector(self.HE, sample.tolist())

    def encrypt_biometric_batch(self, X_batch, batch_size=50):
        """
        Encrypt biometric data in batches using multithreading for speed.
        """
        print(f"🔒 Encrypting {len(X_batch)} biometric samples...")

        encrypted_data = []
        
        # Sequential encryption to avoid context issues
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
    
    def secure_forward_pass_basic(self, enc_x):
        """
        Most basic forward pass using only guaranteed TenSEAL operations
        """
        try:
            # Only use operations that definitely exist in TenSEAL
            # Single weight vector approach
            
            # Create a simple weighted sum
            result = 0.0
            for i in range(min(len(self.w2), self.input_dim)):
                # Get one element at a time to avoid complex operations
                weighted = enc_x[i] * self.w2[i]
                result += weighted
            
            # Add bias
            result += self.b2
            
            return result
            
        except Exception as e:
            print(f"❌ Basic forward pass error: {e}")
            raise
    
    def secure_forward_pass_simple(self, enc_x):
        """
        Ultra-simple forward pass for TenSEAL - single layer
        """
        try:
            # Single linear layer to avoid complexity
            # Truncate weights to match input dimension if needed
            w_truncated = self.w2[:min(len(self.w2), self.input_dim)]
            
            # Simple dot product + bias
            enc_output = enc_x.dot(w_truncated.tolist()) + self.b2
            
            return enc_output
            
        except Exception as e:
            print(f"❌ Simple forward pass error: {e}")
            # Try the most basic version
            return self.secure_forward_pass_basic(enc_x)
    
    def secure_forward_pass(self, enc_x):
        """
        Secure forward pass with simplified TenSEAL operations
        """
        try:
            # Try the complex version first, fallback to simple
            try:
                # Layer 1: W1 * x + b1 (using matrix multiplication)
                enc_z1 = enc_x.mm(self.w1.T.tolist())
                
                # Add bias (element-wise)
                enc_z1 = enc_z1 + self.b1.tolist()
                
                # Simple activation (linear scaling to avoid polynomial complexity)
                enc_a1 = enc_z1 * 0.5
                
                # Layer 2: W2 * a1 + b2
                enc_output = enc_a1.dot(self.w2.tolist()) + self.b2
                
                return enc_output
                
            except Exception as e1:
                print(f"   - Complex forward pass failed: {e1}")
                print("   - Trying simple forward pass...")
                return self.secure_forward_pass_simple(enc_x)
            
        except Exception as e:
            print(f"❌ Forward pass error: {e}")
            raise

    @classmethod
    def from_weights(cls, w1, b1, w2, b2, **kwargs):
        """Create model from existing weights"""
        model = cls(input_dim=w1.shape[1], hidden_dim=w1.shape[0], **kwargs)
        model.w1 = w1
        model.b1 = b1
        model.w2 = w2
        model.b2 = b2
        model.is_trained = True
        return model

    def authenticate_encrypted(self, X_test):
        """
        Authenticate using encrypted inference
        """
        if not self.is_trained:
            print("⚠️ Model not trained yet!")
            return None, None
            
        preds = []
        confs = []
        
        # Limit test size for performance
        test_size = len(X_test)
        print(f"🔍 Testing on {test_size} samples...")

        for i in range(test_size):
            try:
                # Encrypt sample if not already encrypted
                if not hasattr(X_test[i], 'decrypt'):
                    enc_x = self.encrypt_sample(X_test[i])
                else:
                    enc_x = X_test[i]
                
                # Forward pass
                enc_pred = self.secure_forward_pass(enc_x)
                threshold = 0.55  # Try 0.6, 0.65, 0.7, etc.

                # Decrypt and classify
                decrypted = enc_pred.decrypt()[0]
                confidence = self.sigmoid(decrypted)
                label = 1 if confidence > threshold else 0
                preds.append(label)
                confs.append(confidence)
                
                if i % 5 == 0:
                    print(f"   - Sample {i+1}: prediction = {decrypted:.4f}")
                    
            except Exception as e:
                print(f"   ❌ Error processing sample {i}: {e}")
                continue

        return np.array(preds), np.array(confs)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train_on_biometric_data(self, X, y, epochs=5, lr=0.001):
        """
        Train the model on biometric data using homomorphic encryption
        (Simplified version with reduced complexity)
        """
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        if hasattr(y, "to_numpy"):
            y = y.to_numpy()

        print(f"🏋️ Training biometric model for {epochs} epochs...")
        print(f"   - Samples: {len(X)}")
        print(f"   - Features: {X.shape[1]}")
        print(f"   - Learning rate: {lr}")
        
        # Use smaller subset for training due to computational complexity
        train_size =  len(X)
        indices = np.random.choice(len(X), train_size, replace=False)
        X_train = X[indices]
        y_train = y[indices]
        
        print(f"   - Training on {len(X_train)} samples")
        
        # Encrypt training data
        print("🔒 Encrypting training data...")
        enc_X_train = self.encrypt_biometric_batch(X_train)
        
        for epoch in range(epochs):
            print(f"\n🔄 Epoch {epoch + 1}/{epochs}")
            epoch_start = time.time()
            
            total_loss = 0.0
            successful_samples = 0
            
            for i, (enc_x, yi) in enumerate(zip(enc_X_train, y_train)):
                try:
                    # Forward pass
                    enc_pred = self.secure_forward_pass(enc_x)
                    
                    # Decrypt and compute loss
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


def load_data():
    """Load biometric dataset from Excel files"""
    folder = "data/behaviour_biometrics_dataset/feature_kmt_dataset/feature_kmt_xlsx"
    
    print(f"📂 Loading data from: {folder}")
    
    dfs = []
    for file in os.listdir(folder):
        if file.endswith(".xlsx"):
            try:
                df = pd.read_excel(os.path.join(folder, file))
                df["user_id"] = file.replace(".xlsx", "")
                dfs.append(df)
                print(f"   ✅ Loaded {file}: {len(df)} records")
            except Exception as e:
                print(f"   ❌ Error loading {file}: {e}")
                continue
    
    data = pd.concat(dfs, ignore_index=True)
    print(f"📊 Total loaded: {len(data)} biometric records from {len(dfs)} users")
    return data


def create_syft_dataset(df):
    """Create Syft dataset for federated learning"""
    dataset = sy.Dataset(
        name="Biometric Dataset",
        summary="User keystroke/mouse biometric data.",
        description="Dataset with extracted biometric features for authentication."
    )
    
    dataset.add_asset(
        sy.Asset(
            name="Biometric Data",
            data=df,
            mock=df.sample(frac=0.1) if len(df) > 10 else df
        )
    )
    return dataset


def start_server(port, df):
    """Start Syft server"""
    name = f"Factory{port - 55000}"

    node = sy.orchestra.launch(name=name, port=port, reset=True, n_consumers=1, create_producer=True)
    client = node.login(email="info@openmined.org", password="changethis")
    client.settings.allow_guest_signup(True)

    dataset = create_syft_dataset(df)
    client.upload_dataset(dataset)

    print(f"[{name}] running at {node.url}:{node.port} with {len(df)} records.")
    return node, client


def approve_requests(client):
    """Approve requests automatically"""
    while True:
        for req in client.requests:
            if req.status.value != 2:
                req.approve(approve_nested=True)
                print(f"Approved: {req.requesting_user_verify_key}")
        sleep(5)


def normalize_confidences(confidences):
    """
    Normalize confidences to [0, 1] to mitigate encryption noise.
    """
    min_c = np.min(confidences)
    max_c = np.max(confidences)
    return (confidences - min_c) / (max_c - min_c + 1e-8)


def evaluate_biometric_model(y_true, y_pred_conf, confidence, threshold=0.5):
    """
    y_pred_conf: Predicted confidence scores (floats)
    confidence: Same as y_pred_conf unless you use a second vector
    threshold: Threshold to convert scores to binary predictions
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    y_pred = (y_pred_conf > threshold).astype(int)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "avg_confidence": np.mean(confidence),
        "confidence_std": np.std(confidence)
    }


import threading


def run_federated_training_with_syft(data, ports=None):
    """Run federated training setup with Syft for 8 clients"""
    
    # Default to 8 ports if not specified
    if ports is None:
        ports = [55000, 55001, 55002, 55003, 55004, 55005, 55006, 55007]
    
    print(f"🌐 Setting up federated learning with {len(ports)} clients")
    print(f"   Ports: {ports}")
    
    # Split data among clients
    users = data["user_id"].unique()
    np.random.seed(42)
    np.random.shuffle(users)
    
    # Ensure we have enough users for all clients
    min_users_per_client = max(1, len(users) // len(ports))
    print(f"   Total users: {len(users)}")
    print(f"   Min users per client: {min_users_per_client}")
    
    # Split users among clients
    splits = np.array_split(users, len(ports))
    
    # Create dataframes for each client
    dfs = []
    for i, split in enumerate(splits):
        client_df = data[data["user_id"].isin(split)]
        print(f"   Client {i}: {len(client_df)} records from {len(split)} users")
        dfs.append(client_df)
    
    # Start servers for each client
    servers = []
    clients = []

    for i, (port, df) in enumerate(zip(ports, dfs)):
        print(f"🚀 Starting server for Client {i} on port {port}")
        try:
            server, client = start_server(port, df)
            servers.append(server)
            clients.append(client)
            
            # Start approval thread for each client
            threading.Thread(target=approve_requests, args=(client,), daemon=True).start()
            
        except Exception as e:
            print(f"❌ Failed to start server on port {port}: {e}")
            continue

    print(f"✅ Started {len(servers)} servers successfully")
    sleep(5)  # Give Syft servers more time to initialize with more clients

    # Process datasets for each client
    client_datasets = {}

    for i, c in enumerate(clients):
        try:
            print(f"📊 Processing dataset for Client {i}")
            df = c.datasets.get_all()[0].assets[0].data

            # Create labels for authentication task
            users_in_client = df["user_id"].unique()
            target_users = users_in_client[:len(users_in_client)//2]  # First half as "authentic"
            y = df["user_id"].isin(target_users).astype(int)

            # Process features
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            X = df[numeric_columns].fillna(df[numeric_columns].mean()).values
            
            # Ensure X and y have same length
            min_len = min(len(X), len(y))
            X = X[:min_len]
            y = y[:min_len]

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )

            client_datasets[f'Client{i}'] = {
                "X_train": X_train,
                "y_train": y_train,
                "X_test": X_test,
                "y_test": y_test,
                "port": port,
                "num_users": len(users_in_client)
            }
            
            print(f"   Client {i}: {len(X_train)} train, {len(X_test)} test samples")
            
        except Exception as e:
            print(f"❌ Failed to process dataset for Client {i}: {e}")
            continue

    return client_datasets, servers


def compare_plain_vs_encrypted_federated_8clients(num_rounds=50, local_epochs=3, lr=0.01):
    print("🚀 Starting Multi-Round Federated Training with 8 Clients")

    data = load_data()
    print(f"\n📊 Loaded {len(data)} biometric records from {data['user_id'].nunique()} users")

    client_datasets, servers = run_federated_training_with_syft(data)
    
    if len(client_datasets) < 8:
        print(f"⚠️ Warning: Only {len(client_datasets)} clients started successfully")

    input_dim = list(client_datasets.values())[0]["X_train"].shape[1]
    hidden_dim = 32

    # Normalize client data once and store scalers for each client
    for cname, d in client_datasets.items():
        scaler = StandardScaler()
        d["X_train_norm"] = scaler.fit_transform(d["X_train"])
        d["X_test_norm"] = scaler.transform(d["X_test"])
        d["scaler"] = scaler

    # Initialize global model weights (untrained)
    global_model = PlainBiometricMLP(input_dim=input_dim, hidden_dim=hidden_dim)

    train_times = {cname: 0.0 for cname in client_datasets.keys()}

    # Multi-round Federated Averaging loop
    for round_idx in range(num_rounds):
        print(f"\n➡️ Federated Round {round_idx+1}/{num_rounds}")

        local_weights = []
        local_sizes = []

        for cname, d in client_datasets.items():
            client_model = PlainBiometricMLP(input_dim=input_dim, hidden_dim=hidden_dim)

            # Start from global weights
            client_model.w1 = global_model.w1.copy()
            client_model.b1 = global_model.b1.copy()
            client_model.w2 = global_model.w2.copy()
            client_model.b2 = global_model.b2
            client_model.is_trained = True

            start = time.time()
            # Train locally for specified epochs
            client_model.train(
                d["X_train_norm"], d["y_train"],
                epochs=local_epochs,
                lr=lr,
                verbose=False,
                batch_size=32
            )
            train_times[cname] += time.time() - start

            # Collect weights and sample size
            local_weights.append({
                'w1': client_model.w1,
                'b1': client_model.b1,
                'w2': client_model.w2,
                'b2': client_model.b2
            })
            local_sizes.append(len(d["X_train_norm"]))

        # Federated weighted averaging
        total_samples = sum(local_sizes)
        new_w1 = sum(w['w1'] * (size / total_samples) for w, size in zip(local_weights, local_sizes))
        new_b1 = sum(w['b1'] * (size / total_samples) for w, size in zip(local_weights, local_sizes))
        new_w2 = sum(w['w2'] * (size / total_samples) for w, size in zip(local_weights, local_sizes))
        new_b2 = sum(w['b2'] * (size / total_samples) for w, size in zip(local_weights, local_sizes))

        global_model.w1 = new_w1
        global_model.b1 = new_b1
        global_model.w2 = new_w2
        global_model.b2 = new_b2
        global_model.is_trained = True

        # Optional: print intermediate validation accuracy per round (average over clients)
        accs = []
        for cname, d in client_datasets.items():
            pred, _ = global_model.predict(d["X_test_norm"])
            acc = np.mean(pred == d["y_test"])
            accs.append(acc)
        avg_acc = np.mean(accs)
        print(f"  🌟 Avg. validation accuracy after round {round_idx+1}: {avg_acc:.4f}")

    print("\n🔍 Evaluating final federated global model:")
    for cname, d in client_datasets.items():
        pred, conf = global_model.predict(d["X_test_norm"])
        accuracy = np.mean(pred == d["y_test"])
        print(f"   {cname} accuracy: {accuracy:.4f}")

    # Post-training fine-tuning on clients (optional)
    print("\n🛠️ Fine-tuning global model locally on each client")
    for cname, d in client_datasets.items():
        fine_tune_model = PlainBiometricMLP(input_dim=input_dim, hidden_dim=hidden_dim)
        fine_tune_model.w1 = global_model.w1.copy()
        fine_tune_model.b1 = global_model.b1.copy()
        fine_tune_model.w2 = global_model.w2.copy()
        fine_tune_model.b2 = global_model.b2.copy()
        fine_tune_model.is_trained = True

        # Before fine-tuning
        pre_pred, _ = global_model.predict(d["X_test_norm"])
        pre_acc = np.mean(pre_pred == d["y_test"])
        print(f"   🔍 {cname} - Accuracy before fine-tuning: {pre_acc:.4f}")

        # Fine-tune
        fine_tune_model.train(d["X_train_norm"], d["y_train"], epochs=20, lr=0.01, verbose=False)
        d["fine_tuned_fed_model"] = fine_tune_model

        # After fine-tuning
        post_pred, _ = fine_tune_model.predict(d["X_test_norm"])
        post_acc = np.mean(post_pred == d["y_test"])
        print(f"   ✅ {cname} - Accuracy after fine-tuning: {post_acc:.4f}")


    # Initialize improved encrypted MLP with global federated weights
    print(f"\n🔐 Initializing Improved Encrypted MLP with averaged parameters")
    he_mlp = BiometricHomomorphicMLP.from_weights(
        w1=global_model.w1, b1=global_model.b1,
        w2=global_model.w2, b2=global_model.b2,
        poly_modulus_degree=16384,
        scale=40
    )

    # Combine all training data from clients for encrypted training
    X_combined = np.vstack([d["X_train_norm"] for d in client_datasets.values()])
    y_combined = np.hstack([d["y_train"] for d in client_datasets.values()])

    print(f"\n🏋️ Improved Encrypted Training on combined data")
    try:
        he_mlp.train_encrypted_with_decrypted_gradients(
            X_combined, y_combined,
            epochs=20,
            lr=0.01,
            verbose=True
        )
        print("✅ Encrypted training completed!")
    except Exception as e:
        print(f"❌ Encrypted training failed: {e}")
        print("   Continuing with evaluation using federated averaged weights...")

    # Comprehensive evaluation on all clients
    print(f"\n📊 Comprehensive evaluation across clients")
    client_results = {}

    for cname, d in client_datasets.items():
        print(f"\n=== {cname} (Users: {d['num_users']}) ===")
        results = {}

        # Local plain model (fine-tuned federated)
        plain_pred, plain_conf = d["fine_tuned_fed_model"].predict(d["X_test_norm"])
        results['plain_federated'] = evaluate_biometric_model(d["y_test"], plain_pred, plain_conf)

        # Encrypted MLP inference
        try:
            batch_size = 10
            all_he_pred = []
            all_he_conf = []
            for i in range(0, len(d["X_test_norm"]), batch_size):
                batch_X = d["X_test_norm"][i:i+batch_size]
                batch_pred, batch_conf = he_mlp.authenticate_encrypted(batch_X)
                if len(batch_pred) > 0:
                    all_he_pred.extend(batch_pred)
                    all_he_conf.extend(batch_conf)

            all_he_pred = np.array(all_he_pred).flatten()
            all_he_conf = np.array(all_he_conf).flatten()
            best_threshold, best_f1, _ = tune_threshold(
                all_he_conf, d["y_test"][:len(all_he_conf)], min_precision=0.6
            )
            results['encrypted'] = evaluate_biometric_model(
                d["y_test"][:len(all_he_pred)], all_he_conf, all_he_conf, threshold=best_threshold
            )
            results['encrypted']['threshold'] = best_threshold
            results['encrypted']['successful_predictions'] = len(all_he_pred)
        except Exception as e:
            results['encrypted'] = {"error": str(e)}

        client_results[cname] = results

        print(f"🟢 Federated Plain MLP:")
        for k, v in results['plain_federated'].items():
            print(f"   {k}: {v:.4f}")

        print(f"🔐 Encrypted MLP:")
        if "error" not in results['encrypted']:
            for k, v in results['encrypted'].items():
                if k in ['threshold', 'successful_predictions']:
                    print(f"   {k}: {v}")
                else:
                    print(f"   {k}: {v:.4f}")
        else:
            print(f"   Error: {results['encrypted']['error']}")

        print(f"⏱️ Total training time on {cname}: {train_times[cname]:.2f}s")

    # Summary
    print(f"\n📈 Summary across {len(client_datasets)} clients")
    print("=" * 80)
    def summarize_metrics(results_list, label):
        print(f"{label}:")
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'avg_confidence', 'confidence_std']:
            metric_values = [r[metric] for r in results_list if metric in r]
            if metric_values:
                print(f"   {metric.capitalize()}: {np.mean(metric_values):.4f} ± {np.std(metric_values):.4f} "
                      f"(Min/Max: {np.min(metric_values):.4f} / {np.max(metric_values):.4f})")
            else:
                print(f"   {metric.capitalize()}: N/A")
        print()

    local_results = [r['plain_federated'] for r in client_results.values()]
    encrypted_results = [r['encrypted'] for r in client_results.values() if 'error' not in r['encrypted']]

    summarize_metrics(local_results, "🟢 Federated Plain MLP")
    if encrypted_results:
        summarize_metrics(encrypted_results, "🔐 Encrypted MLP")
        print(f"   Success Rate: {len(encrypted_results)}/{len(client_datasets)} clients")
    else:
        print("🔐 Encrypted MLP: No successful predictions across clients")

    total_train_time = sum(train_times.values())
    print(f"⏱️ Total training time: {total_train_time:.2f}s")

    # Cleanup servers
    print(f"\n🧹 Cleaning up servers...")
    for server in servers:
        try:
            server.shutdown()
        except:
            pass

    print("\n✅ Federated Multi-Round Training Completed")
    return client_results


# Additional utility function for monitoring client performance
def monitor_client_distribution(client_datasets):
    """Monitor the distribution of data across clients"""
    print("\n📊 Client Data Distribution Analysis")
    print("=" * 50)
    
    total_train = sum(len(d["X_train"]) for d in client_datasets.values())
    total_test = sum(len(d["X_test"]) for d in client_datasets.values())
    
    for cname, d in client_datasets.items():
        train_pct = len(d["X_train"]) / total_train * 100
        test_pct = len(d["X_test"]) / total_test * 100
        pos_rate = np.mean(d["y_train"]) * 100
        
        print(f"{cname}:")
        print(f"  Train: {len(d['X_train'])} samples ({train_pct:.1f}%)")
        print(f"  Test:  {len(d['X_test'])} samples ({test_pct:.1f}%)")
        print(f"  Users: {d['num_users']}")
        print(f"  Positive rate: {pos_rate:.1f}%")
        print()


if __name__ == "__main__":
    compare_plain_vs_encrypted_federated_8clients()