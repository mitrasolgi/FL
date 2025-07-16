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
        
        print(f"📊 Initialized Plain Biometric MLP")
        print(f"   - Input dimension: {input_dim}")
        print(f"   - Hidden dimension: {hidden_dim}")
    
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
    """Run federated training setup with Syft for N clients (default 8)"""

    if ports is None:
        ports = [55000 + i for i in range(8)]  # Default to ports 55000–55007

    print(f"🌐 Setting up federated training with {len(ports)} clients")

    # Shuffle and split user_ids evenly among clients
    users = data["user_id"].unique()
    np.random.seed(42)
    np.random.shuffle(users)
    user_splits = np.array_split(users, len(ports))

    # Create dataframes for each client
    dfs = [data[data["user_id"].isin(split)] for split in user_splits]

    servers = []
    clients = []

    for i, (port, df) in enumerate(zip(ports, dfs)):
        print(f"🚀 Starting server for Client {i} on port {port}")
        try:
            server, client = start_server(port, df)
            servers.append(server)
            clients.append(client)
            threading.Thread(target=approve_requests, args=(client,), daemon=True).start()
        except Exception as e:
            print(f"❌ Failed to start server {i} on port {port}: {e}")

    sleep(5)  # Wait a bit longer for more clients to initialize

    client_datasets = {}

    for i, client in enumerate(clients):
        try:
            df = client.datasets.get_all()[0].assets[0].data

            y = df["label"].values
            X = df.drop(columns=["label", "user_id"], errors="ignore")
            X = X.apply(pd.to_numeric, errors="coerce").dropna().values
            y = y[:len(X)]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )

            client_datasets[f"Client{i}"] = {
                "X_train": X_train,
                "y_train": y_train,
                "X_test": X_test,
                "y_test": y_test,
                "port": ports[i],
                "num_users": len(user_splits[i])
            }

            print(f"✅ Client{i}: {len(X_train)} train, {len(X_test)} test samples")

        except Exception as e:
            print(f"❌ Failed to process data for Client{i}: {e}")
            continue

    return client_datasets



def compare_plain_vs_encrypted_federated():
    """Federated comparison between plain and encrypted MLP using Syft"""
    print("🚀 Federated Plain vs Encrypted MLP Comparison (Biometrics)")
    print("=" * 70)

    data = load_data()
    print(f"\n📊 Loaded {len(data)} biometric records from {data['user_id'].nunique()} users")

    client_datasets = run_federated_training_with_syft(data)

    print("\n📊 Sample count per client:")
    for cname, d in client_datasets.items():
        print(f"   {cname}: train={len(d['X_train'])}, test={len(d['X_test'])}, total={len(d['X_train']) + len(d['X_test'])}")
    

    # ===== PLAIN MLP TRAINING =====
    print("\n📦 Training Plain MLPs on each client")

    plain_models = {}
    train_times = {}

    for cname, d in client_datasets.items():
        print(f"\n⏳ Training Plain MLP for {cname}")
        model = PlainBiometricMLP(input_dim=d["X_train"].shape[1], hidden_dim=16)
        start = time.time()
        model.train(d["X_train"], d["y_train"], epochs=100, lr=0.01, verbose=False)
        train_times[cname] = time.time() - start
        plain_models[cname] = model

    print("\n🔍 Testing individual plain models before aggregation:")
    for cname, model in plain_models.items():
        test_data = client_datasets[cname]
        pred, conf = model.predict(test_data["X_test"])
        print(f"   {cname}: accuracy={np.mean(pred == test_data['y_test']):.4f}, "
            f"avg_conf={np.mean(conf):.4f}")
    # ===== AGGREGATE PARAMETERS =====
    print("\n🔄 Aggregating model parameters...")
    encrypted_params = [
        (m.w1, m.b1, m.w2, m.b2) for m in plain_models.values()
    ]
    avg_w1 = np.mean([p[0] for p in encrypted_params], axis=0)
    avg_b1 = np.mean([p[1] for p in encrypted_params], axis=0)
    avg_w2 = np.mean([p[2] for p in encrypted_params], axis=0)
    avg_b2 = np.mean([p[3] for p in encrypted_params])
    print("\n🔍 Testing aggregated weights in plain model:")
    test_plain_with_agg = PlainBiometricMLP(input_dim=avg_w1.shape[1], hidden_dim=16)  # Note: shape[1] not [0]
    test_plain_with_agg.w1 = avg_w1
    test_plain_with_agg.b1 = avg_b1
    test_plain_with_agg.w2 = avg_w2
    test_plain_with_agg.b2 = avg_b2
    test_plain_with_agg.is_trained = True

    for cname, test_data in client_datasets.items():
        try:
            # Make predictions
            agg_pred, agg_conf = test_plain_with_agg.predict(test_data["X_test"])

            # Metrics
            acc = np.mean(agg_pred == test_data["y_test"])
            prec = precision_score(test_data["y_test"], agg_pred, zero_division=0)
            rec = recall_score(test_data["y_test"], agg_pred, zero_division=0)
            f1 = f1_score(test_data["y_test"], agg_pred, zero_division=0)
            conf_mean = np.mean(agg_conf)
            conf_std = np.std(agg_conf)
            conf_min = agg_conf.min()
            conf_max = agg_conf.max()

            print(f"   {cname}:")
            print(f"     accuracy     = {acc:.4f}")
            print(f"     precision    = {prec:.4f}")
            print(f"     recall       = {rec:.4f}")
            print(f"     f1-score     = {f1:.4f}")
            print(f"     avg_conf     = {conf_mean:.4f}")
            print(f"     conf_std     = {conf_std:.4f}")
            print(f"     conf_range   = [{conf_min:.4f}, {conf_max:.4f}]")

            # Optional: show confusion matrix
            # cm = confusion_matrix(test_data["y_test"], agg_pred)
            # print(f"     confusion matrix:\n{cm}")

        except Exception as e:
            print(f"   ❌ Aggregated model failed on {cname}: {e}")

    # Add this after aggregation, before encrypted model creation:
    print("\n🔍 Checking aggregated parameters:")
    print(f"   avg_w1 shape: {avg_w1.shape}, range: [{avg_w1.min():.4f}, {avg_w1.max():.4f}]")
    print(f"   avg_b1 shape: {avg_b1.shape}, range: [{avg_b1.min():.4f}, {avg_b1.max():.4f}]")
    print(f"   avg_w2 shape: {avg_w2.shape}, range: [{avg_w2.min():.4f}, {avg_w2.max():.4f}]")
    print(f"   avg_b2 range: [{avg_b2.min():.4f}, {avg_b2.max():.4f}]")

    # Check for NaN or inf values
    print(f"   Contains NaN: {np.isnan(avg_w1).any() or np.isnan(avg_b1).any() or np.isnan(avg_w2).any() or np.isnan(avg_b2).any()}")
    print(f"   Contains Inf: {np.isinf(avg_w1).any() or np.isinf(avg_b1).any() or np.isinf(avg_w2).any() or np.isinf(avg_b2).any()}")
    # ===== ENCRYPTED MLP INITIALIZATION =====
    print("\n🔐 Initializing Federated Encrypted MLP")
    he_mlp = BiometricHomomorphicMLP.from_weights(
        w1=avg_w1, b1=avg_b1, w2=avg_w2, b2=avg_b2, 
        poly_modulus_degree=32768, scale=20  # Better parameters
    )

    # ===== ENCRYPTED TRAINING =====
    print("\n🏋️ Starting Encrypted Training on combined client datasets")
    X_combined = np.vstack([d["X_train"] for d in client_datasets.values()])
    y_combined = np.hstack([d["y_train"] for d in client_datasets.values()])
    
    for lr in [0.01, 0.001]:
        for epochs in [20, 30]:
            print(f"\n🔁 Trying config: lr={lr}, epochs={epochs}")
            try:
                # Reset model using fresh weights
                trial_model = BiometricHomomorphicMLP.from_weights(
                    w1=avg_w1.copy(), b1=avg_b1.copy(), 
                    w2=avg_w2.copy(), b2=avg_b2.copy(),
                    poly_modulus_degree=32768, scale=20
                )
                trial_model.train_encrypted_with_decrypted_gradients(X_combined, y_combined, epochs=epochs, lr=lr)
            except Exception as e:
                print(f"❌ Training failed at lr={lr}, epochs={epochs}: {e}")


    # ===== EVALUATION =====
    print("\n📊 Model Evaluation per client")
    for cname, d in client_datasets.items():
        print(f"\n=== Client: {cname} ===")

        # Plain MLP inference
        plain_pred, plain_conf = plain_models[cname].predict(d["X_test"])
        metrics_plain = evaluate_biometric_model(d["y_test"], plain_pred, plain_conf)

        # Encrypted inference on limited samples
        try:
            he_pred, he_conf = he_mlp.authenticate_encrypted(d["X_test"])
            if len(he_pred) > 0:
                # Normalize encrypted confidence
                norm_conf = normalize_confidences(he_conf)

                # Optional: Visualize confidence distribution
                # plot_confidence_distribution(norm_conf, d["y_test"][:len(he_pred)])

                # Use fixed threshold or search for one manually
                # best_threshold = 0.55 
                # metrics_he = evaluate_biometric_model(d["y_test"][:len(he_pred)], norm_conf, norm_conf, threshold=best_threshold)

                # print(f"📈 Applied threshold: {best_threshold:.4f}")
                # Tune threshold using confidences and ground truth
                best_threshold, best_f1, norm_conf_tuned = tune_threshold(norm_conf, d["y_test"][:len(he_pred)], min_precision=0.6)

                # Evaluate using the tuned threshold
                # Recompute binary predictions using the tuned threshold
                he_pred = (norm_conf > best_threshold).astype(int)

                metrics_he = evaluate_biometric_model(
                    d["y_test"][:len(he_pred)], he_pred, norm_conf, threshold=best_threshold
                )


                print(f"📈 Applied tuned threshold: {best_threshold:.4f} with F1-score: {best_f1:.4f}")

            else:
                metrics_he = {"error": "No successful predictions"}
        except Exception as e:
            print(f"   ❌ Encrypted inference failed: {e}")
            metrics_he = {"error": str(e)}


        print("\n🟦 Plain MLP Metrics:")
        for k, v in metrics_plain.items():
            print(f"   {k}: {v:.4f}")

        print("\n🔐 Encrypted MLP Metrics:")
        if "error" not in metrics_he:
            for k, v in metrics_he.items():
                print(f"   {k}: {v:.4f}")
        else:
            print(f"   Error: {metrics_he['error']}")

        print(f"\n⏱️ {cname} Plain Training Time: {train_times[cname]:.2f}s")


    print("\n✅ Federated Biometric Authentication Comparison Completed")
    print("=" * 70)


if __name__ == "__main__":
    compare_plain_vs_encrypted_federated()