#!/usr/bin/env python3
"""
Fixed PySyft Distributed Federated Learning Example
Demonstrates how to use real distributed clients with PySyft with comprehensive error handling
"""

import numpy as np
from syft_utils import run_federated_training_with_syft, load_data
from federated_learning_framework import FedAvgTrainer, SCAFFOLDTrainer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import time
from centralized_training import BiometricHomomorphicLogisticRegression, TrulyEncryptedMLP
import json
import threading
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def run_distributed_federated_learning(num_clients=3):
    """FIXED distributed federated learning with comprehensive error handling"""
    print(f"🌐 Starting Distributed Federated Learning with {num_clients} PySyft servers")
    print("=" * 60)
    
    # Load data with error handling
    try:
        data = load_data()
        print(f"✅ Successfully loaded data: {len(data)} samples")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return None
    
    # Define ports for PySyft servers (each client gets its own port)
    base_port = 55000
    ports = [base_port + i for i in range(num_clients)]
    
    print(f"📡 Setting up {num_clients} PySyft servers on ports: {ports}")
    
    try:
        # FIXED: This will start actual PySyft servers and distribute data with proper class balancing
        client_datasets = run_federated_training_with_syft(data, ports=ports)
        
        if not client_datasets:
            print("❌ Failed to create client datasets")
            return None
        
        print(f"✅ Successfully set up {len(client_datasets)} distributed clients")
        
        # Verify all clients have both classes
        for client_id, client_data in client_datasets.items():
            train_classes = np.unique(client_data['y_train'])
            test_classes = np.unique(client_data['y_test'])
            print(f"   {client_id}: Train classes {train_classes}, Test classes {test_classes}")
            
            if len(train_classes) < 2:
                print(f"⚠️ {client_id} missing classes in training data")
        
        # Now run federated learning on the distributed setup
        results = {}
        
        # FIXED: Test model combinations with conservative settings
        models = [
            ("LogisticRegression", LogisticRegression, {'C': 0.1, 'max_iter': 1000}, "Plain"),
            ("MLPClassifier", MLPClassifier, {'hidden_layer_sizes': (16, 8), 'max_iter': 100}, "Plain"),
            ("HomomorphicLogisticRegression", BiometricHomomorphicLogisticRegression, 
             {'poly_modulus_degree': 8192, 'scale': 2**30}, "Encrypted"),
            ("EncryptedMLP", TrulyEncryptedMLP, {'hidden_dim': 8}, "Encrypted")
        ]
        
        algorithms = ["FedAvg", "SCAFFOLD"]
        
        for model_name, model_class, model_params, encryption_type in models:
            results[model_name] = {}
            
            for alg_name in algorithms:
                print(f"\n🔬 Testing {model_name} + {alg_name} on PySyft ({encryption_type})")
                
                try:
                    start_time = time.time()
                    
                    if alg_name == "FedAvg":
                        trainer = FedAvgTrainer(
                            model_class=model_class,
                            model_params=model_params
                        )
                        global_model, history = trainer.train_federated(
                            client_datasets,
                            num_rounds=4,  # Reduced for faster testing
                            epochs_per_round=2,
                            verbose=True
                        )
                    else:  # SCAFFOLD
                        trainer = SCAFFOLDTrainer(
                            model_class=model_class,
                            model_params=model_params
                        )
                        global_model, history = trainer.train_federated(
                            client_datasets,
                            num_rounds=4,
                            epochs_per_round=2,
                            lr=0.05,
                            verbose=True
                        )
                    
                    training_time = time.time() - start_time
                    
                    # FIXED: Robust evaluation with error handling
                    try:
                        final_metrics = trainer.evaluate_global_model(client_datasets)
                        
                        results[model_name][alg_name] = {
                            'metrics': final_metrics,
                            'training_time': training_time,
                            'num_clients': len(client_datasets),
                            'encryption_type': encryption_type,
                            'accuracy': final_metrics.get('accuracy', 0.0),
                            'f1_score': final_metrics.get('f1', 0.0),
                            'precision': final_metrics.get('precision', 0.0),
                            'recall': final_metrics.get('recall', 0.0),
                            'status': 'success'
                        }
                        
                        print(f"✅ {model_name} + {alg_name} completed:")
                        print(f"   Accuracy: {final_metrics.get('accuracy', 0):.3f}")
                        print(f"   F1 Score: {final_metrics.get('f1', 0):.3f}")
                        print(f"   Training Time: {training_time:.2f}s")
                        print(f"   Encryption: {encryption_type}")
                        
                    except Exception as eval_error:
                        print(f"⚠️ Evaluation failed for {model_name} + {alg_name}: {eval_error}")
                        results[model_name][alg_name] = {
                            'training_time': training_time,
                            'encryption_type': encryption_type,
                            'status': 'evaluation_failed',
                            'error': str(eval_error)
                        }
                    
                except Exception as e:
                    print(f"❌ {model_name} + {alg_name} failed: {e}")
                    results[model_name][alg_name] = {
                        'error': str(e),
                        'encryption_type': encryption_type,
                        'status': 'failed'
                    }
        
        # Print summary table
        print_pysyft_summary_table(results)
        
        # Save results
        with open('pysyft_complete_results.json', 'w') as f:
            json.dump(results, f, indent=4, default=str)
        
        print(f"\n💾 Results saved to 'pysyft_complete_results.json'")
        return results
        
    except Exception as e:
        print(f"❌ Failed to set up distributed federated learning: {e}")
        print("💡 This might be due to port conflicts or PySyft setup issues")
        return None

def print_pysyft_summary_table(results):
    """Print comprehensive summary table for PySyft results"""
    print("\n📊 PYSYFT DISTRIBUTED FEDERATED LEARNING SUMMARY")
    print("=" * 85)
    
    print(f"{'Model':<25} {'Algorithm':<10} {'Type':<10} {'Accuracy':<10} {'F1':<10} {'Time(s)':<8} {'Status':<10}")
    print("-" * 85)
    
    for model_name in results:
        for alg_name in results[model_name]:
            result = results[model_name][alg_name]
            
            if result['status'] == 'success':
                acc = f"{result['accuracy']:.3f}"
                f1 = f"{result['f1_score']:.3f}"
                time_str = f"{result['training_time']:.1f}"
                status = "✅ Success"
            elif result['status'] == 'evaluation_failed':
                acc = "N/A"
                f1 = "N/A"
                time_str = f"{result.get('training_time', 0):.1f}"
                status = "⚠️ Eval Failed"
            else:
                acc = "N/A"
                f1 = "N/A"
                time_str = "N/A"
                status = "❌ Failed"
            
            enc_type = result.get('encryption_type', 'Unknown')
            
            print(f"{model_name:<25} {alg_name:<10} {enc_type:<10} {acc:<10} {f1:<10} {time_str:<8} {status:<10}")
    
    print("-" * 85)
    print("🌐 All experiments used PySyft distributed servers")

def run_simple_federated_demo(num_clients=3):
    """FIXED simple federated demo without PySyft complexity"""
    print(f"🚀 Simple Federated Learning Demo ({num_clients} clients)")
    print("=" * 50)
    
    try:
        # Load and prepare data
        data = load_data()
        feature_columns = ['dwell_avg', 'flight_avg', 'traj_avg']
        
        # Prepare federated datasets using the fixed approach
        client_datasets = prepare_simple_federated_datasets(data, feature_columns, num_clients)
        
        if not client_datasets:
            print("❌ Failed to create federated datasets")
            return None
        
        # Test with LogisticRegression and FedAvg
        print(f"\n🔬 Testing LogisticRegression with FedAvg")
        
        trainer = FedAvgTrainer(
            model_class=LogisticRegression,
            model_params={'C': 0.1, 'max_iter': 1000}
        )
        
        start_time = time.time()
        global_model, history = trainer.train_federated(
            client_datasets,
            num_rounds=5,
            epochs_per_round=3,
            verbose=True
        )
        training_time = time.time() - start_time
        
        # Evaluate
        final_metrics = trainer.evaluate_global_model(client_datasets)
        
        print(f"\n✅ Demo completed successfully!")
        print(f"   Final Accuracy: {final_metrics['accuracy']:.3f}")
        print(f"   Final F1 Score: {final_metrics['f1']:.3f}")
        print(f"   Training Time: {training_time:.2f}s")
        print(f"   Clients Used: {len(client_datasets)}")
        
        return {
            'metrics': final_metrics,
            'training_time': training_time,
            'num_clients': len(client_datasets),
            'status': 'success'
        }
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return None

def prepare_simple_federated_datasets(data, feature_columns, num_clients):
    """FIXED simple federated dataset preparation with guaranteed class balance"""
    print(f"📂 Preparing {num_clients} federated clients...")
    
    # Clean and prepare data
    X = data[feature_columns].fillna(data[feature_columns].mean()).values
    y = data['label'].values.astype(int)
    
    # Remove invalid samples
    valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X, y = X[valid_mask], y[valid_mask]
    
    print(f"📊 Clean dataset: {len(X)} samples")
    print(f"   Global class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # Verify we have both classes
    if len(np.unique(y)) < 2:
        print("❌ Dataset must contain both classes (0 and 1)")
        return None
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # CRITICAL FIX: Use StratifiedKFold to ensure each client gets both classes
    skf = StratifiedKFold(n_splits=num_clients, shuffle=True, random_state=42)
    
    client_datasets = {}
    
    for i, (_, client_indices) in enumerate(skf.split(X_scaled, y)):
        client_id = f"Client_{i}"
        
        X_client = X_scaled[client_indices]
        y_client = y[client_indices]
        
        # Double-check class presence
        unique_client_labels, client_counts = np.unique(y_client, return_counts=True)
        
        if len(unique_client_labels) < 2:
            print(f"⚠️ {client_id} missing a class, adding samples...")
            # Add minimum samples from missing class
            missing_class = 1 - unique_client_labels[0]
            missing_indices = np.where(y == missing_class)[0]
            if len(missing_indices) >= 3:
                add_indices = np.random.choice(missing_indices, 3, replace=False)
                X_client = np.vstack([X_client, X_scaled[add_indices]])
                y_client = np.hstack([y_client, y[add_indices]])
        
        # Train/test split with stratification
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_client, y_client, 
                test_size=0.2, 
                random_state=42, 
                stratify=y_client
            )
        except ValueError as e:
            print(f"⚠️ Stratification failed for {client_id}, using simple split: {e}")
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
        
        # Verify final class distribution
        train_dist = dict(zip(*np.unique(y_train, return_counts=True)))
        test_dist = dict(zip(*np.unique(y_test, return_counts=True)))
        
        print(f"   ✅ {client_id}: {len(X_train)} train, {len(X_test)} test")
        print(f"      Train classes: {train_dist}, Test classes: {test_dist}")
    
    return client_datasets

def compare_simple_vs_distributed():
    """Compare simple partitioning vs distributed PySyft approach"""
    print("⚔️ Simple Partitioning vs Distributed PySyft Comparison")
    print("=" * 60)
    
    # Method 1: Simple partitioning (current default)
    print("\n📊 Method 1: Simple Data Partitioning")
    print("-" * 40)
    
    simple_start = time.time()
    simple_result = run_simple_federated_demo(num_clients=3)
    simple_time = time.time() - simple_start
    
    if simple_result and simple_result['status'] == 'success':
        simple_metrics = simple_result['metrics']
        simple_training_time = simple_result['training_time']
    else:
        print("❌ Simple method failed")
        return
    
    # Method 2: Distributed PySyft (if it works)
    print("\n🌐 Method 2: Distributed PySyft")
    print("-" * 40)
    
    try:
        distributed_start = time.time()
        distributed_results = run_distributed_federated_learning(num_clients=3)
        distributed_total_time = time.time() - distributed_start
        
        if distributed_results:
            # Get LogisticRegression + FedAvg results for comparison
            lr_fedavg = distributed_results.get('LogisticRegression', {}).get('FedAvg', {})
            
            if lr_fedavg.get('status') == 'success':
                distributed_metrics = lr_fedavg['metrics']
                distributed_training_time = lr_fedavg['training_time']
                
                print(f"\n📊 COMPARISON RESULTS:")
                print(f"{'Method':<20} {'Accuracy':<10} {'F1 Score':<10} {'Time (s)':<10}")
                print("-" * 50)
                print(f"{'Simple Partition':<20} {simple_metrics['accuracy']:<10.3f} {simple_metrics['f1']:<10.3f} {simple_training_time:<10.2f}")
                print(f"{'Distributed PySyft':<20} {distributed_metrics['accuracy']:<10.3f} {distributed_metrics['f1']:<10.3f} {distributed_training_time:<10.2f}")
                
                # Analysis
                acc_diff = distributed_metrics['accuracy'] - simple_metrics['accuracy']
                time_ratio = distributed_training_time / simple_training_time
                
                print(f"\n🔍 ANALYSIS:")
                print(f"   Accuracy difference: {acc_diff:+.3f}")
                print(f"   Time ratio (distributed/simple): {time_ratio:.2f}x")
                
                if abs(acc_diff) < 0.05:
                    print("   ✅ Both methods achieve similar accuracy")
                
                if time_ratio > 2:
                    print("   ⚠️ Distributed approach is significantly slower")
                else:
                    print("   ✅ Distributed overhead is reasonable")
                    
            else:
                print("❌ Distributed method evaluation failed")
                
        else:
            print("❌ Distributed method setup failed")
        
    except Exception as e:
        print(f"❌ Distributed method failed: {e}")
        print("💡 Using simple partitioning is recommended for this setup")

def setup_custom_distributed_clients(num_clients, base_port=55000):
    """Setup custom number of distributed clients with error handling"""
    print(f"🛠️ Setting up {num_clients} custom distributed clients")
    
    try:
        from syft_utils import load_data, start_server, approve_requests
        import threading
        
        # Load and split data
        data = load_data()
        
        # Use our fixed data preparation
        feature_columns = ['dwell_avg', 'flight_avg', 'traj_avg']
        client_datasets = prepare_simple_federated_datasets(data, feature_columns, num_clients)
        
        if not client_datasets:
            print("❌ Failed to prepare client datasets")
            return None, None, None
        
        # Convert to DataFrames for PySyft
        dfs = []
        for i, (client_id, client_data) in enumerate(client_datasets.items()):
            # Combine train and test data for this client
            X_combined = np.vstack([client_data['X_train'], client_data['X_test']])
            y_combined = np.hstack([client_data['y_train'], client_data['y_test']])
            
            df = pd.DataFrame(X_combined, columns=feature_columns)
            df['label'] = y_combined
            df['user_id'] = [f"user_{i}_{j}" for j in range(len(df))]
            
            dfs.append(df)
        
        print(f"✅ Prepared {len(dfs)} client DataFrames")
        
        # Start servers (this would normally use actual PySyft)
        servers = []
        clients = []
        ports = [base_port + i for i in range(num_clients)]
        
        for i, (port, df) in enumerate(zip(ports, dfs)):
            print(f"🚀 Would start server for Client {i+1} on port {port}")
            print(f"   Client {i+1} data shape: {df.shape}")
            print(f"   Client {i+1} classes: {dict(zip(*np.unique(df['label'], return_counts=True)))}")
            
            # For demonstration, we won't actually start PySyft servers
            # In a real implementation, you would use:
            # server, client = start_server(port, df)
            # servers.append(server)
            # clients.append(client)
        
        print(f"\n🎯 Would successfully start {num_clients} distributed servers")
        return servers, clients, ports
        
    except Exception as e:
        print(f"❌ Failed to setup custom distributed clients: {e}")
        return None, None, None

def run_scalability_test():
    """Test scalability with different numbers of clients"""
    print("📈 Federated Learning Scalability Test")
    print("=" * 40)
    
    client_counts = [2, 3, 4, 5]
    results = {}
    
    for num_clients in client_counts:
        print(f"\n🧪 Testing with {num_clients} clients")
        print("-" * 30)
        
        try:
            start_time = time.time()
            result = run_simple_federated_demo(num_clients)
            total_time = time.time() - start_time
            
            if result and result['status'] == 'success':
                results[num_clients] = {
                    'accuracy': result['metrics']['accuracy'],
                    'f1_score': result['metrics']['f1'],
                    'training_time': result['training_time'],
                    'total_time': total_time,
                    'status': 'success'
                }
                print(f"✅ {num_clients} clients: Acc={result['metrics']['accuracy']:.3f}, Time={total_time:.1f}s")
            else:
                results[num_clients] = {'status': 'failed'}
                print(f"❌ {num_clients} clients failed")
                
        except Exception as e:
            results[num_clients] = {'status': 'failed', 'error': str(e)}
            print(f"❌ {num_clients} clients failed: {e}")
    
    # Print scalability summary
    print(f"\n📊 SCALABILITY SUMMARY")
    print("-" * 40)
    print(f"{'Clients':<8} {'Accuracy':<10} {'F1 Score':<10} {'Time (s)':<10} {'Status':<10}")
    print("-" * 48)
    
    for num_clients, result in results.items():
        if result['status'] == 'success':
            print(f"{num_clients:<8} {result['accuracy']:<10.3f} {result['f1_score']:<10.3f} {result['total_time']:<10.1f} {'✅ Success':<10}")
        else:
            print(f"{num_clients:<8} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'❌ Failed':<10}")
    
    return results

def main():
    """Main function demonstrating different approaches with comprehensive error handling"""
    print("🚀 Fixed Federated Learning Client Scaling Demo")
    print("=" * 50)
    
    print("\n📋 Available scaling approaches:")
    print("1. Simple Data Partitioning Demo (Recommended)")
    print("2. Distributed PySyft Servers (Advanced)")
    print("3. Scalability Comparison")
    print("4. Custom Distributed Setup Demo")
    print("5. Simple vs Distributed Comparison")
    print("6. Run All Tests")
    
    choice = input("\nSelect approach (1-6): ").strip()
    
    if choice == "1":
        num_clients = int(input("Number of clients (2-10): ") or "3")
        run_simple_federated_demo(num_clients)
        
    elif choice == "2":
        num_clients = int(input("Number of PySyft clients (2-5): ") or "3")
        run_distributed_federated_learning(num_clients)
        
    elif choice == "3":
        run_scalability_test()
        
    elif choice == "4":
        num_clients = int(input("Number of custom clients (2-8): ") or "4")
        setup_custom_distributed_clients(num_clients)
        
    elif choice == "5":
        compare_simple_vs_distributed()
        
    elif choice == "6":
        print("🧪 Running all tests...")
        
        print("\n1. Simple Demo")
        run_simple_federated_demo(3)
        
        print("\n2. Scalability Test")
        run_scalability_test()
        
        print("\n3. Custom Setup Demo")
        setup_custom_distributed_clients(4)
        
        print("\n✅ All tests completed!")
        
    else:
        print("❌ Invalid choice, running simple demo")
        run_simple_federated_demo(3)

if __name__ == "__main__":
    main()