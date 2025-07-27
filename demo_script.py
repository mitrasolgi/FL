#!/usr/bin/env python3
"""
Quick Demo Script for Federated Learning Framework
Shows how to run specific experiments programmatically
"""

def demo_fedavg_vs_scaffold():
    """Demo comparing FedAvg vs SCAFFOLD"""
    from complete_federated_runner import run_algorithm_comparison
    
    print("🔥 DEMO: FedAvg vs SCAFFOLD Comparison")
    print("=" * 50)
    
    results = run_algorithm_comparison()
    
    # Extract and compare results
    if 'FedAvg' in results and 'SCAFFOLD' in results:
        fedavg_acc = results['FedAvg']['metrics']['accuracy']
        scaffold_acc = results['SCAFFOLD']['metrics']['accuracy']
        
        print(f"\n🏆 WINNER:")
        if scaffold_acc > fedavg_acc:
            print(f"   SCAFFOLD wins with {scaffold_acc:.3f} vs {fedavg_acc:.3f}")
        elif fedavg_acc > scaffold_acc:
            print(f"   FedAvg wins with {fedavg_acc:.3f} vs {scaffold_acc:.3f}")
        else:
            print(f"   It's a tie at {fedavg_acc:.3f}")

def demo_encryption_impact():
    """Demo showing encryption impact on performance"""
    from complete_federated_runner import run_encryption_comparison
    
    print("\n🔐 DEMO: Encryption Impact Analysis")
    print("=" * 50)
    
    results = run_encryption_comparison()
    
    # Analyze encryption impact
    if 'LogisticRegression' in results and 'HomomorphicLogisticRegression' in results:
        plain_result = results['LogisticRegression']
        encrypted_result = results['HomomorphicLogisticRegression']
        
        if 'metrics' in plain_result and 'metrics' in encrypted_result:
            plain_acc = plain_result['metrics']['accuracy']
            enc_acc = encrypted_result['metrics']['accuracy']
            
            print(f"\n📊 ENCRYPTION IMPACT:")
            print(f"   Plain Model Accuracy: {plain_acc:.3f}")
            print(f"   Encrypted Model Accuracy: {enc_acc:.3f}")
            print(f"   Performance Loss: {plain_acc - enc_acc:.3f}")
            
            if abs(plain_acc - enc_acc) < 0.05:
                print(f"   ✅ Minimal impact - encryption is viable!")
            else:
                print(f"   ⚠️  Significant impact - consider optimization")

def demo_custom_experiment():
    """Demo running a custom experiment"""
    from complete_federated_runner import run_custom_experiment
    
    print("\n🧪 DEMO: Custom Experiment")
    print("=" * 50)
    
    # Run SCAFFOLD with Homomorphic Logistic Regression
    result = run_custom_experiment(
        model_name="HomomorphicLogisticRegression",
        algorithm="SCAFFOLD", 
        num_clients=4,
        num_rounds=6,
        epochs_per_round=3
    )
    
    if result:
        print(f"\n🎯 CUSTOM EXPERIMENT RESULTS:")
        print(f"   Model: {result['model']}")
        print(f"   Algorithm: {result['algorithm']}")
        print(f"   Final Accuracy: {result['final_metrics']['accuracy']:.3f}")
        print(f"   Final F1: {result['final_metrics']['f1']:.3f}")

def demo_quick_test():
    """Quick test to verify everything works"""
    from complete_federated_runner import run_quick_demo
    
    print("⚡ DEMO: Quick Functionality Test")
    print("=" * 50)
    
    metrics, history = run_quick_demo()
    
    print(f"\n✅ QUICK TEST RESULTS:")
    print(f"   Accuracy: {metrics['accuracy']:.3f}")
    print(f"   F1 Score: {metrics['f1']:.3f}")
    print(f"   Training Rounds: {len(history)}")

def demo_all_models():
    """Demo all model types with FedAvg"""
    from federated_learning_framework import FedAvgTrainer, prepare_federated_datasets
    from centralized_training import BiometricHomomorphicLogisticRegression, TrulyEncryptedMLP
    from syft_utils import load_data
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    
    print("\n🎭 DEMO: All Model Types with FedAvg")
    print("=" * 50)
    
    # Load data
    data = load_data()
    feature_columns = ['dwell_avg', 'flight_avg', 'traj_avg','hold_mean', 'hold_std','flight_mean', 'flight_std']
    client_datasets = prepare_federated_datasets(data, feature_columns, num_clients=2)
    
    models = [
        ("Logistic Regression", LogisticRegression, {'max_iter': 30}),
        ("MLP Classifier", MLPClassifier, {'hidden_layer_sizes': (16,), 'max_iter': 30}),
        ("Homomorphic LogReg", BiometricHomomorphicLogisticRegression, {'poly_modulus_degree': 8192}),
        ("Encrypted MLP", TrulyEncryptedMLP, {'hidden_dim': 8})
    ]
    
    results = {}
    
    for model_name, model_class, params in models:
        print(f"\n🔬 Testing {model_name}...")
        
        try:
            trainer = FedAvgTrainer(model_class, params)
            global_model, history = trainer.train_federated(
                client_datasets, 
                num_rounds=3, 
                epochs_per_round=2, 
                verbose=False
            )
            
            final_metrics = trainer.evaluate_global_model(client_datasets, verbose=False)
            results[model_name] = final_metrics['accuracy']
            
            print(f"   ✅ Accuracy: {final_metrics['accuracy']:.3f}")
            
        except Exception as e:
            print(f"   ❌ Failed: {str(e)[:50]}...")
            results[model_name] = 0.0
    
    # Show comparison
    print(f"\n📊 MODEL COMPARISON:")
    for model, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"   {model}: {acc:.3f}")

def main():
    """Run all demos"""
    print("🚀 FEDERATED LEARNING FRAMEWORK DEMOS")
    print("=" * 60)
    
    try:
        # Quick functionality test
        demo_quick_test()
        
        # Algorithm comparison
        demo_fedavg_vs_scaffold()
        
        # Encryption impact
        demo_encryption_impact()
        
        # All models test
        demo_all_models()
        
        # Custom experiment
        demo_custom_experiment()
        
        print("\n" + "=" * 60)
        print("🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()