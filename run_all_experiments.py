#!/usr/bin/env python3
"""
Run all combinations: MLP/LogReg + FedAvg/SCAFFOLD + Encrypted/Non-encrypted
"""

from complete_federated_runner import run_custom_experiment
import json
import time

def run_all_combinations():
    """Run all model/algorithm combinations"""
    
    # Define all combinations
    models = [
        "LogisticRegression",           # Non-encrypted
        "MLPClassifier",               # Non-encrypted  
        "HomomorphicLogisticRegression", # Encrypted
        "EncryptedMLP"                 # Encrypted
    ]
    
    algorithms = ["FedAvg", "SCAFFOLD"]
    
    results = {}
    
    for model in models:
        results[model] = {}
        
        for algorithm in algorithms:
            print(f"\n🧪 Running {model} + {algorithm}")
            print("=" * 50)
            
            try:
                result = run_custom_experiment(
                    model_name=model,
                    algorithm=algorithm,
                    num_clients=3,  # Use 3 to avoid single-class issues
                    num_rounds=5,
                    epochs_per_round=3
                )
                
                if result:
                    results[model][algorithm] = {
                        'accuracy': result['final_metrics']['accuracy'],
                        'f1_score': result['final_metrics']['f1'],
                        'precision': result['final_metrics']['precision'],
                        'recall': result['final_metrics']['recall'],
                        'status': 'success'
                    }
                    print(f"✅ {model} + {algorithm}: Acc={result['final_metrics']['accuracy']:.3f}")
                else:
                    results[model][algorithm] = {'status': 'failed', 'error': 'No result returned'}
                    
            except Exception as e:
                print(f"❌ {model} + {algorithm} failed: {e}")
                results[model][algorithm] = {'status': 'failed', 'error': str(e)}
    
    # Save results
    with open('all_combinations_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print summary table
    print_results_table(results)
    
    return results

def print_results_table(results):
    """Print formatted results table"""
    print("\n📊 COMPLETE RESULTS TABLE")
    print("=" * 80)
    
    print(f"{'Model':<25} {'Algorithm':<10} {'Encrypted':<10} {'Accuracy':<10} {'F1 Score':<10} {'Status':<10}")
    print("-" * 80)
    
    for model in results:
        is_encrypted = "Yes" if "Homomorphic" in model or "Encrypted" in model else "No"
        
        for algorithm in results[model]:
            result = results[model][algorithm]
            
            if result['status'] == 'success':
                acc = f"{result['accuracy']:.3f}"
                f1 = f"{result['f1_score']:.3f}"
                status = "✅ Success"
            else:
                acc = "N/A"
                f1 = "N/A" 
                status = "❌ Failed"
            
            print(f"{model:<25} {algorithm:<10} {is_encrypted:<10} {acc:<10} {f1:<10} {status:<10}")
    
    print("-" * 80)

if __name__ == "__main__":
    run_all_combinations()