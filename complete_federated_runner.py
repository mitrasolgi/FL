#!/usr/bin/env python3
"""
Federated Learning Experiments Runner
Demonstrates FedAvg and SCAFFOLD on encrypted and non-encrypted models
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path
import time
from datetime import datetime

# Import our federated learning framework
from federated_learning_framework import (
    FedAvgTrainer, SCAFFOLDTrainer, 
    run_comprehensive_federated_experiments,
    prepare_federated_datasets,
    compare_federated_vs_centralized,
    analyze_convergence
)
from centralized_training import (
    BiometricHomomorphicLogisticRegression, 
    TrulyEncryptedMLP,
    run_centralized_training
)
from syft_utils import load_data, evaluate_biometric_model
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

def run_quick_demo():
    """Run a quick demonstration of federated learning"""
    print("🚀 Quick Federated Learning Demo")
    print("=" * 50)
    
    # Load and prepare data
    data = load_data()
    feature_columns = ['dwell_avg', 'flight_avg', 'traj_avg']
    client_datasets = prepare_federated_datasets(data, feature_columns, num_clients=2)
    
    print(f"\n📊 Demo setup: {len(client_datasets)} clients")
    for client_id, dataset in client_datasets.items():
        print(f"  {client_id}: {len(dataset['X_train'])} train samples")
    
    # Test FedAvg with Logistic Regression
    print("\n🔬 Testing FedAvg with Logistic Regression")
    trainer = FedAvgTrainer(
        model_class=LogisticRegression,
        model_params={'C': 1.0}  # Remove max_iter to avoid conflict
    )
    
    start_time = time.time()
    global_model, history = trainer.train_federated(
        client_datasets,
        num_rounds=5,
        epochs_per_round=2,
        verbose=True
    )
    training_time = time.time() - start_time
    
    # Final evaluation
    final_metrics = trainer.evaluate_global_model(client_datasets)
    
    print(f"\n✅ Demo completed in {training_time:.2f} seconds")
    print(f"📊 Final Results:")
    print(f"   Accuracy: {final_metrics['accuracy']:.3f}")
    print(f"   F1 Score: {final_metrics['f1']:.3f}")
    print(f"   Precision: {final_metrics['precision']:.3f}")
    print(f"   Recall: {final_metrics['recall']:.3f}")
    
    return final_metrics, history

def run_encryption_comparison():
    """Compare encrypted vs non-encrypted federated learning"""
    print("\n🔐 Encryption Comparison Experiment")
    print("=" * 50)
    
    # Load data
    data = load_data()
    feature_columns = ['dwell_avg', 'flight_avg', 'traj_avg']
    client_datasets = prepare_federated_datasets(data, feature_columns, num_clients=3)
    
    results = {}
    
    # Models to compare
    models = {
        'LogisticRegression': {
            'class': LogisticRegression,
            'params': {'C': 1.0},  # Remove max_iter to avoid conflict
            'encrypted': False
        },
        'HomomorphicLogisticRegression': {
            'class': BiometricHomomorphicLogisticRegression,
            'params': {'poly_modulus_degree': 8192, 'scale': 2**35},
            'encrypted': True
        }
    }
    
    for model_name, config in models.items():
        print(f"\n🔬 Testing {model_name} ({'Encrypted' if config['encrypted'] else 'Plain'})")
        
        try:
            # Test with FedAvg
            trainer = FedAvgTrainer(
                model_class=config['class'],
                model_params=config['params']
            )
            
            start_time = time.time()
            global_model, history = trainer.train_federated(
                client_datasets,
                num_rounds=4,
                epochs_per_round=2,
                verbose=False
            )
            training_time = time.time() - start_time
            
            final_metrics = trainer.evaluate_global_model(client_datasets, verbose=False)
            
            results[model_name] = {
                'metrics': final_metrics,
                'training_time': training_time,
                'encrypted': config['encrypted'],
                'history': history
            }
            
            print(f"   ✅ Completed in {training_time:.2f}s")
            print(f"   📊 Accuracy: {final_metrics['accuracy']:.3f}")
            
        except Exception as e:
            print(f"   ❌ Failed: {str(e)[:50]}...")
            results[model_name] = {'error': str(e), 'encrypted': config['encrypted']}
    
    # Print comparison
    print(f"\n📊 Encryption Impact Analysis:")
    plain_acc = results.get('LogisticRegression', {}).get('metrics', {}).get('accuracy', 0)
    enc_acc = results.get('HomomorphicLogisticRegression', {}).get('metrics', {}).get('accuracy', 0)
    
    if plain_acc > 0 and enc_acc > 0:
        acc_diff = enc_acc - plain_acc
        print(f"   Plain Accuracy: {plain_acc:.3f}")
        print(f"   Encrypted Accuracy: {enc_acc:.3f}")
        print(f"   Difference: {acc_diff:+.3f}")
        
        if abs(acc_diff) < 0.05:
            print("   🎯 Encryption preserves accuracy well!")
        elif acc_diff < -0.05:
            print("   ⚠️ Encryption reduces accuracy")
        else:
            print("   📈 Encryption improved accuracy (unexpected)")
    
    return results

def run_algorithm_comparison():
    """Compare FedAvg vs SCAFFOLD algorithms"""
    print("\n⚔️ Algorithm Comparison: FedAvg vs SCAFFOLD")
    print("=" * 50)
    
    # Load data
    data = load_data()
    feature_columns = ['dwell_avg', 'flight_avg', 'traj_avg']
    client_datasets = prepare_federated_datasets(data, feature_columns, num_clients=3)
    
    # Model to test
    model_class = LogisticRegression
    model_params = {'C': 1.0}  # Remove max_iter to avoid conflict
    
    results = {}
    
    # Test FedAvg
    print(f"\n🔬 Testing FedAvg")
    trainer_fedavg = FedAvgTrainer(model_class, model_params)
    start_time = time.time()
    global_model_fedavg, history_fedavg = trainer_fedavg.train_federated(
        client_datasets, num_rounds=6, epochs_per_round=2, verbose=False
    )
    fedavg_time = time.time() - start_time
    fedavg_metrics = trainer_fedavg.evaluate_global_model(client_datasets, verbose=False)
    
    results['FedAvg'] = {
        'metrics': fedavg_metrics,
        'training_time': fedavg_time,
        'history': history_fedavg
    }
    
    # Test SCAFFOLD
    print(f"\n🔬 Testing SCAFFOLD")
    trainer_scaffold = SCAFFOLDTrainer(model_class, model_params)
    start_time = time.time()
    global_model_scaffold, history_scaffold = trainer_scaffold.train_federated(
        client_datasets, num_rounds=6, epochs_per_round=2, lr=0.05, verbose=False
    )
    scaffold_time = time.time() - start_time
    scaffold_metrics = trainer_scaffold.evaluate_global_model(client_datasets, verbose=False)
    
    results['SCAFFOLD'] = {
        'metrics': scaffold_metrics,
        'training_time': scaffold_time,
        'history': history_scaffold
    }
    
    # Print comparison
    print(f"\n📊 Algorithm Comparison Results:")
    print(f"   FedAvg:")
    print(f"     Accuracy: {fedavg_metrics['accuracy']:.3f}")
    print(f"     F1 Score: {fedavg_metrics['f1']:.3f}")
    print(f"     Time: {fedavg_time:.2f}s")
    
    print(f"   SCAFFOLD:")
    print(f"     Accuracy: {scaffold_metrics['accuracy']:.3f}")
    print(f"     F1 Score: {scaffold_metrics['f1']:.3f}")
    print(f"     Time: {scaffold_time:.2f}s")
    
    # Determine winner
    if scaffold_metrics['accuracy'] > fedavg_metrics['accuracy']:
        print(f"   🏆 SCAFFOLD wins by {scaffold_metrics['accuracy'] - fedavg_metrics['accuracy']:.3f}")
    elif fedavg_metrics['accuracy'] > scaffold_metrics['accuracy']:
        print(f"   🏆 FedAvg wins by {fedavg_metrics['accuracy'] - scaffold_metrics['accuracy']:.3f}")
    else:
        print(f"   🤝 Tie!")
    
    return results

def create_performance_visualization(results_file="federated_learning_results.json"):
    """Create visualization of federated learning results"""
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"❌ Results file {results_file} not found")
        return
    
    print("\n📊 Creating Performance Visualization")
    
    # Prepare data for plotting
    plot_data = []
    
    for model_name in results:
        for alg_name in results[model_name]:
            result = results[model_name][alg_name]
            if result.get('status') == 'success':
                metrics = result['final_metrics']
                plot_data.append({
                    'Model': model_name,
                    'Algorithm': alg_name,
                    'Accuracy': metrics.get('accuracy', 0),
                    'F1_Score': metrics.get('f1', 0),
                    'Precision': metrics.get('precision', 0),
                    'Recall': metrics.get('recall', 0)
                })
    
    if not plot_data:
        print("❌ No successful results to plot")
        return
    
    df = pd.DataFrame(plot_data)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Federated Learning Performance Comparison', fontsize=16)
    
    metrics = ['Accuracy', 'F1_Score', 'Precision', 'Recall']
    
    for i, metric in enumerate(metrics):
        ax = axes[i//2, i%2]
        
        # Create grouped bar plot
        models = df['Model'].unique()
        algorithms = df['Algorithm'].unique()
        
        x = np.arange(len(models))
        width = 0.35
        
        for j, alg in enumerate(algorithms):
            alg_data = df[df['Algorithm'] == alg]
            values = [alg_data[alg_data['Model'] == model][metric].values[0] 
                     if len(alg_data[alg_data['Model'] == model]) > 0 else 0 
                     for model in models]
            
            ax.bar(x + j*width, values, width, label=alg, alpha=0.8)
        
        ax.set_xlabel('Model')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} by Model and Algorithm')
        ax.set_xticks(x + width/2)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('federated_learning_performance.png', dpi=300, bbox_inches='tight')
    print("📊 Performance plot saved as 'federated_learning_performance.png'")
    
    # Create convergence plot if history is available
    create_convergence_plot(results)

def create_convergence_plot(results):
    """Create convergence plots for federated algorithms"""
    print("📈 Creating convergence plots")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Federated Learning Convergence', fontsize=16)
    
    plot_idx = 0
    
    for model_name in results:
        if plot_idx >= 4:
            break
            
        ax = axes[plot_idx//2, plot_idx%2]
        
        for alg_name in results[model_name]:
            result = results[model_name][alg_name]
            if result.get('status') == 'success' and 'training_history' in result:
                history = result['training_history']
                if history:
                    rounds = [h['round'] for h in history]
                    accuracies = [h['metrics'].get('accuracy', 0) for h in history]
                    
                    ax.plot(rounds, accuracies, marker='o', label=alg_name, linewidth=2)
        
        ax.set_xlabel('Round')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{model_name} Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, 4):
        axes[i//2, i%2].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('federated_learning_convergence.png', dpi=300, bbox_inches='tight')
    print("📈 Convergence plot saved as 'federated_learning_convergence.png'")

def generate_experiment_report():
    """Generate a comprehensive experiment report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"federated_learning_report_{timestamp}.md"
    
    print(f"\n📝 Generating experiment report: {report_file}")
    
    try:
        with open("federated_learning_results.json", 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print("❌ Results file not found")
        return
    
    report = []
    report.append("# Federated Learning Experiment Report")
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("## Executive Summary")
    report.append("")
    
    # Count successful experiments
    total_experiments = 0
    successful_experiments = 0
    
    for model_name in results:
        for alg_name in results[model_name]:
            total_experiments += 1
            if results[model_name][alg_name].get('status') == 'success':
                successful_experiments += 1
    
    report.append(f"- Total experiments: {total_experiments}")
    report.append(f"- Successful experiments: {successful_experiments}")
    report.append(f"- Success rate: {successful_experiments/total_experiments*100:.1f}%")
    report.append("")
    
    # Detailed results
    report.append("## Detailed Results")
    report.append("")
    
    for model_name in results:
        report.append(f"### {model_name}")
        report.append("")
        
        for alg_name in results[model_name]:
            result = results[model_name][alg_name]
            report.append(f"#### {alg_name}")
            
            if result.get('status') == 'success':
                metrics = result['final_metrics']
                report.append("")
                report.append("| Metric | Value |")
                report.append("|--------|-------|")
                report.append(f"| Accuracy | {metrics.get('accuracy', 0):.4f} |")
                report.append(f"| Precision | {metrics.get('precision', 0):.4f} |")
                report.append(f"| Recall | {metrics.get('recall', 0):.4f} |")
                report.append(f"| F1 Score | {metrics.get('f1', 0):.4f} |")
                report.append(f"| Avg Confidence | {metrics.get('avg_confidence', 0):.4f} |")
                report.append("")
                
                # Add convergence info if available
                if 'training_history' in result and result['training_history']:
                    history = result['training_history']
                    initial_acc = history[0]['metrics'].get('accuracy', 0)
                    final_acc = history[-1]['metrics'].get('accuracy', 0)
                    improvement = final_acc - initial_acc
                    report.append(f"**Convergence:** Improved from {initial_acc:.4f} to {final_acc:.4f} (+{improvement:.4f})")
                    report.append("")
            else:
                report.append("")
                report.append(f"**Status:** FAILED")
                report.append(f"**Error:** {result.get('error', 'Unknown error')}")
                report.append("")
        
        report.append("")
    
    # Best performers
    report.append("## Best Performers")
    report.append("")
    
    best_accuracy = 0
    best_f1 = 0
    best_acc_config = None
    best_f1_config = None
    
    for model_name in results:
        for alg_name in results[model_name]:
            result = results[model_name][alg_name]
            if result.get('status') == 'success':
                metrics = result['final_metrics']
                acc = metrics.get('accuracy', 0)
                f1 = metrics.get('f1', 0)
                
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_acc_config = (model_name, alg_name)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_f1_config = (model_name, alg_name)
    
    if best_acc_config:
        report.append(f"**Best Accuracy:** {best_accuracy:.4f} ({best_acc_config[0]} + {best_acc_config[1]})")
    if best_f1_config:
        report.append(f"**Best F1 Score:** {best_f1:.4f} ({best_f1_config[0]} + {best_f1_config[1]})")
    
    report.append("")
    report.append("## Key Findings")
    report.append("")
    
    # Analyze encryption impact
    plain_models = []
    encrypted_models = []
    
    for model_name in results:
        if 'Homomorphic' in model_name or 'Encrypted' in model_name:
            for alg_name in results[model_name]:
                result = results[model_name][alg_name]
                if result.get('status') == 'success':
                    encrypted_models.append(result['final_metrics'].get('accuracy', 0))
        else:
            for alg_name in results[model_name]:
                result = results[model_name][alg_name]
                if result.get('status') == 'success':
                    plain_models.append(result['final_metrics'].get('accuracy', 0))
    
    if plain_models and encrypted_models:
        avg_plain = np.mean(plain_models)
        avg_encrypted = np.mean(encrypted_models)
        encryption_impact = avg_encrypted - avg_plain
        
        report.append(f"- **Encryption Impact:** Average accuracy difference of {encryption_impact:+.4f}")
        if abs(encryption_impact) < 0.02:
            report.append("  - Encryption preserves model performance well")
        elif encryption_impact < -0.02:
            report.append("  - Encryption reduces performance (expected due to approximations)")
        else:
            report.append("  - Encryption unexpectedly improved performance")
    
    # Analyze algorithm comparison
    fedavg_results = []
    scaffold_results = []
    
    for model_name in results:
        if 'FedAvg' in results[model_name] and results[model_name]['FedAvg'].get('status') == 'success':
            fedavg_results.append(results[model_name]['FedAvg']['final_metrics'].get('accuracy', 0))
        if 'SCAFFOLD' in results[model_name] and results[model_name]['SCAFFOLD'].get('status') == 'success':
            scaffold_results.append(results[model_name]['SCAFFOLD']['final_metrics'].get('accuracy', 0))
    
    if fedavg_results and scaffold_results:
        avg_fedavg = np.mean(fedavg_results)
        avg_scaffold = np.mean(scaffold_results)
        alg_diff = avg_scaffold - avg_fedavg
        
        report.append(f"- **Algorithm Comparison:** SCAFFOLD vs FedAvg difference of {alg_diff:+.4f}")
        if abs(alg_diff) < 0.01:
            report.append("  - Both algorithms perform similarly")
        elif alg_diff > 0.01:
            report.append("  - SCAFFOLD shows superior performance")
        else:
            report.append("  - FedAvg shows superior performance")
    
    report.append("")
    report.append("## Technical Notes")
    report.append("")
    report.append("- All experiments used biometric keystroke dynamics data")
    report.append("- Features: dwell_avg, flight_avg, traj_avg")
    report.append("- Federated setup: Multiple clients with user-based data partitioning")
    report.append("- Encryption: TenSEAL CKKS scheme for homomorphic encryption")
    report.append("- Evaluation: Accuracy, Precision, Recall, F1-Score")
    
    # Write report
    with open(report_file, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"📝 Report saved as {report_file}")

def run_custom_experiment(model_name, algorithm, num_clients=3, num_rounds=5, epochs_per_round=3):
    """Run a custom experiment with specified parameters"""
    print(f"\n🧪 Custom Experiment: {model_name} + {algorithm}")
    print(f"   Clients: {num_clients}, Rounds: {num_rounds}, Epochs: {epochs_per_round}")
    print("-" * 50)
    
    # Model configurations
    model_configs = {
        'LogisticRegression': {
            'class': LogisticRegression,
            'params': {'C': 1.0}  # Remove max_iter to avoid conflict
        },
        'MLPClassifier': {
            'class': MLPClassifier,
            'params': {'hidden_layer_sizes': (32, 16)}  # Remove max_iter to avoid conflict
        },
        'HomomorphicLogisticRegression': {
            'class': BiometricHomomorphicLogisticRegression,
            'params': {'poly_modulus_degree': 8192, 'scale': 2**35}
        },
        'EncryptedMLP': {
            'class': TrulyEncryptedMLP,
            'params': {'hidden_dim': 16}
        }
    }
    
    if model_name not in model_configs:
        print(f"❌ Unknown model: {model_name}")
        print(f"Available models: {list(model_configs.keys())}")
        return None
    
    # Load and prepare data
    data = load_data()
    feature_columns = ['dwell_avg', 'flight_avg', 'traj_avg']
    client_datasets = prepare_federated_datasets(data, feature_columns, num_clients=num_clients)
    
    # Initialize trainer
    model_config = model_configs[model_name]
    
    if algorithm.lower() == 'fedavg':
        trainer = FedAvgTrainer(
            model_class=model_config['class'],
            model_params=model_config['params']
        )
        global_model, history = trainer.train_federated(
            client_datasets,
            num_rounds=num_rounds,
            epochs_per_round=epochs_per_round,
            verbose=True
        )
    elif algorithm.lower() == 'scaffold':
        trainer = SCAFFOLDTrainer(
            model_class=model_config['class'],
            model_params=model_config['params']
        )
        global_model, history = trainer.train_federated(
            client_datasets,
            num_rounds=num_rounds,
            epochs_per_round=epochs_per_round,
            lr=0.05,
            verbose=True
        )
    else:
        print(f"❌ Unknown algorithm: {algorithm}")
        print("Available algorithms: FedAvg, SCAFFOLD")
        return None
    
    # Final evaluation
    final_metrics = trainer.evaluate_global_model(client_datasets)
    
    print(f"\n✅ Custom experiment completed!")
    print(f"📊 Final Results:")
    for metric, value in final_metrics.items():
        print(f"   {metric}: {value:.4f}")
    
    return {
        'model': model_name,
        'algorithm': algorithm,
        'final_metrics': final_metrics,
        'history': history,
        'config': {
            'num_clients': num_clients,
            'num_rounds': num_rounds,
            'epochs_per_round': epochs_per_round
        }
    }

def benchmark_scalability():
    """Benchmark federated learning scalability with different numbers of clients"""
    print("\n⚡ Scalability Benchmark")
    print("=" * 40)
    
    data = load_data()
    feature_columns = ['dwell_avg', 'flight_avg', 'traj_avg']
    
    client_counts = [2, 3, 4, 5]
    results = {}
    
    for num_clients in client_counts:
        print(f"\n🔬 Testing with {num_clients} clients")
        
        try:
            client_datasets = prepare_federated_datasets(data, feature_columns, num_clients=num_clients)
            
            # Test with simple LogisticRegression for speed
            trainer = FedAvgTrainer(
                model_class=LogisticRegression,
                model_params={'C': 1.0}  # Remove max_iter to avoid conflict
            )
            
            start_time = time.time()
            global_model, history = trainer.train_federated(
                client_datasets,
                num_rounds=3,
                epochs_per_round=2,
                verbose=False
            )
            training_time = time.time() - start_time
            
            final_metrics = trainer.evaluate_global_model(client_datasets, verbose=False)
            
            results[num_clients] = {
                'training_time': training_time,
                'accuracy': final_metrics['accuracy'],
                'f1_score': final_metrics['f1']
            }
            
            print(f"   ✅ {num_clients} clients: {training_time:.2f}s, Acc: {final_metrics['accuracy']:.3f}")
            
        except Exception as e:
            print(f"   ❌ {num_clients} clients failed: {str(e)[:50]}...")
            results[num_clients] = {'error': str(e)}
    
    # Analyze scalability
    print(f"\n📊 Scalability Analysis:")
    successful_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if len(successful_results) > 1:
        times = [v['training_time'] for v in successful_results.values()]
        accuracies = [v['accuracy'] for v in successful_results.values()]
        
        print(f"   Training time range: {min(times):.2f}s - {max(times):.2f}s")
        print(f"   Accuracy range: {min(accuracies):.3f} - {max(accuracies):.3f}")
        
        # Simple linear scalability check
        if len(times) >= 2:
            time_ratio = max(times) / min(times)
            client_ratio = max(successful_results.keys()) / min(successful_results.keys())
            efficiency = client_ratio / time_ratio
            print(f"   Scaling efficiency: {efficiency:.2f} (1.0 = linear scaling)")
    
    return results

def main_interactive():
    """Interactive main function with menu"""
    print("🚀 Federated Learning Framework")
    print("=" * 50)
    
    while True:
        print("\n📋 Available Experiments:")
        print("1. Quick Demo (2 clients, LogisticRegression + FedAvg)")
        print("2. Encryption Comparison (Plain vs Homomorphic)")
        print("3. Algorithm Comparison (FedAvg vs SCAFFOLD)")
        print("4. Full Comprehensive Experiments")
        print("5. Custom Experiment")
        print("6. Scalability Benchmark")
        print("7. Generate Visualization")
        print("8. Generate Report")
        print("9. Run Centralized Training (for comparison)")
        print("0. Exit")
        
        choice = input("\nSelect experiment (0-9): ").strip()
        
        try:
            if choice == '1':
                run_quick_demo()
            elif choice == '2':
                run_encryption_comparison()
            elif choice == '3':
                run_algorithm_comparison()
            elif choice == '4':
                run_comprehensive_federated_experiments()
            elif choice == '5':
                model = input("Model (LogisticRegression/MLPClassifier/HomomorphicLogisticRegression/EncryptedMLP): ").strip()
                algorithm = input("Algorithm (FedAvg/SCAFFOLD): ").strip()
                clients = int(input("Number of clients (2-8): ") or "3")
                rounds = int(input("Number of rounds (3-20): ") or "5")
                epochs = int(input("Epochs per round (1-10): ") or "3")
                run_custom_experiment(model, algorithm, clients, rounds, epochs)
            elif choice == '6':
                benchmark_scalability()
            elif choice == '7':
                create_performance_visualization()
            elif choice == '8':
                generate_experiment_report()
            elif choice == '9':
                print("🏗️ Running centralized training for comparison...")
                run_centralized_training()
                print("✅ Centralized training completed. Check 'centralized_results.json'")
            elif choice == '0':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please select 0-9.")
        
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        input("\nPress Enter to continue...")

def save_detailed_metrics(results, filename="detailed_federated_metrics.json"):
    """Save detailed metrics for further analysis"""
    detailed_metrics = {}
    
    for model_name in results:
        detailed_metrics[model_name] = {}
        for alg_name in results[model_name]:
            result = results[model_name][alg_name]
            if result.get('status') == 'success':
                detailed_metrics[model_name][alg_name] = {
                    'final_accuracy': result['final_metrics'].get('accuracy', 0),
                    'final_f1': result['final_metrics'].get('f1', 0),
                    'final_precision': result['final_metrics'].get('precision', 0),
                    'final_recall': result['final_metrics'].get('recall', 0),
                    'avg_confidence': result['final_metrics'].get('avg_confidence', 0),
                    'confidence_std': result['final_metrics'].get('confidence_std', 0),
                    'convergence_rounds': len(result.get('training_history', [])),
                }
    
    with open(filename, 'w') as f:
        json.dump(detailed_metrics, f, indent=4)
    
    print(f"💾 Detailed metrics saved to {filename}")

def run_privacy_analysis():
    """Analyze privacy implications of different approaches"""
    print("\n🔒 Privacy Analysis")
    print("=" * 40)
    
    privacy_analysis = {
        'LogisticRegression': {
            'data_sharing': 'Model parameters shared',
            'privacy_level': 'Low',
            'vulnerabilities': ['Model inversion', 'Membership inference'],
            'mitigations': ['Differential privacy', 'Secure aggregation']
        },
        'MLPClassifier': {
            'data_sharing': 'Model parameters shared',
            'privacy_level': 'Low', 
            'vulnerabilities': ['Model inversion', 'Membership inference'],
            'mitigations': ['Differential privacy', 'Secure aggregation']
        },
        'HomomorphicLogisticRegression': {
            'data_sharing': 'Encrypted computations only',
            'privacy_level': 'High',
            'vulnerabilities': ['Side-channel attacks', 'Timing analysis'],
            'mitigations': ['Secure protocols', 'Noise addition']
        },
        'EncryptedMLP': {
            'data_sharing': 'Encrypted computations only',
            'privacy_level': 'High',
            'vulnerabilities': ['Side-channel attacks', 'Approximation leakage'],
            'mitigations': ['Secure protocols', 'Advanced encryption']
        }
    }
    
    print("Privacy Analysis Summary:")
    print("-" * 60)
    
    for model, analysis in privacy_analysis.items():
        print(f"\n{model}:")
        print(f"  Privacy Level: {analysis['privacy_level']}")
        print(f"  Data Sharing: {analysis['data_sharing']}")
        print(f"  Main Vulnerabilities: {', '.join(analysis['vulnerabilities'])}")
        print(f"  Recommended Mitigations: {', '.join(analysis['mitigations'])}")
    
    return privacy_analysis

def run_performance_benchmarks():
    """Run comprehensive performance benchmarks"""
    print("\n⚡ Performance Benchmarks")
    print("=" * 40)
    
    data = load_data()
    feature_columns = ['dwell_avg', 'flight_avg', 'traj_avg']
    client_datasets = prepare_federated_datasets(data, feature_columns, num_clients=3)
    
    benchmarks = {}
    
    # Models to benchmark
    models = [
        ('LogisticRegression', LogisticRegression, {'max_iter': 50}),
        ('MLPClassifier', MLPClassifier, {'hidden_layer_sizes': (32, 16), 'max_iter': 50}),
        ('HomomorphicLogisticRegression', BiometricHomomorphicLogisticRegression, {'poly_modulus_degree': 8192}),
    ]
    
    # Algorithms to test
    algorithms = ['FedAvg', 'SCAFFOLD']
    
    for model_name, model_class, model_params in models:
        benchmarks[model_name] = {}
        
        for algorithm in algorithms:
            print(f"\n🔬 Benchmarking {model_name} + {algorithm}")
            
            try:
                if algorithm == 'FedAvg':
                    trainer = FedAvgTrainer(model_class, model_params)
                else:
                    trainer = SCAFFOLDTrainer(model_class, model_params)
                
                # Measure training time
                start_time = time.time()
                
                if algorithm == 'FedAvg':
                    global_model, history = trainer.train_federated(
                        client_datasets, num_rounds=3, epochs_per_round=2, verbose=False
                    )
                else:
                    global_model, history = trainer.train_federated(
                        client_datasets, num_rounds=3, epochs_per_round=2, lr=0.05, verbose=False
                    )
                
                training_time = time.time() - start_time
                
                # Measure inference time
                test_sample = list(client_datasets.values())[0]['X_test'][:10]
                
                inference_start = time.time()
                if hasattr(trainer.global_model, 'predict_encrypted'):
                    preds, _ = trainer.global_model.predict_encrypted(test_sample)
                else:
                    preds = trainer.global_model.predict(test_sample)
                inference_time = time.time() - inference_start
                
                # Get final metrics
                final_metrics = trainer.evaluate_global_model(client_datasets, verbose=False)
                
                benchmarks[model_name][algorithm] = {
                    'training_time': training_time,
                    'inference_time': inference_time,
                    'accuracy': final_metrics['accuracy'],
                    'f1_score': final_metrics['f1'],
                    'convergence_rounds': len(history)
                }
                
                print(f"   ✅ Training: {training_time:.2f}s, Inference: {inference_time:.3f}s, Acc: {final_metrics['accuracy']:.3f}")
                
            except Exception as e:
                print(f"   ❌ Failed: {str(e)[:50]}...")
                benchmarks[model_name][algorithm] = {'error': str(e)}
    
    # Summary analysis
    print(f"\n📊 Performance Summary:")
    print("-" * 50)
    
    for model_name in benchmarks:
        print(f"\n{model_name}:")
        for alg_name in benchmarks[model_name]:
            result = benchmarks[model_name][alg_name]
            if 'error' not in result:
                print(f"  {alg_name}: Train={result['training_time']:.2f}s, "
                      f"Inference={result['inference_time']:.3f}s, Acc={result['accuracy']:.3f}")
            else:
                print(f"  {alg_name}: FAILED")
    
    # Save benchmarks
    with open('performance_benchmarks.json', 'w') as f:
        json.dump(benchmarks, f, indent=4, default=str)
    
    print(f"\n💾 Benchmarks saved to 'performance_benchmarks.json'")
    return benchmarks

def create_comprehensive_comparison():
    """Create a comprehensive comparison across all dimensions"""
    print("\n🔍 Comprehensive Comparison Analysis")
    print("=" * 50)
    
    try:
        # Load all available results
        results_files = [
            'federated_learning_results.json',
            'centralized_results.json',
            'performance_benchmarks.json'
        ]
        
        all_results = {}
        for file in results_files:
            try:
                with open(file, 'r') as f:
                    all_results[file] = json.load(f)
                print(f"✅ Loaded {file}")
            except FileNotFoundError:
                print(f"⚠️ {file} not found, skipping")
        
        if not all_results:
            print("❌ No results files found. Run experiments first.")
            return
        
        # Create comparison table
        comparison_data = []
        
        # Federated results
        if 'federated_learning_results.json' in all_results:
            fed_results = all_results['federated_learning_results.json']
            for model_name in fed_results:
                for alg_name in fed_results[model_name]:
                    result = fed_results[model_name][alg_name]
                    if result.get('status') == 'success':
                        metrics = result['final_metrics']
                        comparison_data.append({
                            'Type': 'Federated',
                            'Model': model_name,
                            'Algorithm': alg_name,
                            'Accuracy': metrics.get('accuracy', 0),
                            'F1_Score': metrics.get('f1', 0),
                            'Privacy': 'High' if 'Encrypted' in model_name or 'Homomorphic' in model_name else 'Medium'
                        })
        
        # Centralized results
        if 'centralized_results.json' in all_results:
            cent_results = all_results['centralized_results.json']
            model_mapping = {
                'centralized_mlp': 'MLPClassifier',
                'centralized_logreg': 'LogisticRegression',
                'centralized_encrypted_mlp': 'EncryptedMLP',
                'centralized_encrypted_logreg': 'HomomorphicLogisticRegression'
            }
            
            for key, result in cent_results.items():
                if isinstance(result, dict) and 'accuracy' in result:
                    model_name = model_mapping.get(key, key)
                    comparison_data.append({
                        'Type': 'Centralized',
                        'Model': model_name,
                        'Algorithm': 'Centralized',
                        'Accuracy': result.get('accuracy', 0),
                        'F1_Score': result.get('f1', 0),
                        'Privacy': 'Low'
                    })
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            
            print("\n📊 Accuracy Comparison by Type:")
            print("-" * 40)
            
            pivot_acc = df.pivot_table(
                values='Accuracy', 
                index=['Model'], 
                columns=['Type'], 
                aggfunc='mean'
            ).fillna(0)
            
            print(pivot_acc.round(4))
            
            print("\n📊 F1 Score Comparison by Type:")
            print("-" * 40)
            
            pivot_f1 = df.pivot_table(
                values='F1_Score', 
                index=['Model'], 
                columns=['Type'], 
                aggfunc='mean'
            ).fillna(0)
            
            print(pivot_f1.round(4))
            
            # Privacy vs Performance trade-off
            print("\n🔒 Privacy vs Performance Trade-off:")
            print("-" * 40)
            
            privacy_perf = df.groupby('Privacy').agg({
                'Accuracy': ['mean', 'std'],
                'F1_Score': ['mean', 'std']
            }).round(4)
            
            print(privacy_perf)
            
            # Save comprehensive comparison
            df.to_csv('comprehensive_comparison.csv', index=False)
            print(f"\n💾 Comprehensive comparison saved to 'comprehensive_comparison.csv'")
        
        else:
            print("❌ No valid comparison data found")
            
    except Exception as e:
        print(f"❌ Error in comprehensive comparison: {e}")

def main():
    """Main execution function with extended options"""
    print("🚀 Advanced Federated Learning Framework")
    print("Testing FedAvg and SCAFFOLD on Encrypted and Non-Encrypted Data")
    print("=" * 70)
    
    if len(sys.argv) > 1:
        # Command line mode with extended commands
        command = sys.argv[1].lower()
        
        if command == "demo":
            run_quick_demo()
        elif command == "full":
            run_comprehensive_federated_experiments()
        elif command == "encryption":
            run_encryption_comparison()
        elif command == "algorithms":
            run_algorithm_comparison()
        elif command == "scalability":
            benchmark_scalability()
        elif command == "privacy":
            run_privacy_analysis()
        elif command == "benchmarks":
            run_performance_benchmarks()
        elif command == "comparison":
            create_comprehensive_comparison()
        elif command == "viz":
            create_performance_visualization()
        elif command == "report":
            generate_experiment_report()
        elif command == "centralized":
            run_centralized_training()
        elif command == "all":
            print("🚀 Running ALL experiments...")
            run_centralized_training()
            run_comprehensive_federated_experiments()
            run_performance_benchmarks()
            create_performance_visualization()
            generate_experiment_report()
            create_comprehensive_comparison()
            print("✅ All experiments completed!")
        else:
            print(f"Unknown command: {command}")
            print("Available commands:")
            print("  demo, full, encryption, algorithms, scalability")
            print("  privacy, benchmarks, comparison, viz, report")
            print("  centralized, all")
    else:
        # Interactive mode
        main_interactive()

if __name__ == "__main__":
    import sys
    main()