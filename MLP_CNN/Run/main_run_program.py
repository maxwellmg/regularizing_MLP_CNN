

# External Imports
from torch import nn, optim
import pandas as pd
import time
import numpy as np
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Internal Imports
from models.mlp import SimpleMLP
from models.cnn import SimpleCNN

from data.data_loaders import get_loaders
from train_test.training import train
from train_test.testing import test
from regularization.distances import * 
from utils.emissions_tracker import setup_emissions_tracker
from utils.seed_utils import set_global_seed
set_global_seed(42)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

'''output_dir = os.path.abspath("./emissions_data")
os.makedirs(output_dir, exist_ok=True)

emissions_file = os.path.join(output_dir, "emissions_CNN_MNIST.csv")

if os.path.exists(emissions_file):
    os.remove(emissions_file)
    print(f"Removed existing emissions file: {emissions_file}")'''

# Create emissions_data directory in parent directory (same level as Run)
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "emissions_data")
os.makedirs(output_dir, exist_ok=True)

emissions_file = os.path.join(output_dir, "emissions_CNN_MNIST.csv")


def run_experiment(config):
    experiment_results = []
    total_experiments = (len(config['model_types']) * len(config['datasets']) * 
                        len(config['distance_types']) * len(config['lambda_values']))

    print(f"Starting {total_experiments} total experiments")
    
    experiment_count = 0
    
    for model_type in config['model_types']:
        for dataset in config['datasets']:
            
            train_loader, test_loader, in_channels = get_loaders(dataset=dataset, batch_size=config['batch_size'])

            for distance_type in config['distance_types']:
                if distance_type == "baseline":
                    lambda_values = [0.0]
                else:
                    lambda_values = config['lambda_values']

                for lamb in lambda_values:
                    experiment_count += 1
                    print(f"\n{'='*60}")
                    print(f"Experiment {experiment_count}/{total_experiments}")
                    print(f"{'='*60}")

                    # Initialize model based on model_type and dataset for each distance and lambda

                    if dataset == "MNIST":
                        input_height, input_width, in_channels = 28, 28, 1
                    elif dataset == "CIFAR10":
                        input_height, input_width, in_channels = 32, 32, 3

                    if model_type == 'MLP':
                        mlp_input_size = input_height * input_width * in_channels
                        model = SimpleMLP(input_size=mlp_input_size, num_classes=config['num_classes']).to(device)
                    elif model_type == 'CNN':
                        model = SimpleCNN(in_channels, config['num_classes'], input_height, input_width).to(device)

                    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
                    criterion = nn.CrossEntropyLoss()

                    project_name = f"{dataset}_{model_type}_{lamb}_{distance_type}"
                    emissions_file_name = f"{dataset}_{model_type}"

                    print(f"\n Starting: Dataset={dataset}, Model={model_type}, λ={lamb}, reg='{distance_type}'")

                    #AUCC tracker
                    train_accs = []
                    test_accs = []

                    #Call Tracker

                    tracker = setup_emissions_tracker(project_name, emissions_file_name, output_dir)
                    print(f"Starting tracker for {project_name}")
                    tracker.start()

                    for epoch in range(1, config['epochs'] + 1):
                        train_metrics = train(model, train_loader, optimizer, distance_type, criterion, epoch, lamb=lamb)
                        test_metrics = test(model, test_loader, criterion, config['num_classes'])

                        # Accumulate for AUCC
                        train_accs.append(train_metrics['train_acc'])
                        test_accs.append(test_metrics['test_acc'])

                        row = {
                            'dataset': dataset,
                            'model_type': model_type,
                            'lambda': lamb,
                            'distance_type': distance_type,
                            **train_metrics,
                            **test_metrics
                        }

                        experiment_results.append(row)

                        print(f"Epoch {epoch} Summary -- "
                            f"Train Loss: {train_metrics['train_loss']:.4f} | "
                            f"Reg Term: {train_metrics['reg_term']:.4f} | "
                            f"Train Acc: {train_metrics['train_acc']:.2f}% | "
                            f"Test Acc: {test_metrics['test_acc']:.2f}% | "
                            f"Embedding Norm: {train_metrics['embedding_norm']:.4f} | "
                            f"Embedding Sim: {train_metrics['embedding_similarity']:.4f}")

                        #results.append(row)

                    print(f"Stopping tracker for {project_name}")
                    
                    try:
                        emissions_data = tracker.stop()
                        print(f"Tracker stopped successfully for {project_name}")
                        print(f"Emissions data: {emissions_data}")

                        time.sleep(1.0)
                    except Exception as stop_error:
                        print(f"Error stopping tracker: {stop_error}")
                        emissions_data = None
                
                    # Compute AUCC after all epochs
                    # Compute normalized AUCC after all epochs
                    # Convert from percentage (0-100) to decimal (0-1) and normalize by number of epochs
                    train_aucc_normalized = np.trapezoid([acc/100.0 for acc in train_accs], dx=1) / config['epochs']
                    test_aucc_normalized = np.trapezoid([acc/100.0 for acc in test_accs], dx=1) / config['epochs']

                    # Append normalized AUCC to final row
                    experiment_results[-1]['train_aucc'] = round(train_aucc_normalized, 5)
                    experiment_results[-1]['test_aucc'] = round(test_aucc_normalized, 5)     
                   
                    '''try:
                        # Run single experiment
                        experiment_results = run_single_experiment(
                            dataset, model_type, distance_type, lamb, config
                        )
                        all_results.extend(experiment_results)
                        
                        # Save intermediate results periodically
                        if experiment_count % config.get('save_frequency', 10) == 0:
                            df_intermediate = pd.DataFrame(all_results)
                            intermediate_file = f"intermediate_results_{experiment_count}.csv"
                            df_intermediate.to_csv(intermediate_file, index=False)
                            print(f"Saved intermediate results to {intermediate_file}")
                    
                    except Exception as e:
                        print(f"Error in experiment {experiment_count}: {e}")
                        print("Continuing with next experiment...")
                        continue'''
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(experiment_results)
    
    # Create outputs directory path and ensure it exists
    outputs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Save CSV to outputs directory
    output_file = os.path.join(outputs_dir, f'{dataset}_{model_type}.csv')
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")