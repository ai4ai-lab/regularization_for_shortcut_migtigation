import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import argparse
import os

from data.utils import data_loader
from lib.train import *
from lib.regularization import loss_reg

if __name__ == "__main__":
    # Add argument parser
    parser = argparse.ArgumentParser(description='Training with different regularization methods')
    parser.add_argument('--dataset', type=str, choices=['color_mnist', 'multinli', 'mimic'],
                      required=True, help='Dataset to use')
    parser.add_argument('--device', type=str, default='cpu',
                      help='Device to run on (e.g., cpu, cuda)')
    parser.add_argument('--reg_strengths', type=float, nargs='+', 
                      default=[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                      help='List of regularization strengths to test')
    parser.add_argument('--num_runs', type=int, default=10,
                      help='Number of times to run each experiment')
    parser.add_argument('--num_epochs', type=int, default=20,
                      help='Number of epochs to train each model')

    args = parser.parse_args()

    # Set dataset-specific path
    dataset_paths = {
        'color_mnist': "./data/colored_mnist",
        'multinli': "./data/multinli",
        'mimic': "./data/mimic_icu"}
    
    path = dataset_paths[args.dataset]
    dataset = data_loader(args.dataset, path)

    optimizer = optim.Adam
    optimizer_params = {'lr': 0.01, 'weight_decay': 0.0}
    scheduler_params={'step_size': 1000, 'gamma': 0.9}
    num_epochs = args.num_epochs


    reg_strength_list = args.reg_strengths
    experiment_strength_all = {}
    for idx in range(args.num_runs):
        experiment_strength_all[idx] = experiment_regstrength(reg_strength_list, data_loader=dataset, device=args.device,
                                                               optimizer=optimizer, optimizer_params=optimizer_params,
                                                               scheduler_params=scheduler_params, num_epochs=num_epochs,)
    
    # Create results directory if it doesn't exist
    result_dir = f"results"

    # Save results in the appropriate directory
    torch.save(experiment_strength_all, f"{result_dir}/{args.dataset}_experiment.pt")
