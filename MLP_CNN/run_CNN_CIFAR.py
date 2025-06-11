# CNN + CIFAR

# CNN + CIFAR
#import os
#import sys

#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Run.main_run_program import run_experiment



'''
full_config = {
    'epochs': 20,
    'batch_size': 64,
    'learning_rate': 3e-4,
    'num_classes': 10,
    'lambda_values': [0.1, 0.25, 0.5, 0.75, 1, 1.25],
    'datasets': ['CIFAR10', 'MNIST'],
    'model_types': ['CNN', 'MLP'],
    'distance_types': ['baseline', 'euclidean', 'manhattan', 'cosine', 'minkowski', 'chebyshev', 'canberra', 'bray-curtis', 'hamming', 'mahalanobis']

}'''

test_config = {
    'epochs': 20,
    'batch_size': 64,
    'learning_rate': 3e-4,
    'num_classes': 10,
    'lambda_values': [0.1, 0.25, 0.5, 0.75, 1, 1.25],
    'datasets': ['CIFAR10'],
    'model_types': ['CNN'],
    'distance_types': ['baseline', 'euclidean', 'manhattan', 'cosine', 'minkowski', 'chebyshev', 'canberra', 'bray-curtis', 'hamming', 'mahalanobis']
}

run_experiment(test_config)