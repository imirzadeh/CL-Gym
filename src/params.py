import os
import ray
import torch
from ray import tune

toy_reg_params = {
    # benchmark
    'num_tasks': 3,
    'batch_size_train': tune.grid_search([8, 16, 32, 64]),
    'batch_size_memory': 32,
    'batch_size_validation': 128,
    'per_task_memory_examples': 10,
    'per_task_joint_examples': 10,
    'per_task_subset_examples': 70,
    
    # backbone
    'input_dim': 1,
    'hidden_1_dim': tune.grid_search([10, 20]),
    'hidden_2_dim': tune.grid_search([10, 20]),
    'hidden_3_dim': 10,
    'hidden_4_dim': 10,
    'output_dim': 1,
    'dropout_prob': tune.uniform(0.00, 0.25),
    'activation': 'Tanh',
    'final_layer_act': tune.grid_search([False, True]),
    
    # algorithm
    'optimizer': tune.choice(['SGD', 'Adam']),
    'momentum': tune.uniform(0.1, 0.9),
    'epochs_per_task': tune.grid_search([100, 500, 1000]),
    'learning_rate': tune.loguniform(0.001, 0.05),
    'learning_rate_decay': tune.uniform(0.7, 0.999),
    'learning_rate_lower_bound': 0.0005,
    'criterion': torch.nn.MSELoss(),  # torch.nn.CrossEntropyLoss(),
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'mcsgd_init_pos': 0.9,
    'mscgd_line_samples': 20,
    'grad_clip_val': tune.uniform(0.0, 1.0),
    'orm_orthogonal_scale': tune.uniform(1, 10)
    
}


toy_clf_params = {
    # benchmark
    'num_tasks': 4,
    'batch_size_train': 32,  # tune.grid_search([8, 16]),
    'batch_size_memory': 32,
    'batch_size_validation': 128,
    'per_task_memory_examples': 10,
    'per_task_joint_examples': 70,
    
    # backbone
    'input_dim': 2,
    'hidden_1_dim': 10,
    'hidden_2_dim': 10,
    'hidden_3_dim': 10,
    'hidden_4_dim': 10,
    'output_dim': 2,
    'dropout_prob': 0.00,
    'activation': 'ReLU',
    'final_layer_act': False,
    
    # algorithm
    'optimizer': 'SGD',  # tune.choice(['SGD', 'Adam']),
    'momentum': 0.8,
    'epochs_per_task': 10,
    'learning_rate': 0.05,  # tu:e.loguniform(0.001, 0.05),
    'learning_rate_decay': 0.94,  # tune.uniform(0.7, 0.99),
    'learning_rate_lower_bound': 0.0005,
    'criterion':  torch.nn.CrossEntropyLoss(),
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'mcsgd_init_pos': 0.9,
    'mscgd_line_samples': 10,
}


rot_mnist_params = {
    # benchmark
    'num_tasks': 5,
    'batch_size_train': 32,  # tune.grid_search([8, 16]),
    'batch_size_memory': 32,
    'batch_size_validation': 128,
    'per_task_memory_examples': 10,
    'per_task_joint_examples': 70,
    'num_dataloader_workers': os.cpu_count()//2,
    
    # backbone
    'input_dim': 784,
    'hidden_1_dim': 100,
    'hidden_2_dim': 100,
    'hidden_3_dim': 100,
    'hidden_4_dim': 100,
    'output_dim': 10,
    'dropout_prob': 0.00,
    'activation': 'ReLU',
    'final_layer_act': False,
    
    # algorithm
    'optimizer': 'SGD',  # tune.choice(['SGD', 'Adam']),
    'momentum': 0.8,
    'epochs_per_task': 5,
    'learning_rate': 0.1,  # tu:e.loguniform(0.001, 0.05),
    'learning_rate_decay': 0.94,  # tune.uniform(0.7, 0.99),
    'learning_rate_lower_bound': 0.0005,
    'criterion': torch.nn.CrossEntropyLoss(),
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'mcsgd_init_pos': 0.9,
    'mscgd_line_samples': 10,
}
