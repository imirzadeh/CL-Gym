import os
import ray
import torch
from ray import tune

toy_reg_params_tune = {
    # benchmark
    'num_tasks': 3,
    'batch_size_train': tune.choice([8, 16, 32, 64]),
    'batch_size_memory': 32,
    'batch_size_validation': 128,
    'per_task_memory_examples': 10,
    'per_task_joint_examples': 10,
    'per_task_subset_examples': 70,
    
    # backbone
    'input_dim': 1,
    'hidden_1_dim': tune.choice([10, 20]),
    'hidden_2_dim': tune.choice([10, 20]),
    'hidden_3_dim': 10,
    'hidden_4_dim': 10,
    'output_dim': 1,
    'dropout_prob': tune.uniform(0.00, 0.25),
    'activation': 'Tanh',
    'final_layer_act': tune.choice([False, True]),
    
    # algorithm
    'optimizer': tune.choice(['SGD', 'Adam']),
    'momentum': tune.uniform(0.1, 0.9),
    'epochs_per_task': tune.choice([100, 500, 1000]),
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

toy_reg_params = {
    # benchmark
    'num_tasks': 3,
    'batch_size_train': 32,#tune.choice([8, 16, 32, 64]),
    'batch_size_memory': 32,
    'batch_size_validation': 128,
    'per_task_memory_examples': 10,
    'per_task_joint_examples': 10,
    'per_task_subset_examples': 70,
    
    # backbone
    'input_dim': 1,
    'hidden_1_dim': 20,
    'hidden_2_dim': 20,
    'hidden_3_dim': 10,
    'hidden_4_dim': 10,
    'output_dim': 1,
    'dropout_prob': 0.5,
    'activation': 'Tanh',
    'final_layer_act': True,
    
    # algorithm
    'optimizer': 'SGD',
    'momentum': 0.8,
    'epochs_per_task': 250,
    'learning_rate': 0.01,
    'learning_rate_decay': 1.0,
    'learning_rate_lower_bound': 0.0005,
    'criterion': torch.nn.MSELoss(),  # torch.nn.CrossEntropyLoss(),
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'mcsgd_init_pos': 0.9,
    'mscgd_line_samples': 20,
    'grad_clip_val': 1.0,
    'orm_orthogonal_scale': 2.0
}

toy_clf_params = {
    # benchmark
    'num_tasks': 4,
    'batch_size_train': 32,  # tune.grid_search([8, 16]),
    'batch_size_memory': 32,
    'batch_size_validation': 128,
    'per_task_memory_examples': 10,
    'per_task_joint_examples': 64,
    'per_task_subset_examples': 32,
    
    # backbone
    'input_dim': 2,
    'hidden_1_dim': 16,
    'hidden_2_dim': 16,
    'hidden_3_dim': 32,
    'hidden_4_dim': 32,
    'output_dim': 2,
    'dropout_prob': 0.00,
    'activation': 'ReLU',
    'final_layer_act': False,
    
    # algorithm
    'optimizer': 'SGD',  # tune.choice(['SGD', 'Adam']),
    'momentum': 0.8,
    'epochs_per_task': 10,
    'learning_rate': 0.15,  # tu:e.loguniform(0.001, 0.05),
    'learning_rate_decay': 1.0,  # tune.uniform(0.7, 0.99),
    'learning_rate_lower_bound': 0.0005,
    'criterion':  torch.nn.CrossEntropyLoss(),
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'mcsgd_init_pos': 0.9,
    'mscgd_line_samples': 10,
}


rot_mnist_params = {
    # benchmark
    'num_tasks': 20,
    'batch_size_train': tune.choice([8, 16, 32, 64]),
    'batch_size_memory': 32,
    'batch_size_validation': 256,
    'per_task_memory_examples': 10,
    'per_task_joint_examples': 70,
    'num_dataloader_workers': os.cpu_count()//4,
    
    # backbone
    'input_dim': 784,
    'hidden_1_dim': 256,
    'hidden_2_dim': 256,
    'hidden_3_dim': 256,
    'hidden_4_dim': 256,
    'output_dim': 10,
    'dropout_prob': tune.uniform(0.0, 0.5),
    'activation': 'ReLU',
    'final_layer_act': False,
    
    # algorithm
    'optimizer': tune.choice(['SGD', 'Adam']),
    'momentum': tune.uniform(0.5, 0.9),
    'epochs_per_task': 1,
    'learning_rate':  tune.loguniform(0.001, 0.22),
    'learning_rate_decay': tune.uniform(0.4, 1.0),
    'learning_rate_lower_bound': tune.choice([0.0002, 0.0005]),
    'criterion': torch.nn.CrossEntropyLoss(),
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    # 'grad_clip_val': 0.0,
    'orm_orthogonal_scale': tune.uniform(1, 10)
}

cifar_params = {
    'num_tasks': 20,
    'batch_size_train': 16,  # tune.grid_search([8, 16]),
    'batch_size_memory': 16,
    'batch_size_validation': 256,
    'per_task_memory_examples': 10,
    'per_task_joint_examples': 2000,
    
    # algorithm
    'optimizer': 'SGD',  # tune.choice(['SGD', 'Adam']),
    'momentum': 0.8,
    'epochs_per_task': 2,
    'learning_rate': 0.01,  # tu:e.loguniform(0.001, 0.05),
    'learning_rate_decay': 0.99,  # tune.uniform(0.7, 0.99),
    'learning_rate_lower_bound': 0.0005,
    'criterion': torch.nn.CrossEntropyLoss(),
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
}
