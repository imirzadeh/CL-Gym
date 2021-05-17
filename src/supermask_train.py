import comet_ml
import torch
from pathlib import Path
import uuid
import os
import cl_gym as cl
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
import os
os.environ['COMET_DISABLE_AUTO_LOGGING'] = '1'

clf_params = {
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
    'hidden_1_dim': 32,
    'hidden_2_dim': 32,
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
    'learning_rate': 0.1,  # tu:e.loguniform(0.001, 0.05),
    'learning_rate_decay': 1.0,  # tune.uniform(0.7, 0.99),
    'learning_rate_lower_bound': 0.0005,
    'criterion': torch.nn.CrossEntropyLoss(),
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'mcsgd_init_pos': 0.9,
    'mscgd_line_samples': 10,
}

mnist_params = {
    # benchmark
    'num_tasks': 5,
    'batch_size_train': 64,  # tune.grid_search([8, 16]),
    'batch_size_memory': 64,
    'batch_size_validation': 128,
    'per_task_memory_examples': 10,
    'per_task_joint_examples': 64,
    'per_task_subset_examples': 10000,
    'per_task_rotation': 22.5,
    
    # backbone
    'input_dim': 784,
    'hidden_1_dim': 100,
    'hidden_2_dim': 100,
    'output_dim': 10,
    'dropout_prob': 0.00,
    'activation': 'ReLU',
    
    # algorithm
    'optimizer': 'SGD',  # tune.choice(['SGD', 'Adam']),
    'momentum': 0.8,
    'epochs_per_task': 5,
    'learning_rate': 0.1,  # tu:e.loguniform(0.001, 0.05),
    'learning_rate_decay': 1.0,  # tune.uniform(0.7, 0.99),
    'learning_rate_lower_bound': 0.0005,
    'criterion': torch.nn.CrossEntropyLoss(),
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'num_dataloader_workers': os.cpu_count() // 4,
    'eval_interval': 'epochs',
    
    # supermask params
    'supermask_train_epochs': 2,
}


def run(params):
    os.environ['COMET_DISABLE_AUTO_LOGGING'] = '1'
    trial_id = str(uuid.uuid4())
    params['trial_id'] = trial_id
    params['output_dir'] = os.path.join("./outputs/{}".format(trial_id))
    Path(params['output_dir']).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(params['output_dir'], 'plots')).mkdir(parents=True, exist_ok=True)
    
    logger = cl.utils.loggers.CometLogger(project_name='debug-2', workspace='cl-boundary', trial_name=trial_id)
    # benchmark = cl.benchmarks.Toy2DClassification(num_tasks=params['num_tasks'],
    #                                               per_task_memory_examples=params['per_task_memory_examples'],
    #                                               per_task_joint_examples=params['per_task_joint_examples'],
    #                                               per_task_subset_examples=params['per_task_subset_examples'])
    
    benchmark = cl.benchmarks.RotatedMNIST(num_tasks=params['num_tasks'],
                                           per_task_rotation=params['per_task_rotation'])
    
    backbone_main = cl.backbones.supermask.MLP(params['input_dim'],
                                               params['hidden_1_dim'],
                                               params['hidden_2_dim'],
                                               params['output_dim'])
    algorithm = cl.algorithms.ContinualAlgorithm(backbone_main, benchmark, params)
    metric_manager_callback = cl.callbacks.MetricManager(num_tasks=params['num_tasks'],
                                                         epochs_per_task=params['epochs_per_task'],
                                                         intervals=params.get('eval_interval', 'epochs'),
                                                         tuner=False)
    supermask_finder_callback = cl.callbacks.SuperMaskFinder(intervals=params.get('eval_interval', 'epochs'))
    trainer_callbacks = [metric_manager_callback, supermask_finder_callback]
    trainer = cl.trainer.ContinualTrainer(algorithm, params, logger=logger, callbacks=trainer_callbacks)
    trainer.run()


if __name__ == "__main__":
    run(mnist_params)
