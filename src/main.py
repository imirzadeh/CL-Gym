import comet_ml
import ray
import torch
from pathlib import Path
import uuid
import os
import cl_gym as cl
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler


def trial(params):
    trial_id = str(uuid.uuid4())
    params['trial_id'] = trial_id
    params['output_dir'] = os.path.join("./outputs/{}".format(trial_id))
    Path(params['output_dir']).mkdir(parents=True, exist_ok=True)

    logger = cl.utils.loggers.CometLogger(project_name='debug-2', workspace='cl-boundary', trial_name=trial_id)
    # benchmark = cl.benchmarks.Toy1DRegression(num_tasks=params['num_tasks'],
    #                                           per_task_memory_examples=params['per_task_memory_examples'])
    benchmark = cl.benchmarks.Toy2DClassification(num_tasks=params['num_tasks'],
                                                  per_task_memory_examples=params['per_task_memory_examples'],
                                                  per_task_joint_examples=params['per_task_joint_examples'])
    
    backbone = cl.backbones.MLP2Layers(input_dim=params['input_dim'],
                                       hidden_dim_1=params['hidden_1_dim'],
                                       hidden_dim_2=params['hidden_2_dim'],
                                       output_dim=params['output_dim'],
                                       dropout_prob=params.get("dropout_prob", 0.0),
                                       activation=params.get("activation", 'ReLU'))
    
    algorithm = cl.algorithms.ContinualAlgorithm(backbone, benchmark, params)
    # algorithm = cl.algorithms.OGD(backbone, benchmark, params)
    
    metric_manager_callback = cl.callbacks.MetricManager(num_tasks=params['num_tasks'],
                                                         epochs_per_task=params['epochs_per_task'])
    # visualizer_callback = cl.callbacks.ToyRegressionVisualizer()
    visualizer_callback = cl.callbacks.ToyClassificationVisualizer()
    model_checkpoint_callback = cl.callbacks.ModelCheckPoint()
    experiment_manager_callback = cl.callbacks.ExperimentManager()
    trainer = cl.trainer.ContinualTrainer(algorithm, params, logger=logger,
                                          callbacks=[metric_manager_callback,
                                                     visualizer_callback,
                                                     model_checkpoint_callback,
                                                     experiment_manager_callback])
    
    trainer.run()
    # tune.report(average_loss=metric_collector_callback.get_final_metric())


if __name__ == "__main__":
    params = {
        # benchmark
        'num_tasks': 4,
        'batch_size_train': 32,#tune.grid_search([8, 16]),
        'batch_size_memory': 32,
        'batch_size_validation': 128,
        'per_task_memory_examples': 10,
        'per_task_joint_examples': 70,
    
        # backbone
        'input_dim': 2,
        'hidden_1_dim': 50,
        'hidden_2_dim': 50,
        'output_dim': 2,
        'dropout_prob': 0.00,
        'activation': 'relu',
    
        # algorithm
        'optimizer': 'SGD', #tune.choice(['SGD', 'Adam']),
        'momentum': 0.8,
        'epochs_per_task': 20,
        'learning_rate': 0.05,#tu:e.loguniform(0.001, 0.05),
        'learning_rate_decay': 0.94,#tune.uniform(0.7, 0.99),
        'learning_rate_lower_bound': 0.0005,
        'criterion': torch.nn.CrossEntropyLoss(),
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'mcsgd_init_pos': 0.9,
        'mscgd_line_samples': 10,
    }
    
    sched = AsyncHyperBandScheduler()
    # analysis = tune.run(trial, metric='average_loss', mode='min',
    #                     scheduler=sched, num_samples=10, config=params)
    # print("Best config is:", analysis.best_config)
    trial(params)
