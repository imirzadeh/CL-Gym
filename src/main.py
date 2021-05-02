import comet_ml
import ray
import torch
from pathlib import Path
import uuid
import os
import cl_gym as cl
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
import os
os.environ['COMET_DISABLE_AUTO_LOGGING'] = '1'


def trial_toy_regression(params):
    trial_id = str(uuid.uuid4())
    logger = cl.utils.loggers.CometLogger(project_name='orm-pilot', workspace='cl-boundary', trial_name=trial_id)
    params['trial_id'] = trial_id
    params['output_dir'] = os.path.join("./outputs/{}".format(trial_id))
    Path(params['output_dir']).mkdir(parents=True, exist_ok=True)

    benchmark = cl.benchmarks.Toy1DRegression(num_tasks=params['num_tasks'],
                                              per_task_subset_examples=params['per_task_subset_examples'],
                                              per_task_joint_examples=params['per_task_joint_examples'],
                                              per_task_memory_examples=params['per_task_memory_examples'])
    # benchmark = cl.benchmarks.Toy2DClassification(num_tasks=params['num_tasks'],
    #                                               per_task_memory_examples=params['per_task_memory_examples'],
    #                                               per_task_joint_examples=params['per_task_joint_examples'])
    
    backbone = cl.backbones.MLP2Layers(input_dim=params['input_dim'],
                                       hidden_dim_1=params['hidden_1_dim'],
                                       hidden_dim_2=params['hidden_2_dim'],
                                       output_dim=params['output_dim'],
                                       dropout_prob=params.get("dropout_prob", 0.0),
                                       activation=params.get("activation", 'ReLU'),
                                       include_final_layer_act=params.get('final_layer_act', False))
    
    # algorithm = cl.algorithms.ContinualAlgorithm(backbone, benchmark, params)
    # algorithm = cl.algorithms.OGD(backbone, benchmark, params)
    # algorithm = cl.algorithms.MCSGD(backbone, benchmark, params)
    algorithm = cl.algorithms.ORM(backbone, benchmark, params)
    
    metric_manager_callback = cl.callbacks.MetricManager(num_tasks=params['num_tasks'],
                                                         epochs_per_task=params['epochs_per_task'])
    visualizer_callback = cl.callbacks.ToyRegressionVisualizer()
    # visualizer_callback = cl.callbacks.ToyClassificationVisualizer()
    model_checkpoint_callback = cl.callbacks.ModelCheckPoint()
    experiment_manager_callback = cl.callbacks.ExperimentManager()
    trainer = cl.trainer.ContinualTrainer(algorithm, params, logger=logger,
                                          callbacks=[metric_manager_callback,
                                                     visualizer_callback,
                                                     model_checkpoint_callback,
                                                     experiment_manager_callback])
    
    trainer.run()
    # tune.report(average_loss=metric_manager_callback.get_final_metric())


def trial_toy_classification(params):
    trial_id = str(uuid.uuid4())
    params['trial_id'] = trial_id
    params['output_dir'] = os.path.join("./outputs/{}".format(trial_id))
    Path(params['output_dir']).mkdir(parents=True, exist_ok=True)
    
    logger = cl.utils.loggers.CometLogger(project_name='debug-2', workspace='cl-boundary', trial_name=trial_id)
    benchmark = cl.benchmarks.Toy2DClassification(num_tasks=params['num_tasks'],
                                                  per_task_memory_examples=params['per_task_memory_examples'],
                                                  per_task_joint_examples=params['per_task_joint_examples'])
    
    backbone = cl.backbones.MLP2Layers(input_dim=params['input_dim'],
                                       hidden_dim_1=params['hidden_1_dim'],
                                       hidden_dim_2=params['hidden_2_dim'],
                                       output_dim=params['output_dim'],
                                       dropout_prob=params.get("dropout_prob", 0.0),
                                       activation=params.get("activation", 'ReLU'),
                                       include_final_layer_act=params.get('final_layer_act', False))
    
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


def trial_rot_mnist(params):
    trial_id = str(uuid.uuid4())
    params['trial_id'] = trial_id
    params['output_dir'] = os.path.join("./outputs/{}".format(trial_id))
    Path(params['output_dir']).mkdir(parents=True, exist_ok=True)
    
    logger = cl.utils.loggers.CometLogger(project_name='debug-2', workspace='cl-boundary', trial_name=trial_id)
    benchmark = cl.benchmarks.RotatedMNIST(num_tasks=params['num_tasks'],
                                           per_task_memory_examples=params['per_task_memory_examples'],
                                           per_task_joint_examples=params['per_task_joint_examples'])
    
    backbone = cl.backbones.MLP2Layers(input_dim=params['input_dim'],
                                       hidden_dim_1=params['hidden_1_dim'],
                                       hidden_dim_2=params['hidden_2_dim'],
                                       output_dim=params['output_dim'],
                                       dropout_prob=params.get("dropout_prob", 0.0),
                                       activation=params.get("activation", 'ReLU'),
                                       include_final_layer_act=False)
    
    algorithm = cl.algorithms.ContinualAlgorithm(backbone, benchmark, params)
    # algorithm = cl.algorithms.OGD(backbone, benchmark, params)
    
    metric_manager_callback = cl.callbacks.MetricManager(num_tasks=params['num_tasks'],
                                                         epochs_per_task=params['epochs_per_task'])
    model_checkpoint_callback = cl.callbacks.ModelCheckPoint()
    experiment_manager_callback = cl.callbacks.ExperimentManager()
    trainer = cl.trainer.ContinualTrainer(algorithm, params, logger=logger,
                                          callbacks=[metric_manager_callback,
                                                     model_checkpoint_callback,
                                                     experiment_manager_callback])
    
    trainer.run()
    # tune.report(average_loss=metric_collector_callback.get_final_metric())


if __name__ == "__main__":
    from params import toy_clf_params, toy_reg_params, rot_mnist_params
    sched = AsyncHyperBandScheduler(metric='average_loss', mode='min')
    analysis = tune.run(trial_toy_regression,
                        scheduler=sched, num_samples=2, config=toy_reg_params,
                        stop={'average_loss': 0.01})
    print("Best config is:", analysis.best_config)
    # trial_toy_regression(toy_reg_params)
    # trial_toy_classification(toy_clf_params)
    # trial_rot_mnist(rot_mnist_params)
