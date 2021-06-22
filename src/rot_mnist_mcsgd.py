import torch
import cl_gym as cl


def make_params():
    import os
    from pathlib import Path
    import uuid

    params = {
            # benchmark
            'num_tasks': 20,
            'epochs_per_task': 1,
            'per_task_memory_examples': 10,
            'per_task_subset_examples': 500,
            'batch_size_train': 64,
            'batch_size_memory': 64,
            'batch_size_validation': 256,
            'num_dataloader_workers': os.cpu_count()//2,

            # algorithm
            'mcsgd_alpha': 0.1,
            'mcsgd_line_samples': 10,
            'mcsgd_line_optim_lr': 0.05,
            'dropout': 0.2,
            'optimizer': 'SGD',
            'learning_rate': 0.1,
            'learning_rate_lower_bound': 0.00001,
            'momentum': 0.8,
            'learning_rate_decay': 0.8,
            'criterion': torch.nn.CrossEntropyLoss(),
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'), }

    trial_id = str(uuid.uuid4())
    params['trial_id'] = trial_id
    params['output_dir'] = os.path.join("./outputs/{}".format(trial_id))
    Path(params['output_dir']).mkdir(parents=True, exist_ok=True)

    return params


def train(params):
    # benchmark: Rotated MNIST
    benchmark = cl.benchmarks.RotatedMNIST(num_tasks=params['num_tasks'],
                                           per_task_memory_examples=params['per_task_memory_examples'],
                                           per_task_subset_examples=params['per_task_memory_examples'],
                                           per_task_rotation=9.0)

    # backbone: MLP with 2 hidden layers
    backbone = cl.backbones.MLP2Layers(hidden_dim_1=256, hidden_dim_2=256, dropout_prob=params['dropout'])

    # Algorithm: A-GEM
    algorithm = cl.algorithms.MCSGD(backbone, benchmark, params)

    # Callbacks
    metric_manager_callback = cl.callbacks.MetricCollector(num_tasks=params['num_tasks'],
                                                           eval_interval='epoch',
                                                           epochs_per_task=params['epochs_per_task'])

    # Make trainer
    trainer = cl.trainer.ContinualTrainer(algorithm, params, callbacks=[metric_manager_callback])

    trainer.run()


if __name__ == "__main__":
    params = make_params()
    train(params)


