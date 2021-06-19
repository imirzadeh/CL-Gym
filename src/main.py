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


def run_main(params):
    trial_id = str(uuid.uuid4())
    params['trial_id'] = trial_id
    params['output_dir'] = os.path.join("./outputs/{}".format(trial_id))
    Path(params['output_dir']).mkdir(parents=True, exist_ok=True)
    
    logger = cl.utils.loggers.CometLogger(project_name='clgym-debug', workspace='cl-boundary', trial_name=trial_id)
    benchmark = cl.benchmarks.SplitCIFAR100(num_tasks=params['num_tasks'],
                                           per_task_memory_examples=params['per_task_memory_examples'],
                                           per_task_joint_examples=params['per_task_joint_examples'])
    
    backbone = cl.backbones.ResNet18Small(100)
    
    algorithm = cl.algorithms.ContinualAlgorithm(backbone, benchmark, params)
    # algorithm = cl.algorithms.OGD(backbone, benchmark, params)
    # algorithm = cl.algorithms.ORM(backbone, benchmark, params)
    
    metric_manager_callback = cl.callbacks.MetricCollector(num_tasks=params['num_tasks'],
                                                         epochs_per_task=params['epochs_per_task'])
    
    model_checkpoint_callback = cl.callbacks.ModelCheckpoint()
    experiment_manager_callback = cl.callbacks.ExperimentManager()
    trainer = cl.trainer.ContinualTrainer(algorithm, params, logger=logger,
                                          callbacks=[metric_manager_callback,
                                                     model_checkpoint_callback,
                                                     experiment_manager_callback])
    
    trainer.run()
    # tune.report(average_loss=metric_collector_callback.get_final_metric())


if __name__ == "__main__":
    from params import toy_clf_params, toy_reg_params, rot_mnist_params, cifar_params
    run_main(cifar_params)

