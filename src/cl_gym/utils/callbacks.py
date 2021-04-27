from cl_gym.metrics import AverageMetric, AverageForgetting
import numpy as np
import torch
import os
from matplotlib import pyplot as plt
from pathlib import Path


class ContinualCallback:
    def __init__(self, name=''):
        self.name = name
    
    def log_text(self, trainer, text):
        if trainer.logger:
            trainer.logger.log_text(text)
        print(text)
        
    def log_metric(self, trainer, metric_name, metric_value, metric_step=None):
        if trainer.logger:
            trainer.logger.log_metric(metric_name, metric_value, metric_step)
    
    def on_before_fit(self, trainer):
        pass
    
    def on_after_fit(self, trainer):
        pass
    
    def on_before_training_task(self, trainer):
        pass
    
    def on_after_training_task(self, trainer):
        pass
    
    def on_before_training_epoch(self, trainer):
        pass

    def on_after_training_epoch(self, trainer):
        pass
    
    def on_before_training_step(self, trainer):
        pass
    
    def on_after_training_step(self, trainer):
        pass


class MetricManager(ContinualCallback):
    def __init__(self, num_tasks, epochs_per_task=1, intervals='tasks'):
        super(MetricManager, self).__init__('MetricManager')
        
        # checks
        if intervals.lower() not in ['tasks', 'epochs']:
            raise ValueError("MetricManager supports only `tasks` and `epochs` for intervals.")
        self.intervals = intervals
        if self.intervals == 'tasks':
            epochs_per_task = 1
        
        self.metric = AverageMetric(num_tasks, epochs_per_task)
        self.forget_metric = AverageForgetting(num_tasks)
        self.eval_type = None
        self.save_path = None
    
    def on_before_fit(self, trainer):
        self.save_path = os.path.join(trainer.params['output_dir'], 'metrics')
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        
        benchmark_name = str(trainer.algorithm.benchmark).lower()
        if 'reg' in benchmark_name:
            self.eval_type = 'regression'
        else:
            self.eval_type = 'classification'
        
    def on_before_training_task(self, trainer):
        text = f"------------ Training task {trainer.current_task} -------------"
        self.log_text(trainer, text)
     
    def _calculate_eval_epoch(self, trainer):
        if self.intervals == 'epochs':
            task_epoch = trainer.current_epoch % trainer.params['epochs_per_task']
            if task_epoch == 0:
                task_epoch = trainer.params['epochs_per_task']
            return task_epoch
        else:
            return 1

    def _collect_metrics(self, trainer):
        eval_epoch = self._calculate_eval_epoch(trainer)
        step = trainer.current_task if self.intervals == 'tasks' else trainer.current_epoch
        for task in range(1, trainer.current_task + 1):
            eval_metrics = trainer.validate_algorithm_on_task(task)
            if self.eval_type == 'classification':
                acc = eval_metrics['accuracy']
                loss = eval_metrics['loss']
                # update meters
                
                self.metric.update(trainer.current_task, task, acc, eval_epoch)
                self.forget_metric.update(trainer.current_task, task, acc)
                # logging
                self.log_text(trainer, f"eval metrics for task {task}: acc={round(acc, 2)}, loss={round(loss, 5)}")
                self.log_metric(trainer, f'acc_{task}', round(acc, 2), step)
                self.log_metric(trainer, f'loss_{task}', round(loss, 5), step)
                if task == trainer.current_task:
                    self.log_metric(trainer, f'average_acc', round(self.metric.compute(trainer.current_task)), step)
            else:
                loss = eval_metrics['loss']
                self.metric.update(trainer.current_task, task, loss, eval_epoch)
                self.log_text(trainer, f"eval metrics for task {task}: loss={round(loss, 5)}")
                self.log_metric(trainer, f'loss_{task}', round(loss, 5), step)
                if task == trainer.current_task:
                    self.log_metric(trainer, f'average_loss', round(self.metric.compute(trainer.current_task), 5), step)

    def on_after_training_task(self, trainer):
        if self.intervals != 'tasks':
            return
        self._collect_metrics(trainer)
        
    def on_after_training_epoch(self, trainer):
        if self.intervals != 'epochs':
            return
        step = self._calculate_eval_epoch(trainer)
        text = f"epoch {step}/{trainer.params['epochs_per_task']} >>"
        self.log_text(trainer, text)
        self._collect_metrics(trainer)
    
    def on_after_fit(self, trainer):
        filepath = os.path.join(self.save_path, "metrics.npy")
        with open(filepath, 'wb') as f:
            np.save(f, self.metric.data)


class ToyRegressionVisualizer(ContinualCallback):
    def __init__(self):
        self.map_functions = [lambda x: (x + 3.),
                 lambda x: 2. * np.power(x, 2) - 1,
                 lambda x: np.power(x - 3., 3)]
        self.domains = [[-4, -2], [-1, 1], [2, 4]]
        self.x_min = -4.5
        self.x_max = 4.5
        self.save_path = None

        super(ToyRegressionVisualizer, self).__init__()
    
    def on_before_fit(self, trainer):
        self.save_path = os.path.join(trainer.params['output_dir'], 'plots')
        Path(self.save_path).mkdir(parents=True, exist_ok=True)

    def on_after_training_task(self, trainer):
        net = trainer.algorithm.backbone
        net.eval()
        test_x = torch.from_numpy(np.linspace(self.x_min, self.x_max, 128).reshape((128, 1))).float()
        test_x = test_x.to(trainer.params['device'])
        pred = net(test_x).to('cpu').detach().clone().numpy().reshape(128)
        plt.scatter(test_x.reshape(128), pred)
        plt.ylim(-1.2, 1.2)
        plt.yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        filename = f"reg_task_{trainer.current_task}"
        path = f"{os.path.join(self.save_path, filename)}.pdf"
        if trainer.logger:
            trainer.logger.log_figure(plt, filename)
        plt.savefig(path, dpi=220)
        

class ExperimentManager(ContinualCallback):
    def __init__(self):
        super(ExperimentManager, self).__init__()
    
    def on_before_fit(self, trainer):
        if trainer.logger:
            trainer.logger.log_parameters(trainer.params)
    
    def on_after_fit(self, trainer):
        if trainer.logger:
            path = trainer.params['output_dir']
            trainer.logger.log_folder(folder_path=path)
            trainer.logger.terminate()


class ModelCheckPoint(ContinualCallback):
    def __init__(self, interval='task', name_prefix=None):
        if interval.lower() not in ['task', 'epoch']:
            raise ValueError("Checkpoint callback supports can only save after each 'task' or each 'epoch'")
        self.interval = interval
        self.name_prefix = name_prefix
        super(ModelCheckPoint, self).__init__()
    
    def __get_save_path(self, trainer):
        checkpoint_folder = os.path.join(trainer.params['output_dir'], 'checkpoints')
        Path(checkpoint_folder).mkdir(parents=True, exist_ok=True)
        model_name = self.name_prefix if self.name_prefix else ''
        if trainer.current_step <= 1:
            model_name += 'init.pt'
        else:
            model_name += f"{self.interval}_"
            if self.interval == 'task':
                model_name += f"{trainer.current_task}.pt"
            else:
                model_name += f"{trainer.current_epoch}.pt"
            
        filepath = os.path.join(checkpoint_folder, model_name)
        return filepath

    def __save_model(self, trainer):
        # get the model (backbone)
        model = trainer.algorithm.backbone
        model.eval()
        file_path = self.__get_save_path(trainer)
        torch.save(model.to('cpu').state_dict(), file_path)
    
    def on_before_fit(self, trainer):
        # save init params
        self.__save_model(trainer)
    
    def on_after_training_task(self, trainer):
        if self.interval == 'task':
            self.__save_model(trainer)
        
    def on_after_training_epoch(self, trainer):
        if self.interval == 'epoch':
            self.__save_model(trainer)
