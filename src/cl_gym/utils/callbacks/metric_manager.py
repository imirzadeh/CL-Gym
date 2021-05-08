import os
import numpy as np
from ray import tune
from pathlib import Path
from cl_gym.utils.callbacks import ContinualCallback
from cl_gym.utils.metrics import AverageMetric, AverageForgetting


class MetricManager(ContinualCallback):
    def __init__(self, num_tasks, epochs_per_task=1, intervals='tasks', tuner=True):
        super(MetricManager, self).__init__('MetricManager')
        self.tuner = tuner
        
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
                    avg_acc = round(self.metric.compute(trainer.current_task), 2)
                    self.log_metric(trainer, f'average_acc', avg_acc, step)
                    if self.tuner:
                        tune.report(average_loss=avg_acc)
            else:
                loss = eval_metrics['loss']
                self.metric.update(trainer.current_task, task, loss, eval_epoch)
                self.log_text(trainer, f"eval metrics for task {task}: loss={round(loss, 5)}")
                self.log_metric(trainer, f'loss_{task}', round(loss, 5), step)
                if task == trainer.current_task:
                    avg_loss = round(self.metric.compute(trainer.current_task), 5)
                    self.log_metric(trainer, f'average_loss', avg_loss, step)
                    if self.tuner:
                        tune.report(average_loss=avg_loss)
    
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
