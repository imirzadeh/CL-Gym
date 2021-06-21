import os
import numpy as np
from typing import Optional, Callable, Dict
from cl_gym.utils.loggers import Logger
from cl_gym.utils.callbacks import ContinualCallback
from cl_gym.utils.callbacks.helpers import IntervalCalculator, Visualizer
from cl_gym.utils.metrics import ContinualMetric, PerformanceMetric, ForgettingMetric


class MetricCollector(ContinualCallback):
    """
    Collects metrics during the learning.
    This callback can support various metrics such as average accuracy/error, and average forgetting.
    """
    def __init__(self, num_tasks: int,
                 epochs_per_task: Optional[int] = 1,
                 collect_on_init: bool = False,
                 collect_metrics_for_future_tasks: bool = False,
                 eval_interval: str = 'epoch',
                 eval_type: str = 'classification',
                 tuner_callback: Optional[Callable[[float, bool], None]] = None):
        """
        
        Args:
            num_tasks: The number of task for the learning experience.
            epochs_per_task: The number of epochs per task.
            collect_on_init: Should also collect metrics before training starts?
            collect_metrics_for_future_tasks: Should collect metrics for future tasks? (e.g.,, for forward-transfer)
            eval_interval: The intervals at which the algorithm will be evaluated. Can be either `task` or `epoch`
            eval_type: Is this a `classification` task or `regression` task?
            tuner_callback: Optional tuner callback than can be called with eval metrics for parameter optimization.
        """
        self.num_tasks = num_tasks
        self.epochs_per_task = epochs_per_task
        self.collect_on_init = collect_on_init
        self.collect_metrics_for_future_tasks = collect_metrics_for_future_tasks
        self.eval_interval = eval_interval.lower()
        self.tuner_callback = tuner_callback
        self.save_dirs = ['plots', 'metrics']
        self.eval_type = eval_type.lower()
        self.meters = self._prepare_meters()
        self.interval_calculator = IntervalCalculator(self.num_tasks, self.epochs_per_task, self.eval_interval)
        super(MetricCollector, self).__init__("MetricCollector", self.save_dirs)
        self._verify_inputs()

    def _prepare_meters(self) -> Dict[str, ContinualMetric]:
        if self.eval_type == 'classification':
            return {'accuracy': PerformanceMetric(self.num_tasks, self.epochs_per_task),
                    'forgetting': ForgettingMetric(self.num_tasks, self.epochs_per_task),
                    'loss': PerformanceMetric(self.num_tasks, self.epochs_per_task)}
        else:
            return {'loss': PerformanceMetric(self.num_tasks, self.epochs_per_task)}
        
    def _verify_inputs(self):
        if self.num_tasks <= 0 or self.epochs_per_task <= 0:
            raise ValueError("'num_tasks' and 'epochs_per_task' should be greater than 0")
        if self.eval_interval not in ['epoch', 'task']:
            raise ValueError("'eval_interval' should be either 'task' or 'epoch'")
        if self.eval_type not in ['classification', 'regression']:
            raise ValueError("'eval_type' for metrics should be either 'classification' or 'regression'")

    def _update_meters(self, task_learned: int, task_evaluated: int, metrics: dict, relative_step: int):
        if self.eval_type == 'classification':
            self.meters['loss'].update(task_learned, task_evaluated, metrics['loss'], relative_step)
            self.meters['accuracy'].update(task_learned, task_evaluated, metrics['accuracy'], relative_step)
            self.meters['forgetting'].update(task_learned, task_evaluated, metrics['accuracy'], relative_step)
        else:
            self.meters['loss'].update(task_learned, task_evaluated, metrics['loss'], relative_step)
    
    def _update_logger(self, trainer, task_evaluated: int, metrics: dict, global_step: int):
        if trainer.logger is None:
            return
        
        if self.eval_type == 'classification':
            trainer.logger.log_metric(f'acc_{task_evaluated}', round(metrics['accuracy'], 2), global_step)
            trainer.logger.log_metric(f'loss_{task_evaluated}', round(metrics['loss'], 2), global_step)
            if trainer.current_task > 0:
                avg_acc = round(self.meters['accuracy'].compute(trainer.current_task), 2)
                trainer.logger.log_metric(f'average_acc', avg_acc, global_step)
        else:
            trainer.logger.log_metric(f'loss_{task_evaluated}', round(metrics['loss'], 2), global_step)
            if trainer.current_task > 0:
                avg_loss = round(self.meters['loss'].compute(trainer.current_task), 5)
                trainer.logger.log_metric(f'average_loss', avg_loss, global_step)

    def _update_tuner(self, is_final_score: bool):
        if self.tuner_callback is None:
            return
        if self.eval_type == 'classification':
            score = self.meters['accuracy'].compute_final()
        else:
            score = self.meters['loss'].compute_final()
        self.tuner_callback(score, is_final_score)
    
    def log_metrics(self, trainer, task_learned: int, task_evaluated: int,
                     metrics: dict, global_step: int, relative_step: int):
        self._update_meters(task_learned, task_evaluated, metrics, relative_step)
        self._update_logger(trainer, task_evaluated, metrics, global_step)
        self._update_tuner(is_final_score=False)
    
    def _collect_eval_metrics(self, trainer, start_task: int, end_task: int):
        global_step = trainer.current_task if self.eval_interval == 'task' else trainer.current_epoch
        relative_step = self.interval_calculator.get_step_within_task(trainer.current_epoch)
        for eval_task in range(start_task, end_task + 1):
            task_metrics = trainer.validate_algorithm_on_task(eval_task)
            self.log_metrics(trainer, trainer.current_task, eval_task, task_metrics, global_step, relative_step)
            print(f"[{global_step}] Eval metrics for task {eval_task} >> {task_metrics}")

    def save_metrics(self):
        metrics = ['accuracy', 'loss'] if self.eval_type == 'classification' else ['loss']
        for metric in metrics:
            filepath = os.path.join(self.save_paths['metrics'], metric + ".npy")
            with open(filepath, 'wb') as f:
                np.save(f, self.meters[metric].data)
    
    def _prepare_plot_params(self):
        xticks = self.interval_calculator.get_tick_times()
        if self.collect_on_init:
            xticks = [0] + xticks
        return {
            'show_legend': True,
            'legend_loc': 'lower left',
            'xticks': xticks,
            'xlabel': 'Epochs',
            'ylabel': 'Validation Accuracy'
        }
    
    def _extract_task_history(self, task):
        if self.collect_on_init:
            start, end = 0, self.interval_calculator.get_task_range(task)[1]
            offset = 0 if self.eval_interval == 'task' else self.epochs_per_task - 1
            metrics = self.meters['accuracy'].get_raw_history(task, 0)[start+offset:end+offset]
            # print(metrics)
        elif self.collect_metrics_for_future_tasks:
            start, end = 1, self.interval_calculator.get_task_range(task)[1]
            metrics = self.meters['accuracy'].get_raw_history(task, 1)
        else:
            start, end = self.interval_calculator.get_task_range(task)
            metrics = self.meters['accuracy'].get_raw_history(task)[start-1:]
            if self.eval_interval == 'task':
                metrics = self.meters['accuracy'].get_raw_history(task)
                start = self.epochs_per_task-1
                metrics = metrics[start::self.epochs_per_task]
                print(f"DEBUG >> task {task}, metric shape= {metrics.shape}")
        return range(start, end), metrics
        
    def plot_metrics(self, logger: Optional[Logger] = None):
        if self.eval_type != 'classification':
            return
        
        plot_params = self._prepare_plot_params()
        data, labels = [], []
        for task in range(1, self.num_tasks + 1):
            data.append(self._extract_task_history(task))
            labels.append(f"Task{task}")
        Visualizer.line_plot(data, labels=labels, save_dir=self.save_paths['plots'], filename="metrics",
                             cmap='g10', plot_params=plot_params, logger=logger)
    
    def on_before_fit(self, trainer):
        if self.collect_on_init:
            print(f"---------------------------- Init -----------------------")
            self._collect_eval_metrics(trainer, 1, end_task=self.num_tasks)
    
    def on_after_training_epoch(self, trainer):
        if self.eval_interval != 'epoch':
            return
        if self.collect_on_init or self.collect_metrics_for_future_tasks:
            self._collect_eval_metrics(trainer, 1, self.num_tasks)
        else:
            self._collect_eval_metrics(trainer, 1, trainer.current_task)
    
    def on_before_training_task(self, trainer):
        print(f"---------------------------- Task {trainer.current_task+1} -----------------------")
        
    def on_after_training_task(self, trainer):
        if self.eval_interval != 'task':
            return
        if self.collect_on_init or self.collect_metrics_for_future_tasks:
            self._collect_eval_metrics(trainer, 1, self.num_tasks)
        else:
            self._collect_eval_metrics(trainer, 1, trainer.current_task)

    def on_after_fit(self, trainer):
        self._update_tuner(is_final_score=True)
        self.save_metrics()
        self.plot_metrics(trainer.logger)