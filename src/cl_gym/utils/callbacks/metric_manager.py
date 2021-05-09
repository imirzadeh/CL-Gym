import os
import matplotlib
import numpy as np
from ray import tune
import seaborn as sns
from pathlib import Path
from matplotlib import rc
from matplotlib import pyplot as plt
from cl_gym.utils.callbacks import ContinualCallback
from cl_gym.utils.metrics import AverageMetric, AverageForgetting
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
np.set_printoptions(precision=4, suppress=True)


class MetricManager(ContinualCallback):
    def __init__(self, num_tasks, epochs_per_task=1, intervals='tasks', tuner=True):
        super(MetricManager, self).__init__('MetricManager')
        self.num_tasks = num_tasks
        self.epochs_per_task = epochs_per_task
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
        # step = self._calculate_eval_epoch(trainer)
        # text = f"epoch {step}/{trainer.params['epochs_per_task']} >>"
        # self.log_text(trainer, text)
        self._collect_metrics(trainer)
    
    def get_plot_colors(self):
        if self.num_tasks <= 10:
            colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
                      '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
        else:
            colors = ['#00429d', '#26559c', '#3a679c', '#4c7a9b', '#5e8c9b',
                      '#719e9a', '#86af9a', '#9dc09a', '#b6d19b', '#d2e19b',
                      '#ffe488', '#ffc981', '#ffad78', '#fa9270', '#f17867',
                      '#e55e5f', '#d64456', '#c42a4d', '#ae1044', '#93003a']
        return colors
    
    def get_plot_xticks(self):
        if self.intervals == 'tasks':
            return range(1, self.num_tasks+1)
        else:
            return list(range(1, (self.num_tasks+1)*self.epochs_per_task, self.epochs_per_task))
    
    def plot_metrics(self, trainer):
        plt.close('all')
        rc('text', usetex=True)
        sns.set_context("paper", rc={"lines.linewidth": 3.5,
                                     'xtick.labelsize': 20,
                                     'ytick.labelsize': 20,
                                     'lines.markersize': 8,
                                     'legend.fontsize': 17,
                                     'axes.labelsize': 20,
                                     'legend.handlelength': 1,
                                     'legend.handleheight': 1, })
        save_path = os.path.join(trainer.params['output_dir'], 'plots')
        colors = self.get_plot_colors()
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        for task in range(1, self.num_tasks+1):
            metrics = self.metric.get_raw_history(task)
            plt.plot(range(1, len(metrics)+1), metrics, color=colors[task-1], label=f"Task{task}")

        ylabel = 'Validation Accuracy' if self.eval_type == 'classification' else 'Validation Loss'
        xticks = self.get_plot_xticks()
        if self.eval_type == 'classification':
            plt.ylim((0.1, None))
        plt.xlabel("Epochs" if self.intervals == 'epochs' else "Tasks Learned")
        plt.ylabel(ylabel)
        plt.xticks(xticks)
        plt.legend(loc='lower left')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "metrics.pdf"), dpi=200)
        if trainer.logger:
            trainer.logger.log_figure(plt, 'metrics')
        plt.close()

    def on_after_fit(self, trainer):
        filepath = os.path.join(self.save_path, "metrics.npy")
        with open(filepath, 'wb') as f:
            np.save(f, self.metric.data)
        
        self.plot_metrics(trainer)
