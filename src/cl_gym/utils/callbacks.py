from cl_gym.metrics import AverageMetric, AverageForgetting
import numpy as np
import torch
import os
from matplotlib import pyplot as plt
from pathlib import Path
from matplotlib.colors import ListedColormap
from ray import tune


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


class ToyRegressionVisualizer(ContinualCallback):
    def __init__(self):
        self.map_functions = [lambda x: (x + 3.),
                 lambda x: 2. * np.power(x, 2) - 1,
                 lambda x: np.power(x - 3., 3)]
        self.domains = [[-4, -2], [-1, 1], [2, 4]]
        self.colors = ['#36008D', '#FE5E54', '#00C9B8']
        self.x_min = -4.5
        self.x_max = 4.5
        self.save_path = None

        super(ToyRegressionVisualizer, self).__init__()
    
    def on_before_fit(self, trainer):
        self.save_path = os.path.join(trainer.params['output_dir'], 'plots')
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
    
    def plot_task(self, trainer, task=1):
        net = trainer.algorithm.backbone
        net.eval()
        # data
        num_examples = 12
        x_min, x_max = self.domains[task-1]
        color = self.colors[task-1]
        data = np.linspace(x_min, x_max, num_examples).reshape((num_examples, 1))
        test_x = torch.from_numpy(data).float()
        test_x = test_x.to(trainer.params['device'])
        test_y = np.vectorize(self.map_functions[task-1])(test_x.cpu().numpy()).reshape(num_examples, 1)

        pred = net(test_x).to('cpu').detach().clone().numpy().reshape(num_examples)
        plt.plot(data.reshape(num_examples), test_y.reshape(num_examples),
                 color=color, alpha=0.6, linewidth=3)
        plt.plot(test_x.cpu().numpy().reshape(num_examples), pred,
                    color=color, linewidth=3, linestyle='--')
        plt.ylim(-2.5, 2.5)
        plt.yticks([-2, -1, 0, 1, 2])
        plt.xticks([-4, -3, -2, -1, 0, 1, 2, 3, 4])
        plt.xlim(-4.5, 4.5)

    def on_after_training_task(self, trainer):
        for task in range(1, trainer.current_task+1):
            self.plot_task(trainer, task)
        filename = f"reg_task_{trainer.current_task}"
        path = f"{os.path.join(self.save_path, filename)}.pdf"
        if trainer.logger:
            trainer.logger.log_figure(plt, filename)
        plt.savefig(path, dpi=220)
        plt.close('all')
        

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


class ToyClassificationVisualizer(ContinualCallback):
    def __init__(self):
        super(ToyClassificationVisualizer, self).__init__('visualizer')
    
    def on_before_fit(self, trainer):
        self.save_path = os.path.join(trainer.params['output_dir'], 'plots')
        Path(self.save_path).mkdir(parents=True, exist_ok=True)

    def extract_points(self, loader):
        xs, ys = [], []
        for inp, targ, task_ids in loader:
            batch_size = len(inp)
            for batch in range(batch_size):
                xs.append(inp[batch].numpy())
                ys.append(targ[batch])
        return np.array(xs), np.array(ys)

    def _plot_decision_boundary(self, trainer):
        net = trainer.algorithm.backbone.to('cpu')
        net.eval()

        h = .05
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        
        _, loader = trainer.algorithm.benchmark.load_joint(trainer.current_task, batch_size=64)
        xx, yy = np.meshgrid(np.arange(-4, 4, h), np.arange(-4, 4, h))
        X, y = self.extract_points(loader)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright, edgecolors='k')
        Z = net(torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()).data.max(1, keepdim=True)[1].detach().numpy()
        # print(Z)
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=cm, alpha=.7)
        plt.text(-3, 2, f"Task {trainer.current_task} - Epoch {trainer.current_epoch}", fontsize=20)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, f'decisions-{trainer.current_epoch}.png'), dpi=200)
        
        # trainer.logger.log_figure(plt, 'decisions', step=trainer.current_epoch)
        plt.close('all')
        
    def on_after_training_epoch(self, trainer):
        self._plot_decision_boundary(trainer)
