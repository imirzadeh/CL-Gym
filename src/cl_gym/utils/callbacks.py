from cl_gym.metrics import AverageAccuracy, AverageForgetting
import numpy as np
import torch
import os
from matplotlib import pyplot as plt
from pathlib import Path


class ContinualCallback:
    def __init__(self):
        pass
     
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


class AccuracyLogger(ContinualCallback):
    def __init__(self):
        self.avg_accs = []
        self.acc_meter = AverageAccuracy(num_tasks=8)
        self.forget_meter = AverageForgetting(num_tasks=8)
        super(AccuracyLogger, self).__init__()
    
    def on_before_training_task(self, trainer):
        print("-------- Training task {} ----------".format(trainer.current_task))
        
    def on_after_training_task(self, trainer):
        sum_accs = 0
        for task in range(1, trainer.current_task + 1):
            metrics = trainer.validate_algorithm_on_task(task)
            acc = metrics['accuracy']
            print(f"Validation on {task} => {metrics}")
            self.acc_meter.update(trainer.current_task, task, acc)
            self.forget_meter.update(trainer.current_task, task, acc)
            sum_accs += acc
        print(f"Average Accuracy => {sum_accs/trainer.current_task}")
        self.avg_accs.append(round(sum_accs/trainer.current_task, 2))
    
    def on_after_fit(self, trainer):
        print(f"Average Accuracy History => {self.avg_accs}")
        print(f"Avg acc score => {round(self.acc_meter.compute_final(), 3)}")
        print(f"Avg forget score => {round(self.forget_meter.compute_final(), 3)}")


class LossLogger(ContinualCallback):
    def __init__(self, num_tasks):
        self.num_tasks = num_tasks
        self.avg_losses = []
        self.loss_meter = AverageAccuracy(num_tasks=num_tasks)
        super(LossLogger, self).__init__()
    
    def on_before_training_task(self, trainer):
        print("-------- Training task {} ----------".format(trainer.current_task))
    
    def on_after_training_task(self, trainer):
        sum_losses = 0
        for task in range(1, trainer.current_task + 1):
            metrics = trainer.validate_algorithm_on_task(task)
            loss = metrics['loss']
            print(f"Validation on {task} => {metrics}")
            self.loss_meter.update(trainer.current_task, task, loss)
            sum_losses += loss
        print(f"Average Accuracy => {sum_losses / trainer.current_task}")
        self.avg_losses.append(round(sum_losses / trainer.current_task, 4))
    
    def on_after_fit(self, trainer):
        print(f"Average Loss History => {self.avg_losses}")
        print(f"Avg loss score => {round(self.loss_meter.compute_final(), 4)}")
    
    def get_final_metric(self):
        return self.loss_meter.compute_final()
    

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
        pred = net(test_x).detach().clone().numpy().reshape(128)
        plt.scatter(test_x.reshape(128), pred)
        plt.ylim(-1.2, 1.2)
        plt.yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        filename = "reg_task_{}.png".format(trainer.current_task)
        path = os.path.join(self.save_path, filename)
        plt.savefig(path, dpi=200)
        

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
        # checkpoint
        file_path = self.__get_save_path(trainer)
        # save
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
