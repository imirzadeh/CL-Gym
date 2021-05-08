import numpy as np
from typing import Optional


class ContinualMetric:
    def __init__(self,
                 num_tasks: int,
                 epochs_per_task: Optional[int] = 1,
                 validations_steps_per_epoch: Optional[int] = 1):
        self.num_tasks = num_tasks
        self.epochs_per_task = epochs_per_task
        self.validations_steps_per_epoch = validations_steps_per_epoch
        # data shape => [task_learned x task_evaluated x epoch_per task x validations_per_epoch]
        # check `update()` method to see how this array will be updated.
        self.data = np.zeros((num_tasks, num_tasks, epochs_per_task, validations_steps_per_epoch))
    
    def update(self,
               task_learned: int,
               task_evaluated: int,
               value: float,
               epoch: Optional[int] = 1,
               validation_step: Optional[int] = 1):
        
        if task_learned < 1 or task_evaluated < 1 or epoch < 1 or validation_step < 1:
            raise ValueError("Tasks, epochs, and validation steps are 1-based. i.e., the first task is task 1 not 0")
        
        self.data[task_learned-1][task_evaluated-1][epoch-1][validation_step-1] = value
    
    def compute(self, current_task: int):
        raise NotImplementedError
    
    def compute_final(self) -> float:
        raise NotImplementedError

