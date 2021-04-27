import numpy as np
from typing import Optional
from cl_gym.metrics import ContinualMetric


class AverageAccuracy(ContinualMetric):
    def __init__(self,
                 num_tasks: int,
                 epochs_per_task: Optional[int] = 1,
                 validations_steps_per_epoch: Optional[int] = 1):
        super().__init__(num_tasks, epochs_per_task, validations_steps_per_epoch)
    
    def compute(self, current_task: int):
        if current_task < 1:
            raise ValueError("Tasks are 1-based. i.e., the first task's id is 1, not 0.")
        
        return np.mean(self.data[current_task-1, :current_task, -1, -1])
        
    def compute_final(self):
        return np.mean(self.data[-1, :, -1, -1])
