import numpy as np
from typing import Optional


class ContinualMetric:
    def __init__(self, num_tasks: int, epochs_per_task: Optional[int] = 1):
        self.num_tasks = num_tasks
        self.epochs_per_task = epochs_per_task
        # data shape => [(task_learned+1) x (task_evaluated+1) x epoch_per task]
        # The 0 index is reserved for 'initialization' metrics
        self.data = np.zeros((num_tasks+1, num_tasks+1, epochs_per_task))
    
    def update(self, task_learned: int, task_evaluated: int, value: float, epoch: Optional[int] = 1):
        if epoch < 1:
            raise ValueError("Epoch number is 1-based")
        self.data[task_learned][task_evaluated][epoch-1] = value
    
    def get_raw_history(self, task: int, start_task: Optional[int] = 1,
                        ravel: Optional[bool] = True) -> np.array:
        history = self.data[start_task:, task, :]
        if ravel:
            return np.ravel(history)
        return history
    
    def compute(self, task: int) -> float:
        raise NotImplementedError
    
    def compute_final(self) -> float:
        raise NotImplementedError

