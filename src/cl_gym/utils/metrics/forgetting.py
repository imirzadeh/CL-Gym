import numpy as np
from cl_gym.metrics import ContinualMetric


class AverageForgetting(ContinualMetric):
    def __init__(self, num_tasks: int):
        super().__init__(num_tasks)
    
    def compute(self, current_task):
        # forgetting is defined for more than one task!
        if current_task == 1:
            return 0.0
        total_forgetting = 0
        for task in range(1, current_task):
            max_acc = np.max(self.data[:, task-1, -1, -1])
            last_acc = self.data[current_task-1][task-1][-1][-1]
            task_forgetting = max_acc - last_acc
            total_forgetting += task_forgetting
        return total_forgetting/(current_task-1)

    def compute_final(self):
        return self.compute(self.num_tasks)
        # total_forgetting = 0
        # for task in range(1, self.num_tasks):
        #     max_acc = np.max(self.data[:, task-1, -1, -1])
        #     last_acc = self.data[-1][task-1][-1][-1]
        #     task_forgetting = max_acc - last_acc
        #     total_forgetting += task_forgetting
        # return total_forgetting/(self.num_tasks-1)
        
    

