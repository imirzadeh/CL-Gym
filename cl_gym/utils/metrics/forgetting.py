import numpy as np
from typing import Optional
from cl_gym.utils.metrics import ContinualMetric


class ForgettingMetric(ContinualMetric):
    def __init__(self, num_tasks: int, epochs_per_task: Optional[int] = 1):
        super().__init__(num_tasks, epochs_per_task)
    
    def compute(self, current_task) -> float:
        # forgetting is defined for more than one task!
        if current_task == 1:
            return 0.0
        total_forgetting = 0
        for task in range(1, current_task):
            max_acc = np.max(self.data[1:, task, -1])
            last_acc = self.data[current_task][task][-1]
            task_forgetting = max_acc - last_acc
            total_forgetting += task_forgetting
        return total_forgetting/(current_task-1)

    def compute_final(self) -> float:
        return self.compute(self.num_tasks)

# if __name__ == "__main__":
#     avg_acc = AverageForgetting(3, 2)
#     # init
#     avg_acc.update(0, 1, 50, 1)
#     avg_acc.update(0, 1, 50, 2)
#     avg_acc.update(0, 2, 60, 1)
#     avg_acc.update(0, 2, 60, 2)
#
#     # task 1
#     avg_acc.update(1, 1, 90, 1)
#     avg_acc.update(1, 1, 100, 2)
#
#     # task 2
#     avg_acc.update(2, 1, 70, 1)
#     avg_acc.update(2, 1, 60, 2)
#     avg_acc.update(2, 2, 90, 1)
#     avg_acc.update(2, 2, 100, 2)
#
#     # task 3
#     avg_acc.update(3, 1, 50, 1)
#     avg_acc.update(3, 1, 50, 2)
#     avg_acc.update(3, 2, 70, 1)
#     avg_acc.update(3, 2, 60, 2)
#     avg_acc.update(3, 3, 95, 1)
#     avg_acc.update(3, 3, 100, 2)
#
#     print(avg_acc.data)
#     print(avg_acc.data.shape)
#     # print(avg_acc.compute(1))
#     print(avg_acc.compute(2))
#     # print(avg_acc.compute(3))
#     print(avg_acc.compute_final())

