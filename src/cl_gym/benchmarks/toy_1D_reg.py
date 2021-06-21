import torch
import numpy as np
from typing import Optional
from torch.utils.data import Dataset
from cl_gym.benchmarks import Benchmark

DEFAULT_TOY_DATASET_SIZE = 100
DEFAULT_STD = 0.05


class Toy1DRegDataset(Dataset):
    """
    Toy task: 1D Regression problem with 3 tasks
    This task was introduced by Oswald & Henning et. al.
    Please see: https://openreview.net/pdf?id=SJgwNerKvB
    """
    def __init__(self, task_id, num_examples: int, noise_std: float = 0.05):
        self.task_id = task_id
        self.num_examples = num_examples
        self.noise_std = noise_std
        self.data = []
        self.targets = []
        self.__generate_data()
    
    def __generate_data(self):
        map_functions = [lambda x: (x + 3.),
                     lambda x: 2. * np.power(x, 2) - 1,
                     lambda x: np.power(x - 3., 3)]
        x_domains = [[-4, -2], [-1, 1], [2, 4]]
        
        start, end = x_domains[self.task_id-1]
        self.data = np.random.uniform(low=start, high=end, size=(self.num_examples, 1))
        self.targets = map_functions[self.task_id-1](self.data)
        if self.noise_std > 0.0:
            noise = np.random.normal(loc=0.0, scale=self.noise_std, size=(self.num_examples, 1))
            self.targets += noise
        
        assert len(self.data) == len(self.targets)
        self.data = torch.from_numpy(self.data).float()
        self.targets = torch.from_numpy(self.targets).float()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        return self.data[index], self.targets[index], self.task_id


class Toy1DRegression(Benchmark):
    """
    Toy benchmark: 1D Regression problem with 3 tasks
    Task `t` will be a polynomial with degree `t`
    Please see: https://openreview.net/pdf?id=SJgwNerKvB
    """
    def __init__(self,
                 num_tasks: int = 3,
                 per_task_examples: Optional[int] = None,
                 per_task_joint_examples: Optional[int] = 0,
                 per_task_memory_examples: Optional[int] = 0,
                 per_task_subset_examples: Optional[int] = 0,
                 task_size: Optional[int] = DEFAULT_TOY_DATASET_SIZE,
                 noise_std: Optional[float] = DEFAULT_STD):
        super().__init__(num_tasks, per_task_examples, per_task_joint_examples, per_task_memory_examples,
                         per_task_subset_examples)
        self.task_size = task_size
        self.noise_std = noise_std
        self.load_datasets()
        self.prepare_datasets()
    
    def load_datasets(self):
        for task in range(1, self.num_tasks + 1):
            self.trains[task] = Toy1DRegDataset(task, self.task_size, self.noise_std)
            self.tests[task] = Toy1DRegDataset(task, self.task_size, self.noise_std)
    
    def precompute_memory_indices(self):
        for task in range(1, self.num_tasks + 1):
            self.memory_indices_train[task] = np.random.randint(0, len(self.trains[task]), size=self.per_task_memory_examples)
            self.memory_indices_test[task] = np.random.randint(0, len(self.tests[task]), size=self.per_task_memory_examples)
