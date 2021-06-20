import torch
from typing import Optional
from sklearn.datasets import make_blobs
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from cl_gym.benchmarks import Benchmark

DEFAULT_TOY_DATASET_SIZE = 100
DEFAULT_CLUSTER_STD = 0.3
SMALL_COORD = 0.5
LARGE_COORD = 2.0


class Toy2DCLFDataset(Dataset):
    def __init__(self, num_tasks: int, task_id: int, samples_per_task: int, cluster_std: float):
        self.data = []
        self.targets = []
        self.num_tasks = num_tasks
        self.task_id = task_id
        self.cluster_std = cluster_std
        self.samples_per_task = samples_per_task
        self.__generate_data()
    
    def __get_centers(self):
        if self.num_tasks == 2:
            task_centers = {1: ((SMALL_COORD, SMALL_COORD), (LARGE_COORD, LARGE_COORD)),
                            2: ((-SMALL_COORD, -SMALL_COORD), (-LARGE_COORD, -LARGE_COORD))}
        elif self.num_tasks == 4:
            task_centers = {1: ((SMALL_COORD, SMALL_COORD), (LARGE_COORD, LARGE_COORD)),
                            2: ((SMALL_COORD, -SMALL_COORD), (LARGE_COORD, -LARGE_COORD)),
                            3: ((-SMALL_COORD, -SMALL_COORD), (-LARGE_COORD, -LARGE_COORD)),
                            4: ((-SMALL_COORD, SMALL_COORD), (-LARGE_COORD, LARGE_COORD))}
        else:
            raise ValueError("2D Toy Classification dataset can have either 2 or 4 tasks")
        return task_centers
        
    def __generate_data(self):
        centers = self.__get_centers()[self.task_id]
        inp, targ = make_blobs(n_samples=self.samples_per_task, n_features=2, centers=centers, cluster_std=self.cluster_std)
        self.data = torch.from_numpy(inp).float()
        self.targets = torch.from_numpy(targ)
        
    def __getitem__(self, index):
        return self.data[index], int(self.targets[index]), self.task_id
    
    def __len__(self):
        return len(self.data)


class Toy2DClassification(Benchmark):
    """
    Toy benchmark: each task will be a binary classification with linearly separable classes in 2D space.
    Essentially, each task is a Gaussian cluster at some coordinates.
    """
    def __init__(self,
                 num_tasks: int,
                 per_task_examples: Optional[int] = None,
                 per_task_joint_examples: Optional[int] = 0,
                 per_task_memory_examples: Optional[int] = 0,
                 per_task_subset_examples: Optional[int] = 0,
                 cluster_size: Optional[int] = DEFAULT_TOY_DATASET_SIZE,
                 cluster_std: Optional[float] = DEFAULT_CLUSTER_STD):
        super().__init__(num_tasks, per_task_examples, per_task_joint_examples, per_task_memory_examples,
                         per_task_subset_examples)
        self.cluster_size = cluster_size
        self.cluster_std = cluster_std
        self.load_datasets()
        self.prepare_datasets()

    def load_datasets(self):
        for task in range(1, self.num_tasks + 1):
            self.trains[task] = Toy2DCLFDataset(self.num_tasks, task, self.cluster_size, self.cluster_std)
            self.tests[task] = Toy2DCLFDataset(self.num_tasks, task, self.cluster_size, self.cluster_std)

    def precompute_memory_indices(self):
        for task in range(1, self.num_tasks + 1):
            indices_train = self.sample_uniform_class_indices(self.trains[task], 0, 1, self.per_task_memory_examples)
            indices_test = self.sample_uniform_class_indices(self.tests[task], 0, 1, self.per_task_memory_examples)
            assert len(indices_train) == len(indices_test) == self.per_task_memory_examples
            self.memory_indices_train[task] = indices_train[:]
            self.memory_indices_test[task] = indices_test[:]
