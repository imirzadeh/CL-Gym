import torchvision
from typing import Optional, Tuple
from cl_gym.benchmarks.utils import DEFAULT_DATASET_DIR
from cl_gym.benchmarks.base import Benchmark, DynamicTransformDataset
from cl_gym.benchmarks.transforms import get_default_mnist_fashion_mnist_transform


class MNISTFashionMNIST(Benchmark):
    def __init__(self,
                 num_tasks: int = 2,
                 per_task_examples: Optional[int] = None,
                 per_task_joint_examples: Optional[int] = 0,
                 per_task_memory_examples: Optional[int] = 0,
                 per_task_subset_examples: Optional[int] = 0,
                 task_input_transforms: Optional[list] = None,
                 task_target_transforms: Optional[list] = None):
        
        if num_tasks != 2:
            raise ValueError("MNIST-FashionMNIST benchmark can have only two tasks")
        
        if task_input_transforms is None:
            task_input_transforms = get_default_mnist_fashion_mnist_transform()
            
        super().__init__(num_tasks, per_task_examples, per_task_joint_examples, per_task_memory_examples,
                         per_task_subset_examples, task_input_transforms, task_target_transforms)
        
        self.load_datasets()
        self.prepare_datasets()
    
    def __load_mnist(self):
        self.mnist_train = torchvision.datasets.MNIST(DEFAULT_DATASET_DIR, train=True, download=True)
        self.mnist_test = torchvision.datasets.MNIST(DEFAULT_DATASET_DIR, train=False, download=True)
        
    def __load_fashion_mnist(self):
        self.mnist_train = torchvision.datasets.FashionMNIST(DEFAULT_DATASET_DIR, train=True, download=True)
        self.mnist_test = torchvision.datasets.FashionMNIST(DEFAULT_DATASET_DIR, train=False, download=True)

    def load_datasets(self):
        for task in range(1, self.num_tasks + 1):
            if task == 1:
                self.__load_mnist()
            else:
                self.__load_fashion_mnist()
            input_transform = self.task_input_transforms[task - 1]
            target_transform = self.task_target_transforms[task - 1] if self.task_target_transforms else None
            self.trains[task] = DynamicTransformDataset(task, self.mnist_train, input_transform, target_transform)
            self.tests[task] = DynamicTransformDataset(task, self.mnist_test, input_transform, target_transform)
    
    def precompute_memory_indices(self):
        for task in range(1, self.num_tasks + 1):
            indices_train = self.sample_uniform_class_indices(self.trains[task].dataset, 0, 9,
                                                              self.per_task_memory_examples)
            indices_test = self.sample_uniform_class_indices(self.tests[task].dataset, 0, 9,
                                                             self.per_task_memory_examples)
            assert len(indices_train) == len(indices_test) == self.per_task_memory_examples
            self.memory_indices_train[task] = indices_train[:]
            self.memory_indices_test[task] = indices_test[:]

