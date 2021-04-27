import torchvision
from typing import Optional, Tuple
from cl_gym.benchmarks.utils import DEFAULT_DATASET_DIR
from cl_gym.benchmarks import Benchmark, SplitDataset
from cl_gym.benchmarks.transforms import get_default_cifar_transform


class SplitCIFAR(Benchmark):
    def __init__(self,
                 num_tasks: int,
                 per_task_examples: Optional[int] = None,
                 per_task_joint_examples: Optional[int] = 0,
                 per_task_memory_examples: Optional[int] = 0,
                 per_task_subset_examples: Optional[int] = 0,
                 task_input_transforms: Optional[list] = None,
                 task_target_transforms: Optional[list] = None,
                 is_cifar_100: bool = True):
        
        # CIFAR-100 vs CIFAR-10 book-keeping variables:
        # CIFAR-100 has 20 tasks (5 classes per task)
        # CIFAR-10  has 5 tasks (2 classes per task)
        self.is_cifar_100 = is_cifar_100
        if task_input_transforms is None:
            task_input_transforms = get_default_cifar_transform(num_tasks, is_cifar_100)
        self.num_classes_per_split = 5 if self.is_cifar_100 else 2
        super().__init__(num_tasks, per_task_examples, per_task_joint_examples, per_task_memory_examples,
                         per_task_subset_examples, task_input_transforms, task_target_transforms)
        self.load_datasets()
        self.prepare_datasets()

    def __load_cifar(self):
        transforms = self.task_input_transforms[0]
        CIFAR_dataset = torchvision.datasets.CIFAR100 if self.is_cifar_100 else torchvision.datasets.CIFAR10
        self.cifar_train = CIFAR_dataset(DEFAULT_DATASET_DIR, train=True, download=True, transform=transforms)
        self.cifar_test = CIFAR_dataset(DEFAULT_DATASET_DIR, train=False, download=True, transform=transforms)

    def load_datasets(self):
        self.__load_cifar()
        for task in range(1, self.num_tasks + 1):
            self.trains[task] = SplitDataset(task, self.num_classes_per_split, self.cifar_train)
            self.tests[task] = SplitDataset(task, self.num_classes_per_split, self.cifar_test)
    
    def precompute_memory_indices(self):
        for task in range(1, self.num_tasks + 1):
            start_cls = (task - 1) * self.num_classes_per_split
            end_cls = task * self.num_classes_per_split - 1
            num_examples = self.per_task_memory_examples
            indices_train = self.sample_uniform_class_indices(self.trains[task], start_cls, end_cls, num_examples)
            indices_test = self.sample_uniform_class_indices(self.tests[task], start_cls, end_cls, num_examples)
            assert len(indices_train) == len(indices_test) == self.per_task_memory_examples
            self.memory_indices_train[task] = indices_train[:]
            self.memory_indices_test[task] = indices_test[:]


class SplitCIFAR100(SplitCIFAR):
    def __init__(self,
                 num_tasks: int,
                 per_task_examples: Optional[int] = None,
                 per_task_joint_examples: Optional[int] = 0,
                 per_task_memory_examples: Optional[int] = 0,
                 per_task_subset_examples: Optional[int] = 0,
                 task_input_transforms: Optional[list] = None,
                 task_target_transforms: Optional[list] = None):

        super().__init__(num_tasks, per_task_examples, per_task_joint_examples, per_task_memory_examples,
                         per_task_subset_examples, task_input_transforms, task_target_transforms, is_cifar_100=True)


class SplitCIFAR10(SplitCIFAR):
    def __init__(self,
                 num_tasks: int,
                 per_task_examples: Optional[int] = None,
                 per_task_joint_examples: Optional[int] = 0,
                 per_task_memory_examples: Optional[int] = 0,
                 per_task_subset_examples: Optional[int] = 0,
                 task_input_transforms: Optional[list] = None,
                 task_target_transforms: Optional[list] = None):
        
        super().__init__(num_tasks, per_task_examples, per_task_joint_examples, per_task_memory_examples,
                         per_task_subset_examples, task_input_transforms, task_target_transforms, is_cifar_100=False)


    
if __name__ == "__main__":
    benchmark = SplitCIFAR100(20)
    # benchmark = SplitMNISTBenchmark(num_tasks=5)
    num_tasks = 20
    for t in range(1, num_tasks+1):
        benchmark.load(t, 32)
