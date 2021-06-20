import torchvision
from typing import Optional, Tuple
from cl_gym.benchmarks.utils import DEFAULT_DATASET_DIR
from cl_gym.benchmarks import Benchmark, SplitDataset
from cl_gym.benchmarks.transforms import get_default_cifar_transform


class SplitCIFAR(Benchmark):
    """
    Base class for Split-CIFAR benchmarks.
    """
    def __init__(self,
                 num_tasks: int,
                 per_task_examples: Optional[int] = None,
                 per_task_joint_examples: Optional[int] = 0,
                 per_task_memory_examples: Optional[int] = 0,
                 per_task_subset_examples: Optional[int] = 0,
                 task_input_transforms: Optional[list] = None,
                 task_target_transforms: Optional[list] = None,
                 is_cifar_100: bool = True):
        """
        Args:
            num_tasks: Number of tasks. 20 for CIFAR-100 and 5 for CIFAR-10.
            per_task_examples: If set, each task will include part of the original benchmark rather than full data.
            per_task_joint_examples: If set, the benchmark will support joint/multitask loading of tasks.
            per_task_memory_examples: If set, the benchmark will support episodic memory/replay buffer loading of tasks.
            per_task_subset_examples: If set, the benchmark will support loading a pre-defined subset of each task.
            task_input_transforms: If set, the benchmark will use the provided torchvision transform.
            task_target_transforms: If set, the benchmark will use the provided target transform for targets.
            is_cifar_100: If true, it will set prepare for CIFAR-100, otherwise CIFAR-10.
            
        . note::
            If :attr:`task_input_transforms` or :attr:`task_target_transforms`, they should be a list
            of size `num_tasks` where each element of the list can be a torchvision (Composed) transform.
            
        . note::
            Similar to Torchvision, CIFAR benchmarks in CL-Gym are first loaded in the memory for faster loading.
        """
        
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
        """
        Loades CIFAR-Dataset [In memory]
        """
        self.__load_cifar()
        for task in range(1, self.num_tasks + 1):
            self.trains[task] = SplitDataset(task, self.num_classes_per_split, self.cifar_train)
            self.tests[task] = SplitDataset(task, self.num_classes_per_split, self.cifar_test)
    
    def precompute_memory_indices(self):
        """
        Precomputes memory indices for each task.
        
        . note::
            The default behavior is class-uniform sampling.
            i.e., each class will have roughly equal number of samples in the memory.
            You can inherit this class and override this method for custom behavior. But a better way
            is to move this logic to your algorithm component's code.
        """
        for task in range(1, self.num_tasks + 1):
            start_cls = (task - 1) * self.num_classes_per_split
            end_cls = task * self.num_classes_per_split - 1
            num_examples = self.per_task_memory_examples
            indices_train = self.sample_uniform_class_indices(self.trains[task], start_cls, end_cls, num_examples)
            indices_test = self.sample_uniform_class_indices(self.tests[task], start_cls, end_cls, num_examples)
            assert len(indices_train)  == self.per_task_memory_examples
            self.memory_indices_train[task] = indices_train[:]
            self.memory_indices_test[task] = indices_test[:]


class SplitCIFAR100(SplitCIFAR):
    """
    Split CIFAR-100 benchmark.
    Has 20 tasks, each task with 5 classes of CIFAR-100.
    """
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
    """
    Split CIFAR-10 benchmark.
    has 5 tasks, each with 2 classes of CIFAR-10.
    """
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

