import torch
import torchvision
import numpy as np
from numpy.random import randint
from typing import Tuple, Optional, Dict
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from PIL import Image


class Benchmark:
    """
    Base class for continual learning benchmark that includes loading/serving continual learning datasets.
    """
    def __init__(self,
                 num_tasks: int,
                 per_task_examples: Optional[int] = None,
                 per_task_joint_examples: Optional[int] = 0,
                 per_task_memory_examples: Optional[int] = 0,
                 per_task_subset_examples: Optional[int] = 0,
                 task_input_transforms: Optional[list] = None,
                 task_target_transforms: Optional[list] = None):
        """
        
        :param num_tasks: Number of tasks in benchmark.
        :param per_task_examples: Optional, To load only part of examples per task and not all examples.
        :param per_task_joint_examples: Optional, for joint (multitask) training.
        :param per_task_memory_examples: Optional, for creating episodic memory/replay buffer.
        :param per_task_subset_examples: Optional, for accessing to subset of task data, in addition to all examples.
        """
        
        # Task details
        self.num_tasks = num_tasks
        self.per_task_seq_examples = per_task_examples
        self.per_task_joint_examples = per_task_joint_examples
        self.per_task_memory_examples = per_task_memory_examples
        self.per_task_subset_examples = per_task_subset_examples
        
        # Optional transformations
        self.task_input_transforms = task_input_transforms
        self.task_target_transforms = task_target_transforms

        # Book-keeping variables: mostly used for storing indices of data points for each task
        self.trains, self.tests = {}, {}
        self.joint_indices_train, self.joint_indices_test = {}, {}
        self.memory_indices_train, self.memory_indices_test = {}, {}
        self.seq_indices_train, self.seq_indices_test = {}, {}
        self.subset_indices_train, self.subset_indices_test = {}, {}
        self.sanity_check_inputs()
    
    def sanity_check_inputs(self):
        self.sanity_check_transforms()
    
    def sanity_check_transforms(self):
        if self.task_input_transforms is not None and len(self.task_input_transforms) != self.num_tasks:
            raise ValueError("task_input_transform is either None (using default transform),\
                             or should be a list of size `num_tasks` that provides transforms for each task")
            
    def prepare_datasets(self):
        if self.per_task_joint_examples:
            self.precompute_joint_indices()
        if self.per_task_memory_examples:
            self.precompute_memory_indices()
        if self.per_task_seq_examples:
            self.precompute_seq_indices()
        if self.per_task_subset_examples:
            self.precompute_subset_indices()
   
    def load_datasets(self):
        raise NotImplementedError
    
    def precompute_joint_indices(self):
        for task in range(1, self.num_tasks+1):
            self.joint_indices_train[task] = randint(0, len(self.trains[task]), size=self.per_task_joint_examples)
            self.joint_indices_test[task] = randint(0, len(self.tests[task]), size=self.per_task_joint_examples)
            
    def precompute_memory_indices(self):
        raise NotImplementedError
    
    def precompute_subset_indices(self):
        for task in range(1, self.num_tasks + 1):
            self.subset_indices_train[task] = randint(0, len(self.trains[task]), size=self.per_task_subset_examples)
            self.subset_indices_test[task] = randint(0, len(self.tests[task]), size=min(self.per_task_subset_examples, len(self.tests[task])))

    def precompute_seq_indices(self):
        if self.per_task_seq_examples > len(self.trains[1]):
            raise ValueError(f"per task examples = {self.per_task_seq_examples} but first task's examples = {len(self.trains[1])}")
        
        for task in range(1, self.num_tasks+1):
            self.seq_indices_train[task] = randint(0, len(self.trains[task]), size=self.per_task_seq_examples)
            self.seq_indices_test[task] = randint(0, len(self.tests[task]), size=min(self.per_task_seq_examples, len(self.tests[task])))
    
    def _calculate_num_examples_per_class(self, start_class, end_class, num_samples):
        num_classes = end_class - start_class + 1
        num_examples_per_class = num_samples//num_classes
        result = [num_examples_per_class]*num_classes
        
        # if memory_size can't be divided by num_class classes
        # e.g., memory_size is 32, but we have 5 classes.
        if num_classes * num_examples_per_class < num_samples:
            # how many more examples we need?
            diff = num_samples - (num_classes * num_examples_per_class)
            # add examples
            while diff:
                diff -= 1
                result[randint(0, num_classes)] += 1
        return result
   
    def sample_uniform_class_indices(self, dataset, start_class, end_class, num_samples):
        target_classes = dataset.targets.clone().detach().numpy()
        num_examples_per_class = self._calculate_num_examples_per_class(start_class, end_class, num_samples)
        class_indices = []
        # choose num_examples_per_class for each class
        for i, cls_number in enumerate(range(start_class, end_class+1)):
            target = (target_classes == cls_number)
            #  maybe that class doesn't exist
            num_candidate_examples = len(np.where(target == 1)[0])
            if num_candidate_examples:
                selected_indices = np.random.choice(np.where(target == 1)[0],
                                                    min(num_candidate_examples, num_examples_per_class[i]),
                                                    replace=False)
                class_indices += list(selected_indices)
        return class_indices

    def load(self,
             task: int,
             batch_size: int,
             shuffle: Optional[bool] = True,
             num_workers: Optional[int] = 0,
             pin_memory: Optional[bool] = True) -> Tuple[DataLoader, DataLoader]:
        
        if task > self.num_tasks:
            raise ValueError(f"Asked to load task {task} but the benchmark has {self.num_tasks} tasks")
        
        if self.per_task_seq_examples:
            trainset = Subset(self.trains[task], self.seq_indices_train[task])
            testset = Subset(self.tests[task], self.seq_indices_test[task])
        else:
            trainset = self.trains[task]
            testset = self.tests[task]
        train_loader = DataLoader(trainset, batch_size, shuffle, num_workers=num_workers,
                                  pin_memory=pin_memory)
        test_loader = DataLoader(testset, 256, True, num_workers=num_workers, pin_memory=pin_memory)
    
        return train_loader, test_loader

    def load_joint(self,
                   task: int,
                   batch_size: int,
                   shuffle: Optional[bool] = True,
                   num_workers: Optional[int] = 0,
                   pin_memory: Optional[bool] = True) -> Tuple[DataLoader, DataLoader]:
    
        if not self.per_task_joint_examples:
            raise ValueError("Called load_joint() but per_task_joint_examples is not set")
        
        if task > self.num_tasks:
            raise ValueError(f"Asked to load task {task} but the benchmark has {self.num_tasks} tasks")
    
        trains, tests = [], []
        for prev_task in range(1, task + 1):
            prev_train = Subset(self.trains[prev_task], self.joint_indices_train[prev_task])
            prev_test = Subset(self.tests[prev_task], self.joint_indices_test[prev_task])
            trains.append(prev_train)
            tests.append(prev_test)
    
        trains, tests = ConcatDataset(trains), ConcatDataset(tests)
        train_loader = DataLoader(trains, batch_size, shuffle, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(tests, batch_size, shuffle, num_workers=num_workers, pin_memory=pin_memory)
        return train_loader, test_loader

    def load_subset(self,
                    task: int,
                    batch_size: int,
                    shuffle: Optional[bool] = True,
                    num_workers: Optional[int] = 0,
                    pin_memory: Optional[bool] = True) -> Tuple[DataLoader, DataLoader]:
        if not self.per_task_subset_examples:
            raise ValueError("Called load_subset() without setting per_task_subset_examples")
        subset_train = Subset(self.trains[task], self.subset_indices_train[task])
        subset_test = Subset(self.tests[task], self.subset_indices_test[task])
        train_loader = DataLoader(subset_train, batch_size, shuffle, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(subset_test, batch_size, shuffle, num_workers=num_workers, pin_memory=pin_memory)
        return train_loader, test_loader

    def load_memory(self,
                    task: int,
                    batch_size: int,
                    shuffle: Optional[bool] = True,
                    num_workers: Optional[int] = 0,
                    pin_memory: Optional[bool] = True) -> Tuple[DataLoader, DataLoader]:
        if not self.per_task_memory_examples:
            raise ValueError("Called load_memory() but per_task_memory_examples is not set")
        
        if task > self.num_tasks:
            raise ValueError(f"Asked for memory of task={task} while the benchmark has {self.num_tasks} tasks")
    
        train_indices = self.memory_indices_train[task]
        test_indices = self.memory_indices_test[task]
        train_dataset = Subset(self.trains[task], train_indices)
        test_dataset = Subset(self.tests[task], test_indices)
    
        train_loader = DataLoader(train_dataset, batch_size, shuffle, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(test_dataset, batch_size, shuffle, num_workers=num_workers, pin_memory=pin_memory)
    
        return train_loader, test_loader

    def load_memory_joint(self,
                          task: int,
                          batch_size: int,
                          shuffle: Optional[bool] = True,
                          num_workers: Optional[int] = 0,
                          pin_memory: Optional[bool] = True) -> Tuple[DataLoader, DataLoader]:
        if task > self.num_tasks:
            raise ValueError(f"Asked to load memory of task={task} but the benchmark has {self.num_tasks} tasks")
        trains, tests = [], []
        for t in range(1, task + 1):
            train_indices = self.memory_indices_train[t]
            test_indices = self.memory_indices_test[t]
            train_dataset = Subset(self.trains[t], train_indices)
            test_dataset = Subset(self.tests[t], test_indices)
            trains.append(train_dataset)
            tests.append(test_dataset)
    
        trains, tests = ConcatDataset(trains), ConcatDataset(tests)
        train_loader = DataLoader(trains, batch_size, shuffle, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(tests, batch_size, shuffle, num_workers=num_workers, pin_memory=pin_memory)
        return train_loader, test_loader


class DynamicTransformDataset(Dataset):
    def __init__(self, task_id: int, dataset: Dataset, input_transform=None, target_transform=None):
        self.task_id = task_id
        self.dataset = dataset
        self.input_transform = input_transform
        self.target_transform = target_transform
    
    def __getitem__(self, index):
        img, target = self.dataset.data[index], int(self.dataset.targets[index])
        img = Image.fromarray(img.numpy(), mode='L')
        
        # transforms
        if self.input_transform is not None:
            img = self.input_transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target, self.task_id
    
    def __len__(self) -> int:
        return len(self.dataset.data)


class SplitDataset(Dataset):
    def __init__(self, task_id, classes_per_split, dataset):
        self.inputs = []
        self.targets = []
        self.task_id = task_id
        self.classes_per_split = classes_per_split
        self.__build_split(dataset, task_id)

    def __build_split(self, dataset, task_id):
        start_class = (task_id-1) * self.classes_per_split
        end_class = task_id * self.classes_per_split
        # For CIFAR
        if isinstance(dataset.targets, list):
            target_classes = np.asarray(dataset.targets)
        else:
            target_classes = dataset.targets.clone().detach().numpy()
        # target_classes = dataset.targets.clone().detach().numpy()
        selected_indices = np.where(np.logical_and(start_class <= target_classes, target_classes < end_class))[0]
        for idx in selected_indices:
            img, target = dataset[idx]
            target = torch.tensor(target)
            self.inputs.append(img)
            self.targets.append(target)
        
        self.inputs = torch.stack(self.inputs)
        self.targets = torch.stack(self.targets)

    def __getitem__(self, index: int):
        img, target = self.inputs[index], int(self.targets[index])
        return img, target, self.task_id

    def __len__(self) -> int:
        return len(self.inputs)