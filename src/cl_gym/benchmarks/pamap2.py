import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from cl_gym.benchmarks import Benchmark
from typing import Optional, Tuple
from torchvision.datasets.utils import download_and_extract_archive

PAMAP_ZIP_URL = "https://www.imirzadeh.me/dl/PAMAP2.zip"
DOWNLOAD_DIR = './data/'


class PAMAP2Loader:
    def __init__(self, subject_id, window_seconds=2.56, overlap=0.5, location=None, test_ratio=0.25):
        download_and_extract_archive(PAMAP_ZIP_URL, DOWNLOAD_DIR)
        self.subject_id = subject_id
        self.frequency_Hz = 100
        self.window_seconds = window_seconds
        self.overlap = overlap
        self.location = location.lower() if location else location
        self.test_ratio = test_ratio
        self.raw_data = []
        self.check_inputs()
        self.load_data()
        self.data = self.extract_windows()
    
    def check_inputs(self):
        if not 1 <= self.subject_id <= 8:
            raise ValueError("PAMAP2 has 8 subjects (1 to 8)")
        if not 0 <= self.overlap < 1:
            raise ValueError("Overlap is between 0 and 1 (i.e., relative to window size)")
        if self.location and self.location not in ['chest', 'hand', 'ankle']:
            raise ValueError("PAMAP2 supposrts 'chest', 'hand', and 'ankle' for locations")
    
    def _get_columns(self):
        columns = ['time', 'activity',
                   'hand:accel_X', 'hand:accel_Y', 'hand:accel_Z',
                   'hand:gyro_X', 'hand:gyro_Y', 'hand:gyro_Z',
                   'chest:accel_X', 'chest:accel_Y', 'chest:accel_Z',
                   'chest:gyro_X', 'chest:gyro_Y', 'chest:gyro_Z',
                   'ankle:accel_X', 'ankle:accel_Y', 'ankle:accel_Z',
                   'ankle:gyro_X', 'ankle:gyro_Y', 'ankle:gyro_Z']
        if self.location:
            columns = ['time', 'activity']
            for axis in ["accel_X", 'accel_Y', 'accel_Z', 'gyro_X', 'gyro_Y', 'gyro_Z']:
                columns.append(f"{self.location}:{axis}")
        return columns
        
    def load_data(self):
        filepath = os.path.join(DOWNLOAD_DIR, 'PAMAP2', '{}.csv'.format(self.subject_id))
        self.raw_data = pd.read_csv(filepath)
    
    def _calculate_window_range(self, idx):
        window_steps = int(self.window_seconds * self.frequency_Hz)
        overlap_steps = int(window_steps * self.overlap)
        window_start = max(idx * window_steps - overlap_steps, 0)
        window_end = window_start + window_steps
        return window_start, window_end
    
    @staticmethod
    def correct_labels(label):
        # PAMAP labels: [ 1  2  3 17 16 12 13  4  6  7  5 24]
        if 1 <= label <= 7:
            return label - 1
        else:
            map = {12: 7, 13: 8, 16: 9, 17: 10, 24: 11}
            return map[label]
    
    def extract_windows(self):
        columns = self._get_columns()
        df = self.raw_data[columns].sort_values(by="time").drop("time", axis=1)
        df[['activity']] = df[['activity']].applymap(lambda x: self.correct_labels(x))
        idx = 0
        windows = []
        while True:
            start, end = self._calculate_window_range(idx)
            if end > len(df)-1:
                break
            window = df.iloc[start:end]
            windows.append(window)
            idx += 1
        return windows
    
    def create_train_test(self):
        data = self.data
        # shuffle
        np.random.shuffle(data)
        
        # separate train and test
        num_trains = int((1-self.test_ratio) * len(data))
        trains = data[:num_trains]
        tests = data[num_trains:]
        return trains, tests
        

class PAMAP2Dataset(Dataset):
    def __init__(self, data):
        # data is list of pandas df
        inputs = []
        targets = []
        for d in data:
            inputs.append(d.drop('activity', axis=1).fillna(method='ffill', axis=1).fillna(value=0).to_numpy(dtype=np.float32)/10.0)
            targets.append(d.iloc[0].activity)
        
        inputs = np.array(inputs)
        targets = np.array(targets, dtype=np.long)
        self.inp = torch.from_numpy(inputs)
        self.targets = torch.from_numpy(targets)
    
    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.inp[idx, :], self.targets[idx], [None]*len(idx)


class PAMAP2(Benchmark):
    """
    PAMAP2 benchmark: activity recognition with at most 8 tasks.
    This is a time-series benchmark.
    """
    def __init__(self,
                 num_tasks: int = 8,
                 per_task_examples: Optional[int] = None,
                 per_task_joint_examples: Optional[int] = 0,
                 per_task_memory_examples: Optional[int] = 0,
                 shuffle_subjects=True):
        # TODO: Add window-size and overlap to benchmark
        super().__init__(num_tasks, per_task_examples, per_task_joint_examples, per_task_memory_examples)
        self.shuffle_subjects = shuffle_subjects
        self.load_datasets()
        self.prepare_datasets()
    
    def load_datasets(self):
        subject_order = list(range(1, 9))
        if self.shuffle_subjects:
            np.random.shuffle(subject_order)
            
        for task in range(1, self.num_tasks+1):
            trains, tests = PAMAP2Loader(subject_order[task-1]).create_train_test()
            self.trains[task] = PAMAP2Dataset(trains)
            self.tests[task] = PAMAP2Dataset(tests)
            
    def precompute_memory_indices(self):
        for task in range(1, self.num_tasks + 1):
            indices_train = self.sample_uniform_class_indices(self.trains[task], 0, 11, self.per_task_memory_examples)
            indices_test = self.sample_uniform_class_indices(self.tests[task], 0, 11, self.per_task_memory_examples)
            # assert len(indices_train) == len(indices_test)
            self.memory_indices_train[task] = indices_train[:]
            self.memory_indices_test[task] = indices_test[:]