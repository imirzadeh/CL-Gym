import torch
# from torch.utils.data import DataLoader, TensorDataset


class ContinualAlgorithm:
    """
    | Base class for continual learning algorithms.
    | It contains abstractions for implementing different algorithms, and also implementations shared among all algorithms.
    | It can be used for Naive(Finetune) algorithm or Stable SGD algorithm by Mirzadeh et. al.
    """
    def __init__(self, backbone, benchmark, params, requires_memory=False):
        """
        :param backbone: Neural network.
        :param benchmark: Benchmark object for preparing task training/memory data.
        :param params:
        :param requires_memory: Whether the algorithm needs episodic memory/replay buffer (e.g., A-GEM) or not.
        """
        self.backbone = backbone
        self.benchmark = benchmark
        self.params = params
        self.current_task = 1
        self.requires_memory = requires_memory
        self.episodic_memory_loader = None
        self.episodic_memory_iter = None
    
    def setup(self):
        pass
    
    def teardown(self):
        pass
    
    def update_episodic_memory(self):
        self.episodic_memory_loader, _ = self.benchmark.load_memory_joint(self.current_task,
                                                                          batch_size=self.params['batch_size_memory'],
                                                                          shuffle=True,
                                                                          pin_memory=True)
        self.episodic_memory_iter = iter(self.episodic_memory_loader)
    
    def sample_batch_from_memory(self):
        try:
            batch = next(self.episodic_memory_iter)
        except StopIteration:
            self.episodic_memory_iter = iter(self.episodic_memory_loader)
            batch = next(self.episodic_memory_iter)
        
        device = self.params['device']
        inp, targ, task_id = batch
        return inp.to(device), targ.to(device), task_id.to(device)

    def training_task_end(self):
        if self.requires_memory:
            self.update_episodic_memory()
        self.current_task += 1
    
    def training_epoch_end(self):
        pass
    
    def training_step_end(self):
        pass
    
    def prepare_train_loader(self, task_id):
        num_workers = self.params.get('num_dataloader_workers', 0)
        return self.benchmark.load(task_id, self.params['batch_size_train'],
                                   num_workers=num_workers, pin_memory=True)[0]
    
    def prepare_validation_loader(self, task_id):
        num_workers = self.params.get('num_dataloader_workers', 0)
        return self.benchmark.load(task_id, self.params['batch_size_validation'],
                                   num_workers=num_workers, pin_memory=True)[1]
    
    def prepare_optimizer(self, task_id):
        if self.params.get('learning_rate_decay'):
            lr_lower_bound = self.params.get('learning_rate_lower_bound', 0.0)
            lr = max(self.params['learning_rate'] * (self.params['learning_rate_decay'] ** (task_id-1)), lr_lower_bound)
        else:
            lr = self.params['learning_rate']
        if self.params['optimizer'].lower() == 'sgd':
            return torch.optim.SGD(self.backbone.parameters(), lr=lr, momentum=self.params['momentum'])
        elif self.params['optimizer'].lower() == 'adam':
            return torch.optim.Adam(self.backbone.parameters(), lr=lr)
        else:
            raise ValueError("Only 'SGD' and 'Adam' are accepted. To use another optimizer, override this method.")

    def prepare_criterion(self, task_id):
        return self.params['criterion']

    def training_step(self, task_ids, inp, targ, optimizer, criterion):
        optimizer.zero_grad()
        pred = self.backbone(inp, task_ids)
        loss = criterion(pred, targ)
        loss.backward()
        optimizer.step()
