import torch
from typing import Dict, Iterable, Optional
from cl_gym.algorithms import ContinualAlgorithm
from cl_gym.utils.callbacks import ContinualCallback
from cl_gym.utils.loggers import Logger
from cl_gym.trainer.state_manager import TrainerStateManagerMixin
from cl_gym.trainer.callback_hooks import TrainerCallbackHookMixin


class ContinualTrainer(TrainerStateManagerMixin,
                       TrainerCallbackHookMixin):
    """
    Base class for Trainer component. Basically, orchestrates the training by implementing a state-machine.
    For further info, please see the guides on how the trainer works.
    """
    def __init__(self,
                 algorithm: ContinualAlgorithm,
                 params: dict,
                 callbacks=Iterable[ContinualCallback],
                 logger: Optional[Logger] = None):
        self.params = params
        self.algorithm = algorithm
        self.callbacks = callbacks
        self.logger = logger
    
    def setup(self):
        for cb in self.callbacks:
            cb.connect(self)
    
    def teardown(self):
        pass
    
    def train_algorithm_on_task(self, task: int):
        train_loader = self.algorithm.prepare_train_loader(task)
        optimizer = self.algorithm.prepare_optimizer(task)
        criterion = self.algorithm.prepare_criterion(task)
        device = self.params['device']
        for epoch in range(1, self.params['epochs_per_task']+1):
            self.on_before_training_epoch()
            self.tick('epoch')
            self.algorithm.backbone.train()
            self.algorithm.backbone = self.algorithm.backbone.to(device)
            for batch_idx, (inp, targ, task_ids) in enumerate(train_loader):
                self.on_before_training_step()
                self.tick('step')
                self.algorithm.training_step(task_ids.to(device), inp.to(device), targ.to(device), optimizer, criterion)
                self.algorithm.training_step_end()
                self.on_after_training_step()
            self.algorithm.training_epoch_end()
            self.on_after_training_epoch()
        self.algorithm.training_task_end()
    
    def validate_algorithm_on_task(self, task: int, validate_on_train: bool = False) -> Dict[str, float]:
        self.algorithm.backbone.eval()
        device = self.params['device']
        self.algorithm.backbone = self.algorithm.backbone.to(device)
        test_loss = 0
        correct = 0
        total = 0
        if validate_on_train:
            eval_loader = self.algorithm.prepare_train_loader(task)
        else:
            eval_loader = self.algorithm.prepare_validation_loader(task)
        criterion = self.algorithm.prepare_criterion(task)
        with torch.no_grad():
            for (inp, targ, task_ids) in eval_loader:
                inp, targ, task_ids = inp.to(device), targ.to(device), task_ids.to(device)
                pred = self.algorithm.backbone(inp, task_ids)
                total += len(targ)
                test_loss += criterion(pred, targ).item()
                pred = pred.data.max(1, keepdim=True)[1]
                correct += pred.eq(targ.data.view_as(pred)).sum()
            test_loss /= total
            correct = correct.cpu()
            avg_acc = 100.0 * float(correct.numpy()) / total
            return {'accuracy': avg_acc, 'loss': test_loss}

    def fit(self):
        for task in range(1, self.params['num_tasks']+1):
            self.on_before_training_task()
            self.tick('task')
            self.train_algorithm_on_task(task)
            self.on_after_training_task()
    
    def _run_setup(self):
        self.on_before_setup()
        self.setup()
        self.on_after_setup()
    
    def _run_fit(self):
        self.on_before_fit()
        self.fit()
        self.on_after_fit()
    
    def _run_teardown(self):
        self.on_before_teardown()
        self.teardown()
        self.on_after_teardown()

    def run(self):
        # setup
        self._run_setup()
        # fit: main training loop
        self._run_fit()
        # teardown
        self._run_teardown()
