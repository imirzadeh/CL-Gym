import torch


class ContinualTrainer:
    def __init__(self, algorithm, params, callbacks=(), logger=None):
        self.params = params
        self.algorithm = algorithm
        self.callbacks = callbacks
        self.current_step = 1
        self.current_epoch = 1
        self.current_task = 1
        self.logger = logger
    
    # TODO: instead of on_before/after_<stage>, write call_hook()
    def on_before_fit(self):
        for cb in self.callbacks:
            cb.on_before_fit(self)
    
    def on_after_fit(self):
        for cb in self.callbacks:
            cb.on_after_fit(self)

    def on_before_training_task(self):
        for cb in self.callbacks:
            cb.on_before_training_task(self)

    def on_after_training_task(self):
        for cb in self.callbacks:
            cb.on_after_training_task(self)
        self.current_task += 1

    def on_before_training_epoch(self):
        for cb in self.callbacks:
            cb.on_before_training_epoch(self)
            
    def on_after_training_epoch(self):
        for cb in self.callbacks:
            cb.on_before_training_epoch(self)
        self.current_epoch += 1

    def on_before_training_step(self):
        for cb in self.callbacks:
            cb.on_before_training_step(self)

    def on_after_training_step(self):
        for cb in self.callbacks:
            cb.on_after_training_step(self)
        self.current_step += 1
    
    def train_algorithm_on_task(self, task):
        train_loader = self.algorithm.prepare_train_loader(task)
        optimizer = self.algorithm.prepare_optimizer(task)
        criterion = self.algorithm.prepare_criterion(task)
        
        for epoch in range(1, self.params['epochs_per_task']+1):
            self.on_before_training_epoch()
            self.algorithm.backbone.train()
            for batch_idx, (inp, targ, task_id) in enumerate(train_loader):
                self.algorithm.training_step(task, inp, targ, optimizer, criterion)
                self.algorithm.training_step_end()
                self.on_after_training_step()
            self.algorithm.training_epoch_end()
            self.on_after_training_epoch()
        self.algorithm.training_task_end()
    
    def validate_algorithm_on_task(self, task, validate_on_train=False):
        self.algorithm.backbone.eval()
        test_loss = 0
        correct = 0
        total = 0
        if validate_on_train:
            eval_loader = self.algorithm.prepare_train_loader(task)
        else:
            eval_loader = self.algorithm.prepare_validation_loader(task)
        criterion = self.algorithm.prepare_criterion(task)
        with torch.no_grad():
            for (inp, targ, task_id) in eval_loader:
                pred = self.algorithm.backbone(inp, task)
                total += len(targ)
                test_loss += criterion(pred, targ).item()
                pred = pred.data.max(1, keepdim=True)[1]
                correct += pred.eq(targ.data.view_as(pred)).sum()
            test_loss /= total
            correct = correct.to('cpu')
            avg_acc = 100.0 * float(correct.numpy()) / total
            return {'accuracy': avg_acc, 'loss': test_loss}

    def fit(self):
        for task in range(1, self.params['num_tasks']+1):
            self.on_before_training_task()
            self.train_algorithm_on_task(task)
            self.on_after_training_task()
    
    def run(self):
        self.on_before_fit()
        self.fit()
        self.on_after_fit()
