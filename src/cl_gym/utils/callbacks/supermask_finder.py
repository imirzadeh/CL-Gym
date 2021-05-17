import os
import torch
from pathlib import Path
from cl_gym.backbones.supermask import SuperMaskMLP
from cl_gym.utils.callbacks import ContinualCallback


class SuperMaskFinder(ContinualCallback):
    def __init__(self):
        super(SuperMaskFinder, self).__init__()
        self.train_loaders, self.test_loaders = {}, {}
        self.params = None
    
    def on_before_fit(self, trainer):
        self.params = trainer.params
        for task in range(1, trainer.params['num_tasks'] + 1):
            device = trainer.params['device']
            batch_size = trainer.params['per_task_subset_examples']
            benchmark = trainer.algorithm.benchmark
            train_loader, test_loader = benchmark.load(task, batch_size=batch_size, shuffle=False,
                                                             pin_memory=True if 'cuda' in str(device) else False)
            self.train_loaders[task], self.test_loaders[task] = train_loader, test_loader
    
    def __build_supermask_net(self, trainer):
        supermask_sparsity = trainer.params.get("supermask_sparsity", 0.5)

        main_net = trainer.algorithm.backbone
        supermask_net = SuperMaskMLP(self.params['input_dim'],
                                     self.params['hidden_1_dim'],
                                     self.params['hidden_2_dim'],
                                     self.params['output_dim'],
                                     supermask_sparsity)

        for layer in [1, 2, 3]:
            supermask_net.replace_weights(layer, main_net.get_layer_weights(layer))
        return supermask_net

    def _train_supermask_net(self, target_task, trainer):
        epochs = trainer.params.get('supermask_train_epochs', 10)
        supermask_net = self.__build_supermask_net(trainer)
        optimizer = torch.optim.SGD([p for p in supermask_net.parameters() if p.requires_grad],
                                    lr=trainer.params.get("supermask_lr", 0.1),
                                    momentum=trainer.params.get("supermask_momentum", 0.9))
        device = trainer.params["device"]
        criterion = torch.nn.CrossEntropyLoss().to(device)
        supermask_net = supermask_net.to(device)
        supermask_net.train()
        for epoch in range(epochs):
            for batch_idx, (data, target, task_ids) in enumerate(self.train_loaders[target_task]):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = supermask_net(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        return supermask_net
        
    def _eval_supermask_net(self, net, target_task, trainer):
        device = trainer.params["device"]
        criterion = torch.nn.CrossEntropyLoss().to(device)
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target, _ in self.test_loaders[target_task]:
                data, target = data.to(device), target.to(device)
                output = net(data)
                total += len(target)
                test_loss += criterion(output, target)
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
        avg_acc = 100.0 * float(correct) / total
        test_loss /= total
        print(f"SuperMask Net Metrics [Task {target_task}]>> loss = {test_loss}, acc={avg_acc}")

    
    def on_after_training_task(self, trainer):
        if trainer.current_task <= 1:
            return
        else:
            supermask_net = self._train_supermask_net(1, trainer)
            self._eval_supermask_net(supermask_net, 1, trainer)