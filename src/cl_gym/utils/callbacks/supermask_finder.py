import os
import copy
import torch
import numpy as np
from pathlib import Path
from cl_gym.backbones.supermask import SuperMaskMLP
from cl_gym.utils.callbacks import ContinualCallback
import matplotlib
import seaborn as sns
from matplotlib import pyplot as plt


class SuperMaskFinder(ContinualCallback):
    def __init__(self, intervals):
        super(SuperMaskFinder, self).__init__()
        self.intervals = intervals
        self.train_loaders, self.test_loaders = {}, {}
        self.params = None
        self.save_path, self.plot_path = None, None
        self.task_masks = {}
        self.mask_history = {}
        
    def on_before_fit(self, trainer):
        self.params = trainer.params
        self.save_path = os.path.join(trainer.params['output_dir'], 'metrics')
        self.plot_path = os.path.join(trainer.params['output_dir'], 'plots')
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        Path(self.plot_path).mkdir(parents=True, exist_ok=True)

        for task in range(1, trainer.params['num_tasks'] + 1):
            device = trainer.params['device']
            batch_size = trainer.params['batch_size_train']
            benchmark = trainer.algorithm.benchmark
            num_workers = min(4, int(os.cpu_count()//4))
            train_loader, test_loader = benchmark.load(task, batch_size=batch_size,
                                                       num_workers=num_workers,
                                                       shuffle=False,
                                                       pin_memory=True if 'cuda' in str(device) else False)
            self.train_loaders[task], self.test_loaders[task] = train_loader, test_loader
    
    def __build_supermask_net(self, trainer):
        supermask_sparsity = trainer.params.get("supermask_sparsity", 0.2)

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
        return {"loss": test_loss, 'accuracy': avg_acc}
    
    def train_and_eval(self, trainer):
        for task in range(1, trainer.current_task + 1):
            if task == trainer.current_task:
                supermask_net = self._train_supermask_net(task, trainer)
                self.task_masks[task] = copy.deepcopy(supermask_net)
            else:
                supermask_net = self.task_masks[task]
                for layer in [1, 2, 3]:
                    supermask_net.replace_weights(layer, trainer.algorithm.backbone.get_layer_weights(layer))

            metrics = self._eval_supermask_net(supermask_net, task, trainer)
            step = trainer.current_task if self.intervals == 'tasks' else trainer.current_epoch
            self.log_metric(trainer, f'sup_acc_{task}', round(metrics['accuracy'], 2), step)
    
    def _extract_masks(self, supermask_net):
        masks = {'1': supermask_net.w1.get_supermask().clone().cpu().numpy(),
                 '2': supermask_net.w2.get_supermask().clone().cpu().numpy(),
                 '3': supermask_net.w3.get_supermask().clone().cpu().numpy()}
        return masks

    def on_after_training_task(self, trainer):
        if self.intervals == 'tasks':
            self.train_and_eval(trainer)
        task_masks = self._extract_masks(self.task_masks[trainer.current_task])
        self.mask_history[trainer.current_task] = task_masks
        
    def on_after_training_epoch(self, trainer):
        if self.intervals == 'epochs':
            self.train_and_eval(trainer)
    
    def _plot_masks(self, task, trainer):
        plt.close('all')
        data = self.mask_history[task]
        sns.set_context("paper", rc={"lines.linewidth": 4.5,
                                     'xtick.labelsize': 12,
                                     'ytick.labelsize': 12,
                                     'lines.markersize': 8,
                                     'legend.fontsize': 19,
                                     'axes.labelsize': 23,
                                     'legend.handlelength': 0.7,
                                     'legend.handleheight': 1, })
        
        for layer in [1, 2, 3]:
            sns.heatmap(data[str(layer)].T, vmin=0, vmax=1.0, cmap='plasma', square=False)
            trainer.logger.log_figure(plt, f"masks_task_{task}_layer_{layer}")
            plt.close('all')
            
        
    def on_after_fit(self, trainer):
        for task in range(1, trainer.current_task):
            filename = os.path.join(self.save_path, f"masks_task_{task}.npz")
            np.savez(filename, **self.mask_history[task])
            self._plot_masks(task, trainer)
