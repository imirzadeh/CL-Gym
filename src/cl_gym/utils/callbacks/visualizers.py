import os
import torch
import matplotlib
import numpy as np
import seaborn as sns
from pathlib import Path
from matplotlib import rc
import torch.nn.functional as F
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from cl_gym.utils.callbacks import ContinualCallback

matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
np.set_printoptions(precision=4, suppress=True)


class ToyRegressionVisualizer(ContinualCallback):
    def __init__(self):
        self.map_functions = [lambda x: (x + 3.),
                              lambda x: 2. * np.power(x, 2) - 1,
                              lambda x: np.power(x - 3., 3)]
        self.domains = [[-4, -2], [-1, 1], [2, 4]]
        self.colors = ['#36008D', '#FE5E54', '#00C9B8']
        self.x_min = -4.5
        self.x_max = 4.5
        self.save_path = None
        
        super(ToyRegressionVisualizer, self).__init__()
    
    def on_before_fit(self, trainer):
        self.save_path = os.path.join(trainer.params['output_dir'], 'plots')
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
    
    def plot_task(self, trainer, task=1):
        net = trainer.algorithm.backbone
        net.eval()
        # data
        num_examples = 12
        x_min, x_max = self.domains[task - 1]
        color = self.colors[task - 1]
        data = np.linspace(x_min, x_max, num_examples).reshape((num_examples, 1))
        test_x = torch.from_numpy(data).float()
        test_x = test_x.to(trainer.params['device'])
        test_y = np.vectorize(self.map_functions[task - 1])(test_x.cpu().numpy()).reshape(num_examples, 1)
        
        pred = net(test_x).to('cpu').detach().clone().numpy().reshape(num_examples)
        plt.plot(data.reshape(num_examples), test_y.reshape(num_examples),
                 color=color, alpha=0.6, linewidth=3)
        plt.plot(test_x.cpu().numpy().reshape(num_examples), pred,
                 color=color, linewidth=3, linestyle='--')
        plt.ylim(-2.5, 2.5)
        plt.yticks([-2, -1, 0, 1, 2])
        plt.xticks([-4, -3, -2, -1, 0, 1, 2, 3, 4])
        plt.xlim(-4.5, 4.5)
    
    def on_after_training_task(self, trainer):
        for task in range(1, trainer.current_task + 1):
            self.plot_task(trainer, task)
        filename = f"reg_task_{trainer.current_task}"
        path = f"{os.path.join(self.save_path, filename)}.pdf"
        if trainer.logger:
            trainer.logger.log_figure(plt, filename)
        plt.savefig(path, dpi=220)
        plt.close('all')


class ToyClassificationVisualizer(ContinualCallback):
    def __init__(self):
        super(ToyClassificationVisualizer, self).__init__('visualizer')
    
    def on_before_fit(self, trainer):
        self.save_path = os.path.join(trainer.params['output_dir'], 'plots')
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
    
    def extract_points(self, loader):
        xs, ys = [], []
        for inp, targ, task_ids in loader:
            batch_size = len(inp)
            for batch in range(batch_size):
                xs.append(inp[batch].numpy())
                ys.append(targ[batch])
        return np.array(xs), np.array(ys)
    
    def __set_viz_context(self):
        sns.set_context("paper", rc={"lines.linewidth": 3.5,
                                     'xtick.labelsize': 20,
                                     'ytick.labelsize': 20,
                                     'lines.markersize': 8,
                                     'legend.fontsize': 17,
                                     'axes.labelsize': 20,
                                     'legend.handlelength': 1,
                                     'legend.handleheight': 1, })

    def _plot_decision_boundary(self, trainer):
        rc('text', usetex=True)
        net = trainer.algorithm.backbone.to('cpu')
        net.eval()
        self.__set_viz_context()
        
        h = .05
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        colors = ['#36008D', '#FE5E54', '#00C9B8']
        # cm_bright = ListedColormap([colors[1], colors[0]])
        _, loader = trainer.algorithm.benchmark.load_joint(trainer.current_task, batch_size=64)
        xx, yy = np.meshgrid(np.arange(-4, 4, h), np.arange(-4, 4, h))
        X, y = self.extract_points(loader)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright, edgecolors='k')
        # Z = net(torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()).data.max(1, keepdim=True)[1].detach().numpy()
        Z = F.softmax(net(torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()), dim=1).detach().numpy()[:, 1]
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=cm, alpha=.6)
        plt.text(-3.8, 3.4, f"Epoch {trainer.current_epoch}(Task {trainer.current_task})", fontsize=22)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, f'decisions-{trainer.current_epoch}.pdf'), dpi=200)
        
        # trainer.logger.log_figure(plt, 'decisions', step=trainer.current_epoch)
        plt.close('all')
    
    def on_after_training_epoch(self, trainer):
        self._plot_decision_boundary(trainer)


class NeuralActivationVisualizer(ContinualCallback):
    # TODO: save activations as numpy array
    def __init__(self, block_keys=('block_1', 'block_2')):
        self.block_keys = block_keys
        self.acts_history = {}
        super(NeuralActivationVisualizer, self).__init__('visualizer')
    
    def on_before_fit(self, trainer):
        self.save_path = os.path.join(trainer.params['output_dir'], 'plots')
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
     
    def __get_activations(self, trainer, task: int, mean_reduction=False):
        acts = {}
        net = trainer.algorithm.backbone
        device = trainer.algorithm.params['device']
        net = net.to(device)
        net.eval()
        train_loader, val_loader = trainer.algorithm.benchmark.load(task, batch_size=32)
        with torch.no_grad():
            for batch in val_loader:
                inp, targ, task_ids = batch
                net_acts = net.record_activations(inp.to(device))
                for key in self.block_keys:
                    key_acts = net_acts[key].cpu().numpy()
                    if acts.get(key):
                        acts[key] = np.concatenate(acts[key], key_acts, axis=0)
                    else:
                        acts[key] = key_acts
        if mean_reduction:
            mean_acts = {}
            for key in acts:
                mean_acts[key] = np.mean(acts[key], axis=0)
            return mean_acts
        else:
            return acts
    
    def on_after_training_task(self, trainer):
        current_task = trainer.current_task
        if not self.acts_history.get(current_task):
            self.acts_history[current_task] = {}
        
        for prev_task in range(1, current_task +1):
            acts = self.__get_activations(trainer, prev_task, mean_reduction=True)
            self.acts_history[current_task][prev_task] = acts

    def __set_viz_context(self):
        rc('text', usetex=True)
        sns.set_context("paper", rc={"lines.linewidth": 3.5,
                                     'xtick.labelsize': 15,
                                     'ytick.labelsize': 15,
                                     'lines.markersize': 8,
                                     'legend.fontsize': 15,
                                     'axes.labelsize': 15,
                                     'legend.handlelength': 1,
                                     'legend.handleheight': 1, })
    
    def get_colormap_range(self, trainer):
        if trainer.params.get("activation", "").lower() == 'relu':
            vmin, vmax = 0.0, 1.0
    
        elif trainer.params.get("activation", "").lower() == 'tanh':
            vmin, vmax = -1.0, 1.0
    
        elif trainer.params.get("activation", "").lower() == 'sigmoid':
            vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = None, None
        return vmin, vmax
    
    def _extract_task_activation_history(self, task, trainer):
        total_tasks = trainer.params.get("num_tasks")
        block_acts_history = {k: [] for k in self.block_keys}
        for target_task in range(task, total_tasks + 1):
            target_acts = self.acts_history[target_task][task]
            for block in target_acts.keys():
                block_acts_history[block].append(target_acts[block])
        return block_acts_history
    
    def plot_heatmap(self, task, acts_history, trainer):
        vmin, vmax = self.get_colormap_range(trainer)
        blcok_to_name = {'block_1': 'layer 1', 'block_2': 'layer 2'}
        fig, axs = plt.subplots(ncols=len(self.block_keys), sharey='row')
        
        for i, key in enumerate(acts_history.keys()):
            im = sns.heatmap(np.array(acts_history[key]), cmap="YlGnBu",
                             cbar_kws={"orientation": "horizontal"},
                             vmin=vmin, vmax=vmax,
                             square=True, ax=axs[i],
                             cbar=False)
            axs[i].set_xlabel(f'Neurons({blcok_to_name[key]})')
            axs[i].set_yticklabels(range(task, trainer.current_task))
            axs[i].set_ylabel('Tasks')

        # mappable = plt.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin, vmax), cmap="YlGnBu")
        # fig.colorbar(mappable, ax=axs, orientation='horizontal', fraction=0.03)
        
        return fig
    
    def save_task_history(self, task, task_act_history, trainer):
        save_path = os.path.join(trainer.params['output_dir'], 'acts')
        Path(save_path).mkdir(parents=True, exist_ok=True)
        filename = os.path.join(save_path, f"acts_history_{task}.npz")
        np.savez(filename, **task_act_history)

    def plot_activation_history(self, task, trainer):
        task_act_history = self._extract_task_activation_history(task, trainer)
        self.save_task_history(task, task_act_history, trainer)
        return self.plot_heatmap(task, task_act_history, trainer)
    
    def on_after_fit(self, trainer):
        self.__set_viz_context()
        fig, all_axs = plt.subplots(nrows=trainer.current_task - 1,
                                    ncols=len(self.block_keys),
                                    sharex='row', sharey='row',
                                    figsize=(10, 3))
        for task in range(1, trainer.current_task):
            fig = self.plot_activation_history(task, trainer)
            path = os.path.join(self.save_path, f"acts_task_{task}.pdf")
            plt.tight_layout()
            fig.savefig(path, bbox_inches="tight", dpi=200)
            plt.close(fig)

        # plt.tight_layout()
        # plt.savefig(os.path.join(self.save_path, f'acts_all.pdf'), dpi=200)


class ActTransitionTracker(ContinualCallback):
    def __init__(self, blocks=('block_1', 'block_2')):
        self.block_keys = blocks
        self.task_codes = {}
        self.task_loaders = {}
        super(ActTransitionTracker, self).__init__('TransitionTracker')
    
    def on_before_fit(self, trainer):
        for task in range(1, trainer.params['num_tasks'] + 1):
            device = trainer.params['device']
            batch_size = trainer.params['per_task_subset_examples']
            benchmark = trainer.algorithm.benchmark
            train_loader, val_loader = benchmark.load_subset(task, batch_size=batch_size, shuffle=False,
                                                             pin_memory=True if 'cuda' in str(device) else False)
            self.task_loaders[task] = val_loader
    
    def on_after_training_epoch(self, trainer):
        current_task = trainer.current_task
        if current_task <= 1:
            return
        for prev_task in range(1, current_task):
            prev_task_reference_codes = self.task_codes[prev_task]
            prev_task_current_codes = self.__calculate_codes(trainer, prev_task)
            for key in prev_task_current_codes.keys():
                ref = prev_task_reference_codes[key]
                cur = prev_task_current_codes[key]
                dist = self.__calculate_distance(ref, cur)
                print(f"prev_task>> {prev_task}, key={key}, dist={dist}")
    
    def on_after_training_task(self, trainer):
        self.task_codes[trainer.current_task] = self.__calculate_codes(trainer, trainer.current_task)
    
    def __calculate_distance(self, v1, v2):
        assert v1.shape[0] == v2.shape[0] and v1.shape[1] == v2.shape[1]
        batch_size = v1.shape[0]
        v1 = v1.view(1, -1)
        v2 = v2.view(1, -1)
        # print("shapes", v1.shape, v2.shape)
        return torch.dist(v1, v2, p=1).detach().cpu().item() / batch_size
    
    def __calculate_codes(self, trainer, target_task):
        codes = {}
        loader = self.task_loaders[target_task]
        net = trainer.algorithm.backbone
        device = trainer.params['device']
        for inp, targ, task_ids in loader:
            inp = inp.to(device)
            with torch.no_grad():
                acts = net.record_activations(inp, detach=True)
                for key in acts:
                    codes[key] = (acts[key] > 0.0).type(torch.float)  # .cpu().numpy()
        return codes

