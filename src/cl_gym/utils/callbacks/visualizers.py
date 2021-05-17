import os
import math
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

matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
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
        if trainer.logger:
            trainer.logger.log_figure(plt, f'decisions', step=trainer.current_epoch)
        
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
            im = sns.heatmap(np.array(acts_history[key]), cmap="coolwarm",
                             cbar_kws={"orientation": "horizontal"},
                             center=0.5,
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
        self.distances_cache = {}
        super(ActTransitionTracker, self).__init__('TransitionTracker')
    
    def on_before_fit(self, trainer):
        self.save_path = os.path.join(trainer.params['output_dir'], 'plots')
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        
        for task in range(1, trainer.params['num_tasks'] + 1):
            device = trainer.params['device']
            batch_size = trainer.params['per_task_subset_examples']
            benchmark = trainer.algorithm.benchmark
            train_loader, val_loader = benchmark.load_subset(task, batch_size=batch_size, shuffle=False,
                                                             pin_memory=True if 'cuda' in str(device) else False)
            self.task_loaders[task] = val_loader
    
    def add_to_distance_cache(self, key, value, step):
        if not self.distances_cache.get(key):
            self.distances_cache[key] = []
        self.distances_cache[key].append((step, value))
        
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
                metric_name = f'code_dist_task_{prev_task}_{key}'
                self.add_to_distance_cache(metric_name, dist, trainer.current_epoch)
                if trainer.logger:
                    trainer.logger.log_metric(metric_name, dist, step=trainer.current_epoch)
    
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
        
        
    def __calculate_xticks(self, trainer, task):
        epochs_per_task = trainer.params['epochs_per_task']
        total_tasks = trainer.params['num_tasks']
        start = (task-1)*epochs_per_task + 1
        ticks = [(t-task)*epochs_per_task+start for t in range(task+1, total_tasks+1)] + [total_tasks*epochs_per_task]
        return ticks
        
    def _plot_code_distances(self, trainer):
        sns.set_style('whitegrid')
        plt.close('all')
        colors = ['#636EFA', '#00CC96', '#EF553B', '#AB63FA', '#FFA15A']
        block_to_name = {'block_1': 'layer 1', 'block_2': 'layer 2',
                         'block_3': 'layer 3', 'block_4': 'layer 4', 'total': 'total'}
        
        for task in range(1, trainer.current_task-1):
            for i, key in enumerate(self.task_codes[task].keys()):
                metric_name = f'code_dist_task_{task}_{key}'
                steps = [x[0] for x in self.distances_cache[metric_name]]
                vals = [x[1] for x in self.distances_cache[metric_name]]
                plt.plot(steps, vals, color=colors[i], label=f"{block_to_name[key]}")
            plt.xlabel("Epochs")
            plt.ylabel(f"Transitions (Task {task})")
            plt.xticks(self.__calculate_xticks(trainer, task))
            plt.legend(loc='upper left')
            # plt.ylim((0, 12))
            # plt.yticks([2, 4, 6, 8, 10])
            plt.tight_layout()
            if trainer.logger:
                trainer.logger.log_figure(plt, f"transitions_task_{task}")
            plt.savefig(os.path.join(self.save_path, f"transitions_task{task}.pdf"), dpi=200)
            plt.close()

    def on_after_fit(self, trainer):
        self._plot_code_distances(trainer)


class DecisionBoundaryTracker(ContinualCallback):
    def __init__(self, blocks=('block_1', 'block_2')):
        self.block_keys = blocks
        self.task_loaders = {}
        self.distance_history = {}
        self.epochs_per_task = None
        super(DecisionBoundaryTracker, self).__init__('DecisionBoundaryTracker')
    
    def on_before_fit(self, trainer):
        self.save_path = os.path.join(trainer.params['output_dir'], 'plots')
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        self.num_tasks = trainer.params['num_tasks']
        self.epochs_per_task = trainer.params['epochs_per_task']
        
        for task in range(1, trainer.params['num_tasks'] + 1):
            self.distance_history[task] = []
            device = trainer.params['device']
            batch_size = trainer.params['per_task_subset_examples']
            benchmark = trainer.algorithm.benchmark
            train_loader, val_loader = benchmark.load_subset(task, batch_size=batch_size, shuffle=False,
                                                             pin_memory=True if 'cuda' in str(device) else False)
            self.task_loaders[task] = val_loader
    
    def _calculate_distances(self, trainer):
        net = trainer.algorithm.backbone
        device = trainer.params['device']
        for task in range(1, trainer.current_task+1):
            loader = self.task_loaders[task]
            for inp, targ, task_ids in loader:
                inp = inp.to(device)
                distances = net.record_distance_to_boundary(inp, reduction='mean')
                heatmap_matrix = self._extract_heatmap(distances)
                self._heatmap_plot(task, trainer.current_epoch, heatmap_matrix, trainer)
                self.distance_history[task].append(distances)
                if task == trainer.current_task:
                    self._plot_block_1(distances, trainer)
                for block in self.block_keys:
                    name = f"task{task}_{block}"
                    # print(f"task{task}, {block}, mean={np.mean(distances[block])}")
                    trainer.logger.experiment.log_histogram_3d(distances[f"{block}_reduction"], name=name, step=trainer.current_epoch)
    
    def __calculate_task_steps(self, task):
        start = (task-1)*self.epochs_per_task+1
        end = self.num_tasks*self.epochs_per_task+1
        return start, end

    def _extract_heatmap(self, distances):
        heatmap_matrix = {k: None for k in self.block_keys}
        for j, block in enumerate(self.block_keys):
            signs = distances[f"{block}_signs"]
            dist = distances[block]
            heatmap_matrix[block] = np.multiply(signs, dist)
        return heatmap_matrix

    def _heatmap_plot(self, task, epoch, heatmap_matrix, trainer):
        plt.close('all')
        self.__set_viz_context()
        fig, axs = plt.subplots(1, 2)
        block_names = self.__get_block_names()
        for i, block in enumerate(self.block_keys):
                im = sns.heatmap(heatmap_matrix[block], cmap="coolwarm",
                                 center=0.0,
                                 cbar_kws={"orientation": "horizontal"},
                                 square=True, ax=axs[i],
                                 cbar=True)
                axs[i].set_title(block_names[block])
        
        plt.tight_layout()
        if trainer.logger:
            trainer.logger.log_figure(plt, f"distances_task_{task}", step=trainer.current_epoch)
        plt.savefig(os.path.join(self.save_path, f"distances_task_{task}_{epoch}.pdf"))
        plt.close()

    def _plot_block_1(self, data, trainer):
        plt.close()
        sns.set_context("paper", rc={"lines.linewidth": 3.5,
                                     'xtick.labelsize': 20,
                                     'ytick.labelsize': 20,
                                     'lines.markersize': 8,
                                     'legend.fontsize': 17,
                                     'axes.titlesize': 25,
                                     'legend.handlelength': 1,
                                     'legend.handleheight': 1,
                                     'text.usetex': True})
        rc('text', usetex=True)
        num_hiddens = trainer.params['hidden_1_dim']
        fig, axs = plt.subplots(int(math.ceil(num_hiddens//4)), 4, figsize=(24, 16))
        axs = axs.reshape(-1)
        import matplotlib.pylab as pl
        colors = pl.cm.jet(np.linspace(0, 1, num_hiddens))
        for i in range(num_hiddens):
            w, b = data['block_1_w'][i], data['block_1_b'][i]
            domain = np.linspace(-4, 4, 10)
            f = lambda x: (-b - w[0]*x)/w[1]
            axs[i].plot(domain, [f(x) for x in domain], color=colors[i])
            axs[i].set_title(f"{i}")
        plt.tight_layout()
        if trainer.logger:
            trainer.logger.log_figure(plt, "weight_lines", step=trainer.current_epoch)
        plt.savefig(os.path.join(self.save_path, f'weight_lines_{trainer.current_epoch}.pdf'))
        plt.close()
        
        sns.set_style('white')
        for i in range(num_hiddens):
                w, b = data['block_1_w'][i], data['block_1_b'][i]
                domain = np.linspace(-4, 4, 10)
                f = lambda x: (-b - w[0] * x) / w[1]
                plt.plot(domain, [f(x) for x in domain], color=colors[i], label=str(i))
        plt.legend(ncol=2, handlelength=1.0, loc='upper left')
        plt.ylim((-8.1, 8.1))
        plt.xlim((-8.1, 8.1))
        plt.tight_layout()
        if trainer.logger:
            trainer.logger.log_figure(plt, 'all_lines', step=trainer.current_epoch)
        plt.savefig(os.path.join(self.save_path, f'lines_all_{trainer.current_epoch}.pdf'))
        plt.close()

    def __get_block_names(self):
        return {'block_1': 'layer 1', 'block_2': 'layer 2',
                'block_3': 'layer 3', 'block_4': 'layer 4'}
    
    def __set_viz_context(self):
        sns.set_context("paper", rc={"lines.linewidth": 3.5,
                                     'xtick.labelsize': 10,
                                     'ytick.labelsize': 10,
                                     'lines.markersize': 8,
                                     'legend.fontsize': 17,
                                     'axes.titlesize': 22,
                                     'legend.handlelength': 1,
                                     'legend.handleheight': 1,
                                     'text.usetex': True})
        rc('text', usetex=True)

    def _line_plot(self, task, distances, trainer):
        plt.close('all')
        sns.set_style('whitegrid')
        colors = ['#636EFA', '#00CC96', '#EF553B', '#AB63FA', '#FFA15A']
        lines = {k: [] for k in self.block_keys}
        block_names = self.__get_block_names()
        total = []
        
        start, end = self.__calculate_task_steps(task)
        for i in range(len(distances)):
            for block in self.block_keys:
                block_dist = np.mean(distances[i][block])
                lines[block].append(block_dist)
        
        for i, block in enumerate(lines.keys()):
            plt.plot(range(start, end), lines[block], color=colors[i], label=block_names[block])
            total.append(lines[block])
        total = np.sum(total, axis=0)
        plt.plot(range(start, end), total, color=colors[len(self.block_keys)], label='total')

        plt.ylabel(f'Average Distance - Task {task}')
        plt.xlabel('Epochs')
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.xticks(self.__calculate_xticks(trainer, task))
        fig_name = f"mean_boundary_distances_{task}"
        trainer.logger.log_figure(plt, fig_name, step=trainer.current_epoch)
        plt.savefig(os.path.join(self.save_path, f'{fig_name}.pdf'), dpi=200)

    def __calculate_xticks(self, trainer, task):
        epochs_per_task = trainer.params['epochs_per_task']
        total_tasks = trainer.params['num_tasks']
        start = (task-1)*epochs_per_task + 1
        ticks = [(t-task)*epochs_per_task+start for t in range(task+1, total_tasks+1)] + [total_tasks*epochs_per_task]
        return ticks
    
    def _plot_distances(self, trainer):
        for task in range(1, trainer.params['num_tasks']):
            distances = self.distance_history[task]
            self._line_plot(task, distances, trainer)
            
    def on_after_training_epoch(self, trainer):
        self._calculate_distances(trainer)
    
    def on_after_fit(self, trainer):
        self._plot_distances(trainer)


class WeightTracker(ContinualCallback):
    def __init__(self):
        super(WeightTracker, self).__init__('WeightTracker')

    def on_before_fit(self, trainer):
        self.save_path = os.path.join(trainer.params['output_dir'], 'plots')
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
    
    def _heatmap_plot(self, params, block_id, trainer):
        plt.close('all')
        heatmap = np.concatenate((params['weight'], params['bias']), axis=1)
        sns.set_context("paper", rc={"lines.linewidth": 3.5,
                                     'xtick.labelsize': 15,
                                     'ytick.labelsize': 15,
                                     'lines.markersize': 8,
                                     'legend.fontsize': 17,
                                     'axes.labelsize': 15,
                                     'axes.titlesize': 22,
                                     'legend.handlelength': 1,
                                     'legend.handleheight': 1, })
        g = sns.heatmap(heatmap, cmap="coolwarm",
                         center=0.0,
                         cbar_kws={"orientation": "vertical"},
                         annot=True,
                         fmt=".1f",
                         square=False,
                         cbar=False)
        plt.title(f"Layer {block_id}")
        plt.ylabel("Neurons")
        plt.xlabel("Weights + Bias")
        g.set_yticklabels(g.get_yticklabels(), rotation=0)
        num_hiddens = heatmap.shape[1]-1
        g.set_xticklabels(["$w_{{{0}}}$".format(i) for i in range(1, num_hiddens+1)]+["$b$"])
        
        plt.yticks(rotation=0)
        plt.tight_layout()
        if trainer.logger:
            trainer.logger.log_figure(plt, f"weights_layer_{block_id}", step=trainer.current_epoch)
        plt.savefig(os.path.join(self.save_path, f"weights_layer_{block_id}_{trainer.current_epoch}.pdf"))
        plt.close('all')

    def _plot_weights(self, net, trainer):
        for i in [1, 2]:
            params = net.get_block_params(i)
            self._heatmap_plot(params, i, trainer)
            
    def on_after_training_epoch(self, trainer):
        net = trainer.algorithm.backbone
        net.eval()
        self._plot_weights(net, trainer)


class WeightedDistanceTracker(ContinualCallback):
    def __init__(self, block_keys=('block_1', 'block_2')):
        super(WeightedDistanceTracker, self).__init__()
        self.block_keys = block_keys
        self.weighted_distance_history = {}
        self.task_loaders = {}

    def on_before_fit(self, trainer):
        self.save_path = os.path.join(trainer.params['output_dir'], 'plots')
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        self.num_tasks = trainer.params['num_tasks']
        self.epochs_per_task = trainer.params['epochs_per_task']

        for task in range(1, trainer.params['num_tasks'] + 1):
            self.weighted_distance_history[task] = []
            device = trainer.params['device']
            batch_size = trainer.params['per_task_subset_examples']
            benchmark = trainer.algorithm.benchmark
            train_loader, val_loader = benchmark.load_subset(task, batch_size=batch_size, shuffle=False,
                                                             pin_memory=True if 'cuda' in str(device) else False)
            self.task_loaders[task] = val_loader

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
                    key_acts = (net_acts[key] > 0.0).type(torch.float).cpu().numpy()
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

    def __calculate_task_steps(self, task):
        start = (task-1)*self.epochs_per_task+1
        end = self.num_tasks*self.epochs_per_task+1
        return start, end
    
    def __get_boundary_distances(self, trainer, task):
        net = trainer.algorithm.backbone
        device = trainer.params['device']
        loader = self.task_loaders[task]
        distances = None
        for inp, targ, task_ids in loader:
            inp = inp.to(device)
            distances = net.record_distance_to_boundary(inp, reduction='mean')
        return distances
            
    def __calculate_neuron_importance(self, data):
        num_classes = 2
        score = lambda x: 1.0 - np.abs(num_classes*x - 1.0)
        result = np.vectorize(score)(data)
        return result / np.sum(result)

    def __calculate_xticks(self, trainer, task):
        epochs_per_task = trainer.params['epochs_per_task']
        total_tasks = trainer.params['num_tasks']
        start = (task-1)*epochs_per_task + 1
        ticks = [(t-task)*epochs_per_task+start for t in range(task+1, total_tasks+1)] + [total_tasks*epochs_per_task]
        return ticks
    
    def _line_plot(self, task, distances, trainer):
        plt.close('all')
        sns.set_style('whitegrid')
        colors = ['#636EFA', '#00CC96', '#EF553B', '#AB63FA', '#FFA15A']
        lines = {k: [] for k in self.block_keys}
        block_names = {'block_1': 'layer 1', 'block_2': 'layer 2'}
        total = []
    
        start, end = self.__calculate_task_steps(task)
        for i in range(len(distances)):
            for block in self.block_keys:
                block_dist = np.mean(distances[i][block])
                lines[block].append(block_dist)
    
        for i, block in enumerate(lines.keys()):
            plt.plot(range(start, end), lines[block], color=colors[i], label=block_names[block])
            total.append(lines[block])
        total = np.sum(total, axis=0)
        plt.plot(range(start, end), total, color=colors[len(self.block_keys)], label='total')
    
        plt.ylabel(f'Average Weighted Distance - Task {task}')
        plt.xlabel('Epochs')
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.xticks(self.__calculate_xticks(trainer, task))
        fig_name = f"weighted_boundary_distances_{task}"
        trainer.logger.log_figure(plt, fig_name, step=trainer.current_epoch)
        plt.savefig(os.path.join(self.save_path, f'{fig_name}.pdf'), dpi=150)

    def on_after_training_epoch(self, trainer):
        for task in range(1, trainer.current_task+1):
            acts = self.__get_activations(trainer, task, mean_reduction=True)
            importance = {block: self.__calculate_neuron_importance(acts[block]) for block in acts.keys()}
            distances = self.__get_boundary_distances(trainer, task)
            weighted_distance = {}
            for block in self.block_keys:
                score = np.mean(np.matmul(distances[block], importance[block].T.reshape(-1, 1)), axis=0)
                weighted_distance[block] = score
            self.weighted_distance_history[task].append(weighted_distance)

    def on_after_fit(self, trainer):
        for task in [1, 2]:
            self._line_plot(task, self.weighted_distance_history[task], trainer)
            
            
        

