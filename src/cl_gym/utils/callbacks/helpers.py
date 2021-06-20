import os
import copy
import torch
import matplotlib
import numpy as np
import seaborn as sns
from pathlib import Path
import matplotlib.pylab as pl
from matplotlib import pyplot as plt
from typing import Literal, List, Tuple, Union, Iterable, Optional
from cl_gym.utils.loggers import Logger


class IntervalCalculator:
    def __init__(self, num_tasks: int,
                 epochs_per_task: int,
                 interval: Literal['epoch', 'task']):
        self.num_tasks = num_tasks
        self.epochs_per_task = epochs_per_task
        self.interval = interval
    
    def get_task_range(self, target_task: int):
        if self.interval == 'task':
            start, end = target_task, self.num_tasks + 1
        elif self.interval == 'epoch':
            start = (target_task - 1) * self.epochs_per_task + 1
            end = self.num_tasks * self.epochs_per_task + 1
        else:
            raise ValueError("Supported intervals are `epoch` and `task`")
        return start, end
    
    def get_tick_times(self):
        if self.interval == 'task':
            return range(1, self.num_tasks+1)
        elif self.interval == 'epoch':
            return list(range(1, (self.num_tasks+1)*self.epochs_per_task, self.epochs_per_task))[:-1] \
                   + [self.num_tasks*self.epochs_per_task]
        else:
            raise ValueError("Supported intervals are `epoch` and `task`")
    
    def get_step_within_task(self, global_epoch: int):
        if self.interval == 'task':  # task-based interval: only one epoch
            return 1
        else:  # epoch-based interval
            relative_epoch = global_epoch % self.epochs_per_task
            if relative_epoch == 0:  # last epoch of the task
                return self.epochs_per_task
            return relative_epoch


class Visualizer:
    def __init__(self):
        pass
    
    @staticmethod
    def get_pallete(palette_name='str'):
        palletes = {
            'plotly': ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
                       '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52'],
            'tensorboard': ['#ff7043', '#33bbee', '#ee3377', '#009988', '#0077bb',
                            '#cc3311', '#7043ff', "#FEB843", "#bbbbbb", '#120031'],
            't10': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
                    '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
            'g10': ['#3366CC', '#FF9900', '#109618', '#DC3912', '#990099', '#0099C6',
                    '#DD4477', '#66AA00', '#B82E2E', '#316395'],
            'colorblind': ["#0072B2", "#D55E00", "#E0D53B", "#009E73", "#C244E1",
                           '#56B4E9', '#a6761d', '#cc001d', '#00cc49', '#000000'],
            'bright': ['#0053d6', '#e1144b', '#e1aa14', '#00d783', '#aa14e1',
                       '#ff630a', '#00bed6', '#bed600', '#d600be', '#120031']}
        return palletes[palette_name.lower()]
    
    
    @staticmethod
    def generate_colors(num_colors: int):
        colors = pl.cm.jet(np.linspace(0, 1, num_colors))
        return colors

    @staticmethod
    def reset():
        plt.close('all')
    
    @staticmethod
    def set_context():
        rc = {"lines.linewidth": 3.5,
              'xtick.labelsize': 18,
              'ytick.labelsize': 18,
              'lines.markersize': 9,
              'axes.labelsize': 20,
              'axes.titlesize': 22,
              'legend.title_fontsize': 18,
              'legend.fontsize': 16,
              'legend.handlelength': 1,
              'legend.handleheight': 1,
              'text.usetex': True}
        
        sns.set_context("notebook", rc=rc)
    
    @staticmethod
    def _prepare_plot_context(plot_type: Literal['line', 'heatmap'], rc_dict=None):
        Visualizer.set_context()
        if rc_dict:
            plt.rcParams.update(rc_dict)
        
        if plot_type.lower() == 'line':
            sns.set_style('whitegrid')
        elif plot_type.lower() == 'heatmap':
            sns.set_style('white')
    
    @staticmethod
    def _apply_plot_details(plot_params):
        if plot_params:
            if plot_params.get('title'):
                plt.title(plot_params['title'])
            if plot_params.get('xlabel'):
                plt.xlabel(plot_params['xlabel'])
            if plot_params.get('ylabel'):
                plt.ylabel(plot_params['ylabel'])
            if plot_params.get("xlim"):
                plt.xlim(plot_params['xlim'])
            if plot_params.get("ylim"):
                plt.ylim(plot_params['ylim'])
            if plot_params.get("xticks"):
                if isinstance(plot_params['xticks'], list):
                    plt.xticks(plot_params['xticks'])
                elif isinstance(plot_params['xticks'], dict):
                    plt.xticks(plot_params['xticks']['ticks'], plot_params['xticks']['labels'])
            if plot_params.get("yticks"):
                if isinstance(plot_params['yticks'], list):
                    plt.yticks(plot_params['yticks'])
                elif isinstance(plot_params['yticks'], dict):
                    plt.xticks(plot_params['yticks']['ticks'], plot_params['yticks']['labels'])
            if plot_params.get("show_legend"):
                legend_loc = plot_params.get("legend_loc", "best")
                legend_ncol = plot_params.get("legend_ncol", 1)
                plt.legend(loc=legend_loc, ncol=legend_ncol)
        
            if plot_params.get("tight_layout"):
                plt.tight_layout()
    
    @staticmethod
    def line_plot(lines: List[Tuple[Iterable, np.ndarray]],
                  lines_std: List[Tuple[Iterable, np.ndarray]] = None,
                  labels: Optional[List[str]] = None,
                  colors: Optional[Union[List[str]]] = None,
                  cmap: Optional[str] = 'colorblind',
                  rc_params: Optional[dict] = None,
                  plot_params: Optional[dict] = None,
                  logger: Optional[Logger] = None,
                  logger_step: Optional[int] = None,
                  save_dir: Optional[str] = None,
                  filename: Optional[str] = None,
                  save_format: Optional[str] = 'pdf'):
        
        # setup
        Visualizer.reset()
        Visualizer._prepare_plot_context('line', rc_params)
        
        if not colors and cmap and len(lines) <= 10:
            colors = Visualizer.get_pallete(cmap)
        else:
            colors = Visualizer.generate_colors(len(lines))
        # plot lines
        for i, line in enumerate(lines):
            x, y = line
            color = colors[i]
            label = labels[i] if labels else None
            plt.plot(x, y, color=color, label=label)
            if lines_std:
                plt.fill_between(x, y-lines_std[i], y+lines_std, color=color, alpha=0.4)
        
        # plot details
        Visualizer._apply_plot_details(plot_params)
        
        if logger:
            logger.log_figure(plt, filename, logger_step)
        
        plt.savefig(os.path.join(save_dir, f"{filename}.{save_format}"),
                    dpi=200, format=save_format, bbox_inches='tight')
        plt.close('all')