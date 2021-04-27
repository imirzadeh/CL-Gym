import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_toy_benchmark(benchmark):
    def extract_points(loader):
        xs, ys = [], []
        for inp, targ in loader:
            batch_size = len(inp)
            for batch in range(batch_size):
                xs.append(inp[batch].numpy())
                ys.append(targ[batch])
        return np.array(xs), np.array(ys)

    for task in range(1, benchmark.num_tasks+1):
        train, eval = benchmark.load(task, batch_size=128)
