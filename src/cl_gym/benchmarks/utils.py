import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
dirname = os.path.dirname(__file__)
DEFAULT_DATASET_DIR = os.path.join(dirname, '..', 'data')



def _get_first_batch(loader):
    for batch, (img, _, task_id) in enumerate(loader):
        return img.numpy()

 
def _extract_images(benchmark, num_tasks, examples_per_task):
    task_examples = {}
    for task in range(1, num_tasks + 1):
        train_loader, _ = benchmark.load(task, batch_size=examples_per_task, num_workers=0, shuffle=False, pin_memory=False)
        # train_loader, _ = benchmark.load_joint(task, batch_size=examples_per_task, num_workers=0, pin_memory=False)
        # train_loader, _ = benchmark.load_memory(task, batch_size=examples_per_task, num_workers=0, pin_memory=False)
        # train_loader, _ = benchmark.load_memory_joint(task, batch_size=examples_per_task, num_workers=0, pin_memory=False)
        task_examples[task] = _get_first_batch(train_loader)
    
    return task_examples


def visualize_benchmark(benchmark, num_tasks=None, examples_per_task=16):
    if num_tasks is None:
        num_tasks = benchmark.num_tasks
        print(f"num examples wasn't provided and set to {num_tasks} from benchmark")
    
    task_examples = _extract_images(benchmark, num_tasks, examples_per_task)
    fig, axs = plt.subplots(num_tasks, examples_per_task, sharex=True)
    
    for task in range(1, num_tasks+1):
        images = task_examples[task]
        for i in range(min(examples_per_task, len(images))):
            axs[task-1][i].imshow(images[i].reshape(28, 28), cmap="Greys")
            axs[task-1][i].set_xticks([])
            axs[task-1][i].set_yticks([])

        axs[task-1][0].set(ylabel=f"Task {task}")
        
    plt.savefig(DIR+"images2.png", dpi=120)
    

if __name__ == "__main__":
    from cl_gym.benchmarks import RotatedMNIST, SplitMNIST, MNISTFashionMNIST
    
    # rot_mnist_benchmark = RotatedMNISTBenchmark(5, per_task_joint_examples=32, per_task_memory_examples=10, per_task_rotation_degrees=22.5)
    # visualize_benchmark(rot_mnist_benchmark, 5, 32)
    # split_mnist_benchmark = SplitMNISTBenchmark(5) #per_task_joint_examples=20, per_task_memory_examples=2)
    # benchmark = PermutedMNISTBenchmark(3)
    # print("Generated ! going to visualize")
    # benchmark = Spl(3, per_task_rotation=30)
    benchmark = MNISTFashionMNIST(2)
    visualize_benchmark(benchmark, 2, 16)
