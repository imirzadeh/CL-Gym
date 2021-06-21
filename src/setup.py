import os
from typing import List
from setuptools import setup, find_packages

_PATH_ROOT = os.path.dirname(__file__)


def _load_requirements(path_dir: str, file_name: str = 'requirements.txt', comment_char: str = '#') -> List[str]:
    # Load requirements from a file
    # source: https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/setup_tools.py
    with open(os.path.join(path_dir, file_name), 'r') as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:
        # filer all comments
        if comment_char in ln:
            ln = ln[:ln.index(comment_char)].strip()
        # skip directly installed dependencies
        if ln.startswith('http') or '@http' in ln:
            continue
        if ln:  # if requirement is not empty
            reqs.append(ln)
    return reqs

setup(
  name='cl-gym',
  # packages=['cl_gym'],
  packages=find_packages(exclude=('tests', 'tests*')),
  version='1.0.1',
  license='MIT',
  description='Continual Learning Gym',
  author='Iman Mirzadeh',
  author_email='seyediman.mirzadeh@wsu.edu',
  url='https://github.com/imirzadeh/Cl-Gym',
  keywords=['continual learning', 'lifelong learning', 'pytorch', 'deep learning'],
  install_requires=_load_requirements(_PATH_ROOT),
  classifiers=[
    'Development Status :: 4 - Beta',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
  ],
)
