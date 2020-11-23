# DeepCluster and Constrained DeepCluster

This is adapted from the implementation of [DeepCluster](https://github.com/facebookresearch/deepcluster) by Facebook Research. 

See [my other repo](https://github.com/cmacho/Semi-Supervised-Meta-Learning) on Semi Supervised Meta Learning for context and to find out what this branch is about.

## Usage:

Download the mini imagenet dataset e.g. from the links that you can find here:
https://github.com/yaoyao-liu/mini-imagenet-tools

Split the 64 classes from the training set of mini imagenet into one group of 12 classes and one group of 52 classes. The directory structure for mini imagenet should then be as shown [mini_imagenet_directory_structure.txt](mini_imagenet_directory_structure.txt)

Update the paths for the mini imagenet directories in [main.sh](main.sh) and [main_constrained.sh](main_constrained_dc.sh) as appropriate. 

Run `sample_labeled_tasks.sh` in order to create the file `labeled_tasks.npy`.
Then run `main.sh` in order to run DeepCluster and `main_constrained.sh` in order to run Constrained DeepCluster.

Note the requirements below. This should be run with python 3 because the latest version of sklearn is much faster than versions that are available for python 2.

## Requirements

- Python version 3.7
- scikit-learn version 0.23.2
- PyTorch version 1.7.0
- cudatoolkit 10.2
