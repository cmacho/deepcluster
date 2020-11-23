# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import os
import pickle
import time

#import faiss
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from clustering_constrained_dc import preprocess_features
from util import load_model
import clustering_constrained_dc as clustering
import models
from util import AverageMeter, Logger, UnifLabelSampler


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')

    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--arch', '-a', type=str, metavar='ARCH',
                        choices=['alexnet', 'vgg16'], default='alexnet',
                        help='CNN architecture (default: alexnet)')
    parser.add_argument('--sobel', action='store_true', help='Sobel filtering')
    parser.add_argument('--nmb_cluster', '--k', type=int, default=10000,
                        help='number of cluster for k-means (default: 10000)')
    parser.add_argument('--num_centers_from_labeled_tasks', type=int, default=15,
                        help='number of cluster to initialize from labeled tasks during kmeans')
    parser.add_argument('--lr', default=0.05, type=float,
                        help='learning rate (default: 0.05)')
    parser.add_argument('--wd', default=-5, type=float,
                        help='weight decay pow (default: -5)')
    parser.add_argument('--reassign', type=float, default=1.,
                        help="""how many epochs of training between two consecutive
                        reassignments of clusters (default: 1)""")
    parser.add_argument('--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of total epochs to run (default: 200)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--batch', default=256, type=int,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to checkpoint (default: None)')
    parser.add_argument('--checkpoints', type=int, default=25000,
                        help='how many iterations between two checkpoints (default: 25000)')
    parser.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
    parser.add_argument('--exp', type=str, default='', help='path to exp folder')
    parser.add_argument('--verbose', action='store_true', help='chatty')
    return parser.parse_args()


def main(args):
    if args.verbose:
        print('Architecture: {}'.format(args.arch))
    model_path = os.path.join(args.exp, "checkpoint.pth.tar")
    # model = load_model(model_path)
    # model.features = torch.nn.DataParallel(model.features)
    # model.cuda()
    # cudnn.benchmark = True
    # # remove head
    # model.top_layer = None
    # model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])

    model = models.__dict__[args.arch](sobel=args.sobel)
    fd = int(model.top_layer.weight.size()[1])
    model.top_layer = None
    model.features = torch.nn.DataParallel(model.features)
    model.cuda()
    cudnn.benchmark = True

    if os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path)
        args.start_epoch = checkpoint['epoch']
        # remove top_layer parameters from checkpoint
        list1 = []
        for key in checkpoint['state_dict']:
            if 'top_layer' in key:
                list1.append(key)
        for key in list1:
            del checkpoint['state_dict'][key]
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(model_path, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(model_path))



    # preprocessing of data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    tra = [transforms.Resize(78),
           transforms.CenterCrop(64),
           transforms.ToTensor(),
           normalize]

    # load the data
    end = time.time()
    dataset = datasets.ImageFolder(args.data, transform=transforms.Compose(tra))
    if args.verbose:
        print('Load dataset: {0:.2f} s'.format(time.time() - end))

    # load labeled tasks of shape (num_tasks, n_way, num_samples_per_class, height, width, channels)
    labeled_tasks_original = np.load("labeled_tasks.npy")
    num_tasks, n_way, num_samples_per_class, height, width, channels = labeled_tasks_original.shape

    # transpose to channel first ordering
    labeled_tasks = np.transpose(labeled_tasks_original, (0,1,2,5,3,4))

    tra2 = [transforms.Resize(78),
           transforms.CenterCrop(64),
           normalize]

    labeled_tasks_flat = labeled_tasks.reshape((num_tasks * n_way * num_samples_per_class, channels, height, width))
    dataset_labeled_tasks = CustomTensorDataset(torch.Tensor(labeled_tasks_flat), transform=transforms.Compose(tra2))

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch,
                                             num_workers=args.workers,
                                             pin_memory=True)

    dataloader_labeled_tasks = torch.utils.data.DataLoader(dataset_labeled_tasks,
                                             batch_size=args.batch,
                                             num_workers=args.workers,
                                             pin_memory=True)

    dataset_wout_transforms = datasets.ImageFolder(args.data, transform=transforms.ToTensor())
    dataloader_wout_transforms = torch.utils.data.DataLoader(dataset_wout_transforms,
                                                             batch_size=args.batch,
                                                             num_workers=args.workers,
                                                             pin_memory=True)

    # clustering algorithm to use
    deepcluster = clustering.Kmeans(args.nmb_cluster, args.num_centers_from_labeled_tasks)


    # remove head
    model.top_layer = None
    model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])

    # get the features for the whole dataset
    data, data_labeled_tasks, images_unlabeled_tasks = compute_features(dataloader, dataloader_labeled_tasks, dataloader_wout_transforms,model,
                                               len(dataset), num_tasks * n_way * num_samples_per_class)


    data_labeled_tasks = data_labeled_tasks.reshape((num_tasks * n_way * num_samples_per_class, -1))

    combined_data = np.concatenate((data_labeled_tasks, data), axis=0)

    # PCA-reducing, whitening and L2-normalization
    xb = preprocess_features(combined_data)

    reduced_data_labeled_tasks = xb[:num_tasks * n_way * num_samples_per_class]
    reduced_data = xb[num_tasks * n_way * num_samples_per_class:]

    assert reduced_data.shape[0] == data.shape[0]

    reduced_data_labeled_tasks = reduced_data_labeled_tasks.reshape((num_tasks, n_way, num_samples_per_class, -1))

    file_out_path_unlabeled = os.path.join(args.exp, "embedding_unlabeled.npy")
    file_out_path_images_unlabeled = os.path.join(args.exp, "images_unlabeled.npy")
    file_out_path_labeled = os.path.join(args.exp, "embedding_labeled.npy")

    np.save(file_out_path_unlabeled, reduced_data)
    np.save(file_out_path_images_unlabeled, images_unlabeled_tasks)
    np.save(file_out_path_labeled, reduced_data_labeled_tasks)


def compute_features(dataloader, dataloader_labeled_tasks, dataloader_wout_transforms, model, N, num_labeled_data):
    if args.verbose:
        print('Compute features')
    batch_time = AverageMeter()
    end = time.time()
    model.eval()
    # discard the label information in the dataloader
    for i, (input_tensor, _) in enumerate(dataloader):
        input_var = torch.autograd.Variable(input_tensor.cuda(), volatile=True)
        aux = model(input_var).data.cpu().numpy()

        if i == 0:
            features = np.zeros((N, aux.shape[1]), dtype='float32')
            images_unlabeled_tasks = np.zeros((N, 3,84,84))

        aux = aux.astype('float32')
        if i < len(dataloader) - 1:
            features[i * args.batch: (i + 1) * args.batch] = aux
        else:
            # special treatment for final batch
            features[i * args.batch:] = aux

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.verbose and (i % 200) == 0:
            print('{0} / {1}\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
                  .format(i, len(dataloader), batch_time=batch_time))

    # create numpy array of images in the same order (as predicated by torchvision.datasets.ImageFolder
    for i, (images_tensor, _) in enumerate(dataloader_wout_transforms):
        if i == 0:
            images_unlabeled_tasks = np.zeros((N, 84,84,3))

        current = images_tensor.numpy()
        current = np.transpose(current, (0,2,3,1))

        if i < len(dataloader_wout_transforms) - 1:
            images_unlabeled_tasks[i * args.batch: (i + 1) * args.batch] = current
        else:
            # special treatment for final batch
            images_unlabeled_tasks[i * args.batch:] = current

    for i, input_tensor in enumerate(dataloader_labeled_tasks):
        input_var = torch.autograd.Variable(input_tensor.cuda(), volatile=True)
        aux = model(input_var).data.cpu().numpy()
        
        if i == 0:
            features_labeled_tasks = np.zeros((num_labeled_data, aux.shape[1]), dtype='float32')

        aux = aux.astype('float32')
        if i < len(dataloader) - 1:
            features_labeled_tasks[i * args.batch: (i + 1) * args.batch] = aux
        else:
            # special treatment for final batch
            features_labeled_tasks[i * args.batch:] = aux

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.verbose and (i % 200) == 0:
            print('{0} / {1}\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
                  .format(i, len(dataloader_labeled_tasks), batch_time=batch_time))

    return features, features_labeled_tasks, images_unlabeled_tasks


class CustomTensorDataset(torch.utils.data.Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensor, transform=None):
        self.tensor = tensor
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensor[index]
        if self.transform:
            x = self.transform(x)
        return x

    def __len__(self):
        return self.tensor.shape[0]

if __name__ == '__main__':
    args = parse_args()
    main(args)
