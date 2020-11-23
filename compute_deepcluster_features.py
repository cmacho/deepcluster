

import argparse
import os
import pickle
import time

import faiss
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
from util import load_model
from clustering import preprocess_features
import models
from util import AverageMeter, Logger, UnifLabelSampler

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')

    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--arch', '-a', type=str, metavar='ARCH',
                        choices=['alexnet', 'vgg16'], default='alexnet',
                        help='CNN architecture (default: alexnet)')
    parser.add_argument('--sobel', action='store_true', help='Sobel filtering')
    parser.add_argument('--clustering', type=str, choices=['Kmeans', 'PIC'],
                        default='Kmeans', help='clustering algorithm (default: Kmeans)')
    parser.add_argument('--nmb_cluster', '--k', type=int, default=10000,
                        help='number of cluster for k-means (default: 10000)')
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
    dataset = datasets.ImageFolder(args.data, transform=transforms.Compose(tra))

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch,
                                             num_workers=args.workers,
                                             pin_memory=True)

    dataset_wout_transforms = datasets.ImageFolder(args.data, transform=transforms.ToTensor())

    dataloader_wout_transforms = torch.utils.data.DataLoader(dataset_wout_transforms,
                                                             batch_size = args.batch,
                                                             num_workers = args.workers,
                                                             pin_memory = True)

    # get the features for the whole dataset
    features, images = compute_features(dataloader, dataloader_wout_transforms, model, len(dataset))

    after_pca_and_normalize = preprocess_features(features, 256)

    file_out_path = os.path.join(args.exp,"embedding.npy")
    file_out_path_images = os.path.join(args.exp, "images.npy")
    np.save(file_out_path, after_pca_and_normalize)
    np.save(file_out_path_images, images)


def compute_features(dataloader, dataloader_wout_transforms, model, N):
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
            images = np.zeros((N, 3,84,84))

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
            images = np.zeros((N, 84,84,3))

        current = images_tensor.numpy()
        current = np.transpose(current, (0,2,3,1))

        if i < len(dataloader_wout_transforms) - 1:
            images[i * args.batch: (i + 1) * args.batch] = current
        else:
            # special treatment for final batch
            images[i * args.batch:] = current

    return features, images

if __name__ == '__main__':
    args = parse_args()
    main(args)