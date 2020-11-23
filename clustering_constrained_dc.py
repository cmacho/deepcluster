# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import time
import numpy as np
from PIL import Image
from PIL import ImageFile
from scipy.sparse import csr_matrix, find
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from constrained_kmeans import cluster_with_constraints

ImageFile.LOAD_TRUNCATED_IMAGES = True

__all__ = ['PIC', 'Kmeans', 'cluster_assign', 'arrange_clustering']


def pil_loader(path):
    """Loads an image.
    Args:
        path (string): path to image file
    Returns:
        Image
    """
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ReassignedDataset(data.Dataset):
    """A dataset where the new images labels are given in argument.
    Args:
        image_indexes (list): list of data indexes
        pseudolabels (list): list of labels for each data
        dataset (list): list of tuples with paths to images
        transform (callable, optional): a function/transform that takes in
                                        an PIL image and returns a
                                        transformed version
    """

    def __init__(self, image_indexes, pseudolabels, dataset, labeled_tasks_flat, transform_pil=None, transform_tensor=None):
        self.imgs, self.labeled_data_pseudolabels = self.make_dataset(image_indexes, pseudolabels, dataset)
        assert len(self.labeled_data_pseudolabels) == labeled_tasks_flat.shape[0]
        assert len(self.imgs) == len(dataset)
        self.labeled_tasks_flat = labeled_tasks_flat
        self.transform_pil = transform_pil
        self.transform_tensor = transform_tensor
        self.len_unlabeled = len(dataset)

    def make_dataset(self, image_indexes, pseudolabels, dataset):
        label_to_idx = {label: idx for idx, label in enumerate(set(pseudolabels))}
        images = []
        labeled_data_pseudolabels = []
        for j, idx in enumerate(image_indexes):
            if idx < len(dataset):
                path = dataset[idx][0]
                pseudolabel = label_to_idx[pseudolabels[j]]
                images.append((path, pseudolabel))
            else:
                pseudolabel = label_to_idx[pseudolabels[j]]
                labeled_data_pseudolabels.append(pseudolabel)

        return images, labeled_data_pseudolabels

    def __getitem__(self, index):
        """
        Args:
            index (int): index of data
        Returns:
            tuple: (image, pseudolabel) where pseudolabel is the cluster of index datapoint
        """
        if index < self.len_unlabeled:
            path, pseudolabel = self.imgs[index]
            img = pil_loader(path)
            if self.transform_pil is not None:
                img = self.transform_pil(img)
            return img, pseudolabel
        else:
            img = self.transform_tensor(self.labeled_tasks_flat[index - self.len_unlabeled])
            pseudolabel = self.labeled_data_pseudolabels[index - self.len_unlabeled]
            return img, pseudolabel

    def __len__(self):
        return len(self.imgs) + len(self.labeled_data_pseudolabels)


def preprocess_features(npdata, dim_to_reduce_to=256):
    """Preprocess an array of features.
    Args:
        npdata (np.array N * ndim): features to preprocess
        pca (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """

    start = time.time()

    _, ndim = npdata.shape
    npdata =  npdata.astype('float32')

    # Apply PCA-whitening with sklearn
    pca = PCA(n_components=dim_to_reduce_to)
    pca.fit(npdata)
    npdata = pca.transform(npdata)

    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)
    npdata = npdata / row_sums[:, np.newaxis]
    print('pca time: {0:.0f} s'.format(time.time() - start))

    return npdata



def cluster_assign(images_lists, dataset, labeled_tasks_flat):
    """Creates a dataset from clustering, with clusters as labels.
    Args:
        images_lists (list of list): for each cluster, the list of image indexes
                                    belonging to this cluster
        dataset (list): initial dataset
    Returns:
        ReassignedDataset(torch.utils.data.Dataset): a dataset with clusters as
                                                     labels
    """
    assert images_lists is not None

    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))

    assert len(image_indexes) == len(dataset) + labeled_tasks_flat.shape[0]

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    t_pil = transforms.Compose([transforms.RandomResizedCrop(64),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize])
    t_tensor = transforms.Compose([transforms.RandomResizedCrop(64),
                            transforms.RandomHorizontalFlip(),
                            normalize])
    return ReassignedDataset(image_indexes, pseudolabels, dataset, labeled_tasks_flat, t_pil, t_tensor)



def arrange_clustering(images_lists):
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))
    indexes = np.argsort(image_indexes)
    return np.asarray(pseudolabels)[indexes]


class Kmeans(object):
    def __init__(self, k, num_centers_from_labeled_tasks):
        self.k = k
        self.num_centers_from_labeled_tasks = num_centers_from_labeled_tasks

    def cluster(self, data, data_labeled_tasks, verbose=False):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        end = time.time()

        num_tasks, n_way, num_samples_per_class, _ = data_labeled_tasks.shape

        data_labeled_tasks = data_labeled_tasks.reshape((num_tasks * n_way * num_samples_per_class, -1))

        combined_data = np.concatenate((data_labeled_tasks, data), axis=0)

        # PCA-reducing, whitening and L2-normalization
        xb = preprocess_features(combined_data)

        reduced_data_labeled_tasks = xb[:num_tasks * n_way * num_samples_per_class]
        reduced_data = xb[num_tasks * n_way * num_samples_per_class:]


        assert reduced_data.shape[0] == data.shape[0]

        reduced_data_labeled_tasks = reduced_data_labeled_tasks.reshape((num_tasks, n_way, num_samples_per_class, -1))

        # cluster the data
        assignments_labeled_data, assignments_unlabeled_data, loss = cluster_with_constraints(reduced_data_labeled_tasks, reduced_data,
                                           self.num_centers_from_labeled_tasks, self.k, max_iter=300)

        for i in range(num_tasks*n_way):
            if i%5 == 0:
                print("--")
            print(assignments_labeled_data[i*num_samples_per_class:(i+1)*num_samples_per_class])

        self.images_lists = [[] for i in range(self.k)]
        for i in range(len(data)):
            self.images_lists[assignments_unlabeled_data[i]].append(i)
        for i in range(len(data_labeled_tasks)):
            self.images_lists[assignments_labeled_data[i]].append(len(data) + i)

        assert len(set.union(*(set(el) for el in self.images_lists))) == len(data) + len(data_labeled_tasks)

        if verbose:
            print('k-means time: {0:.0f} s'.format(time.time() - end))

        return loss


def make_adjacencyW(I, D, sigma):
    """Create adjacency matrix with a Gaussian kernel.
    Args:
        I (numpy array): for each vertex the ids to its nnn linked vertices
                  + first column of identity.
        D (numpy array): for each data the l2 distances to its nnn linked vertices
                  + first column of zeros.
        sigma (float): Bandwidth of the Gaussian kernel.

    Returns:
        csr_matrix: affinity matrix of the graph.
    """
    V, k = I.shape
    k = k - 1
    indices = np.reshape(np.delete(I, 0, 1), (1, -1))
    indptr = np.multiply(k, np.arange(V + 1))

    def exp_ker(d):
        return np.exp(-d / sigma**2)

    exp_ker = np.vectorize(exp_ker)
    res_D = exp_ker(D)
    data = np.reshape(np.delete(res_D, 0, 1), (1, -1))
    adj_matrix = csr_matrix((data[0], indices[0], indptr), shape=(V, V))
    return adj_matrix


def find_maxima_cluster(W, v):
    n, m = W.shape
    assert (n == m)
    assign = np.zeros(n)
    # for each node
    pointers = list(range(n))
    for i in range(n):
        best_vi = 0
        l0 = W.indptr[i]
        l1 = W.indptr[i + 1]
        for l in range(l0, l1):
            j = W.indices[l]
            vi = W.data[l] * (v[j] - v[i])
            if vi > best_vi:
                best_vi = vi
                pointers[i] = j
    n_clus = 0
    cluster_ids = -1 * np.ones(n)
    for i in range(n):
        if pointers[i] == i:
            cluster_ids[i] = n_clus
            n_clus = n_clus + 1
    for i in range(n):
        # go from pointers to pointers starting from i until reached a local optim
        current_node = i
        while pointers[current_node] != current_node:
            current_node = pointers[current_node]

        assign[i] = cluster_ids[current_node]
        assert (assign[i] >= 0)
    return assign

