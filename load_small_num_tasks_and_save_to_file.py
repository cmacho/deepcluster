import meta_data_loader as mdl
import argparse
import numpy as np
import os

def parse_args():
    parser = argparse.ArgumentParser(description="sample small number of tasks")
    parser.add_argument("--small_train_data_path", type=str, default="/home/cle_macho/mini_imagenet/train_split/labeled_use",
                        help="the part of the training data split that is meant for labeled use, e.g. only 12 classes")
    parser.add_argument("--n_way", default=5, type=int, help="number of classes in each task")
    parser.add_argument("--k_shot", default=1, type =int, help="number of support images per class in a task")
    parser.add_argument("--num_query_per_class", default=5, type=int, help="number of query images per class in a task")
    parser.add_argument("--num_tasks", type=int, default=100, help="how many tasks to sample when training with small num of tasks")
    parser.add_argument("--out_file_name", type=str, default="labeled_tasks.npy", help="name of generated npy file")
    return parser.parse_args()



def main():
    args = parse_args()
    if args.out_file_name is None:
        raise Exception("no output filename specified")

    X, Y, total_num_classes = mdl.load_data_from_folders(args.small_train_data_path)
    initial_data_loader = mdl.DataLoader(X, Y, total_num_classes)
    tasks = initial_data_loader.sample_batch(args.n_way, args.k_shot + args.num_query_per_class, args.num_tasks)

    np.save(args.out_file_name, tasks)

if __name__ == "__main__":
    main()