#!/bin/bash

DIR_LABELED="/home/cle_macho/mini_imagenet/train_split/labeled_use"

python load_small_num_tasks_and_save_to_file.py --small_train_data_path ${DIR_LABELED}
