import glob
import argparse
import random
import os
from shutil import copyfile
from pudb import set_trace
import shutil


parser = argparse.ArgumentParser(
    description='Split dataset into train and test')
parser.add_argument('--dataset', required=True,
                    metavar="/path/to/gdxray/dataset/",
                    help='Directory of the gdxray dataset')
parser.add_argument('--train_val_dir', required=True,
                    metavar="/path/to/dir",
                    help='Root directory of train and val dir')

args = parser.parse_args()
assert args.dataset != args.train_val_dir
shutil.rmtree(args.train_val_dir)  # start afresh


json_names_list = glob.glob(args.dataset + '/*/*/*.json')
random.shuffle(json_names_list)
print("Read {} JSON files.".format(len(json_names_list)))

# Creat train and val dirs at the same level
train_root_dir = os.path.join(os.path.normpath(args.train_val_dir), 'train')
val_root_dir = os.path.join(os.path.normpath(args.train_val_dir), 'val')

print('Train dir is {}'.format(train_root_dir))
print('Val dir is {}'.format(val_root_dir))

# Split the dataset into train and test
train_list = json_names_list[:int(len(json_names_list) * 0.8)]
val_list = json_names_list[int(len(json_names_list) * 0.8):]
print('Train list length is {}'.format(len(train_list)))
print('Val list length is {}'.format(len(val_list)))
assert len(json_names_list) == len(train_list) + len(val_list)

# Write out the train list files
for source_path in train_list:
    dest_path = source_path.replace(
        os.path.normpath(args.dataset), train_root_dir)
    # print('Source:{}, Dest:{}'.format(source_path, dest_path))
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    copyfile(source_path, dest_path)

train_names_list = glob.glob(train_root_dir + '/*/*/*.json')
print("Length of trainable JSON files: {}".format(len(train_names_list)))

# Write out the val list files
for source_path in val_list:
    dest_path = source_path.replace(
        os.path.normpath(args.dataset), val_root_dir)
    # print('Source:{}, Dest:{}'.format(source_path, dest_path))
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    copyfile(source_path, dest_path)

val_names_list = glob.glob(val_root_dir + '/*/*/*.json')
print("Length of validation-able JSON files: {}".format(len(val_names_list)))
