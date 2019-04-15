import glob
import argparse
import random
import os
from shutil import copyfile
from pudb import set_trace
import shutil
import logging


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

    # Now copy the image
    image_name_list = glob.glob(os.path.dirname(source_path) + '/*.png')
    if len(image_name_list) != 1 and not os.path.isfile(image_name_list[0]):
        logging.info('File {}, does not exist'.format(image_name_list))
    else:
        source_image_path = image_name_list[0]
    dest_image_path = source_image_path.replace(
        os.path.normpath(args.dataset), train_root_dir)
    copyfile(source_image_path, dest_image_path)

train_names_list = glob.glob(train_root_dir + '/*/*/*.json')
print("Total train JSON files: {}".format(len(train_names_list)))
train_image_list = glob.glob(train_root_dir + '/*/*/*.png')
print("Total train Image files: {}".format(len(train_image_list)))

# Write out the val list files
for source_path in val_list:
    dest_path = source_path.replace(
        os.path.normpath(args.dataset), val_root_dir)
    # print('Source:{}, Dest:{}'.format(source_path, dest_path))
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    copyfile(source_path, dest_path)

    # Now copy the image
    image_name_list = glob.glob(os.path.dirname(source_path) + '/*.png')
    if len(image_name_list) != 1 and not os.path.isfile(image_name_list[0]):
        logging.info('File {}, does not exist'.format(image_name_list))
    else:
        source_image_path = image_name_list[0]
    dest_image_path = source_image_path.replace(
        os.path.normpath(args.dataset), val_root_dir)
    copyfile(source_image_path, dest_image_path)

val_names_list = glob.glob(val_root_dir + '/*/*/*.json')
print("Total validation JSON files: {}".format(len(val_names_list)))
val_image_list = glob.glob(val_root_dir + '/*/*/*.png')
print("Total validation Image files: {}".format(len(val_image_list)))
