import json
import glob
import os
# from pudb import set_trace
import numpy as np
import cv2
import argparse
import logging


def embed_text(img, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, img.shape[0] - 10)
    fontScale = 1
    fontColor = (255, 255, 0)
    lineType = 2

    cv2.putText(img, text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)


def gen_dict_extract(key, var):
    """
    https://stackoverflow.com/questions/ \
    9807634/find-all-occurrences-of-a-key-in-nested-python-dictionaries-and-lists
    """
    if hasattr(var, 'items'):
        for k, v in var.items():
            if k == key:
                yield v
            if isinstance(v, dict):
                for result in gen_dict_extract(key, v):
                    yield result
            elif isinstance(v, list):
                for d in v:
                    for result in gen_dict_extract(key, d):
                        yield result


logging.basicConfig(filename='mask_creation.log', level=logging.INFO)
parser = argparse.ArgumentParser(
    description='Generate the masks and overlay on the images')
parser.add_argument("--root_dir",
                    help="Location of root dir for the json files",
                    required=True)
parser.add_argument("--out_dir",
                    help="Location of output dir for the masked files",
                    required=True)
args = parser.parse_args()
logging.info('Root dir:{}'.format(args.root_dir))
out_root_dir = os.path.join(args.out_dir, 'out')
logging.info('Output Root dir:{}'.format(out_root_dir))

count = 0
for json_name in glob.glob(args.root_dir + '/*/*/*.json'):
    logging.info('\nJSON File name:{}'.format(json_name))
    image_name_list = glob.glob(os.path.dirname(json_name) + '/*.png')
    if len(image_name_list) != 1 and not os.path.isfile(image_name_list[0]):
        logging.info('File {}, does not exist'.format(image_name_list))
    else:
        image_name = image_name_list[0]
    logging.info('\nImage file name:{}'.format(image_name))
    image = cv2.imread(image_name)
    image_overlay = image.copy()
    # write the file names
    embed_text(image_overlay, json_name.split('/')[-1])

    with open(json_name, "r") as read_file:
        regions = json.load(read_file)

    file_list = list(gen_dict_extract('filename', regions))
    logging.info('File Names in JSON file:{}'.format(file_list))
    if len(file_list) > 1:
        logging.error('File list greater that one')
        print("JSON file name: {}".format(json_name))
        print("File list greater that one: {}".format(file_list))

    region_list = list(gen_dict_extract('regions', regions))
    logging.info('How many regions:{}'.format(len(region_list)))
    logging.info('\nRegion list')
    logging.info(region_list)

    logging.info('\nEnumerate shapes')
    for shape_attrib in region_list[0]:
        logging.info('all_points_x {}'.format(
            shape_attrib['shape_attributes']['all_points_x']))
        logging.info('all_points_y {}'.format(
            shape_attrib['shape_attributes']['all_points_y']))
        pts = [[x, y] for x, y in
               zip(shape_attrib['shape_attributes']['all_points_x'],
                   shape_attrib['shape_attributes']['all_points_y'])]
        pts = np.array(pts[0:-1])
        logging.info(pts)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(image_overlay, [pts], (128, 128, 0))
    opacity = 0.4
    cv2.addWeighted(image_overlay, opacity, image, 1 - opacity, 0, image)

    out_file = os.path.join(out_root_dir, image_name.split(args.root_dir)[1])
    out_dir = os.path.dirname(out_file)
    logging.info("Out file: {}".format(out_file))
    logging.info("Out Dir: {}".format(out_dir))
    os.makedirs(out_dir, exist_ok=True)

    cv2.imwrite(out_file, image)
    # cv2.imwrite(image_name.split('/')[-1], image)
    print('.', end='')  # time marker
    count += 1
print("Total JSON-files processed: {}".format(count))
