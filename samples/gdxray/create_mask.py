import json
import glob
import os
from pudb import set_trace
import numpy as np
import cv2

ROOT_DIR = '/home/auro/via/nirmalyalabs-work/'


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


# https://gist.github.com/douglasmiranda/5127251
# def find(key, dictionary):
#     for k, v in dictionary.items():
#         if k == key:
#             yield v
#         elif isinstance(v, dict):
#             for result in find(key, v):
#                 yield result
#         elif isinstance(v, list):
#             for d in v:
#                 for result in find(key, d):
#                     yield result


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


for json_name in glob.glob(ROOT_DIR + '/*/*/*.json'):
    print('\nJSON File name:', json_name)
    image_name_list = glob.glob(os.path.dirname(json_name) + '/*.png')
    if len(image_name_list) != 1 and not os.path.isfile(image_name_list[0]):
        print('File {}, does not exist'.format(image_name_list))
    else:
        image_name = image_name_list[0]
    print('\nImage file name:', image_name)
    image = cv2.imread(image_name)
    image_overlay = image.copy()
    # write the file names
    embed_text(image_overlay, json_name.split('/')[-1])

    with open(json_name, "r") as read_file:
        regions = json.load(read_file)

    set_trace()
    file_list = list(gen_dict_extract('filename', regions))
    print('File Names in JSON file', file_list,
          '****' if len(file_list) > 1 else "")

    region_list = list(gen_dict_extract('regions', regions))
    print('How many regions:{}'.format(len(region_list)))
    print('\nRegion list')
    print(region_list)

    print('\nEnumerate shapes')
    for shape_attrib in region_list[0]:
        print('all_points_x', shape_attrib['shape_attributes']['all_points_x'])
        print('all_points_y', shape_attrib['shape_attributes']['all_points_y'])
        pts = [[x, y] for x, y in
               zip(shape_attrib['shape_attributes']['all_points_x'],
                   shape_attrib['shape_attributes']['all_points_y'])]
        pts = np.array(pts[0:-1])
        print(pts)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(image_overlay, [pts], (128, 128, 0))
    opacity = 0.4
    # set_trace()
    cv2.addWeighted(image_overlay, opacity, image, 1 - opacity, 0, image)
    cv2.imwrite(image_name.split('/')[-1], image)
