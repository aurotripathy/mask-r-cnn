import os
import sys
import tensorflow as tf
from pudb import set_trace
from shutil import rmtree
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
print('ROOT DIR', ROOT_DIR)

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
sys.path.insert(0, ROOT_DIR)

from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
import mrcnn.data_generator as data_generator_lib
from mrcnn.model import log

from samples.gdxray import gdxray

# %matplotlib inline

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


config = gdxray.GdxrayConfig()
GDXRAY_DIR = "/home/auro/via/n-test/"
RESULTS_DIR = "results"
if os.path.isdir(RESULTS_DIR):
    rmtree(RESULTS_DIR)  # start afresh


class InferenceConfig(config.__class__):
    """ Override the training configurations with a few changes
    for inferencing.
    """
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

# Device to load the neural network on.
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"


# Load validation dataset
dataset = gdxray.GdxrayDataset()
dataset.load_gdxray(GDXRAY_DIR, "val")

# Must call before using the dataset
dataset.prepare()

print("Images: {}\nClasses: {}".format(
    len(dataset.image_ids), dataset.class_names))

# Create model in inference mode
print(MODEL_DIR)
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)
print('Finished loading model')

# Set path to gdxray  weights file

# Load the last model you trained, something like...
# ...home/auro/keras-examples/Mask_RCNN/logs/gdxray20190414T1736/mask_rcnn_gdxray_0030.h5
weights_path = model.find_most_recent_checkpoint()

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

# Run Detection on all images
for image_id in dataset.image_ids:
    set_trace()
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        data_generator_lib.load_image_gt(
            dataset, config, image_id, use_mini_mask=False)
    info = dataset.image_info[image_id]
    print("image ID: {}.{} ({}) {}".format(info["source"], info["id"],
                                           image_id,
                                           dataset.image_reference(image_id)))

    # Get a dir ready to write the results in
    rel_path = os.path.relpath(dataset.image_reference(image_id), GDXRAY_DIR)
    file_path = os.path.join(RESULTS_DIR, rel_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Run object detection
    results = model.detect([image], verbose=1)

    # Display results
    r = results[0]
    # visualize.display_differences(image,
    #                               gt_bbox, gt_class_id, gt_mask,
    #                               r['rois'], r['class_ids'],
    #                               r['scores'], r['masks'],
    #                               dataset.class_names, ax=None,
    #                               title=None)
    visualize.save_instances_in_dir(image, r['rois'], r['masks'],
                                    r['class_ids'],
                                    dataset.class_names,
                                    file_path,
                                    ax=None, show_bbox=False,
                                    title="Predictions", captions=[""] * 20)

    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)
