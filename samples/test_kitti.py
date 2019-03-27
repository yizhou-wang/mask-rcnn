import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
DATE = '2011_09_26'
DRIVE = '0009'
FOLDER_NAME = DATE + "_drive_" + DRIVE + "_sync"
BASEDIR = "/mnt/disk1/kitti-dataset/raw_data/"
DATA_ROOT = BASEDIR + DATE + "/" + FOLDER_NAME
DATA_BASE_DIR = DATA_ROOT + "/image_02"
IMAGE_DIR = os.path.join(DATA_BASE_DIR, "data")
OUTPUT_PATH = os.path.join(DATA_BASE_DIR, "masks")
OUTPUT_IMG_PATH = os.path.join(DATA_BASE_DIR, "masks_img")

if not os.path.exists(OUTPUT_PATH):
  os.makedirs(OUTPUT_PATH)
if not os.path.exists(OUTPUT_IMG_PATH):
  os.makedirs(OUTPUT_IMG_PATH)


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.9

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle']

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# Load images from the images folder
imglist = os.listdir(IMAGE_DIR)
print('total number of images:', len(imglist))
imglist.sort()

with open(DATA_BASE_DIR + "/mrcnn_dets.txt", "w") as f:
  f.seek(0)
  f.truncate()
  for imgname in sorted(imglist):
    if imgname.lstrip('0').split('.')[0] == "":
      frame_id = 0
    else:
      frame_id = imgname.lstrip('0').split('.')[0]
    image = skimage.io.imread(os.path.join(IMAGE_DIR, imgname))
    # Run detection
    results = model.detect([image], verbose=1)
    
    # Visualize results
    r = results[0]
    # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names)
    # visualize.display_results(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], save_dir="../masks", img_name="%05d.png"%int(frame_id))

    obj_idx = 0

    for roi, id, score, mask in zip(r['rois'], r['class_ids'], r['scores'], r['masks'].transpose(2,0,1)):
      # if int(id) == 3 or int(id) == 6 or int(id) == 8:
      y,x,h,w = roi[0], roi[1], roi[2]-roi[0], roi[3]-roi[1]

      np.save(OUTPUT_PATH + "/%06d_%02d.npy"%(int(frame_id), obj_idx), mask)
      # print('class_ids shape:', r['class_ids'].shape)
      # print('mask shape:', r['masks'][:,:,0])

      # print(max_roi[np.newaxis].shape)
      # print(max_id[np.newaxis].shape)
      # print(max_score[np.newaxis].shape)
      # print(max_mask.shape)
      # print(max_mask[:,:,np.newaxis].shape)
      save_name = OUTPUT_IMG_PATH+"/%06d_%02d.jpg"%(int(frame_id), obj_idx)
      visualize.display_save_instances(image, save_name, roi[np.newaxis], mask[:,:,np.newaxis], id[np.newaxis], class_names)
      f.write("%s,%s,%s,%s,%s,%s,%s,%s\n" % (frame_id, obj_idx, x, y, w, h, str("%.4f" % score), class_names[id]))
      obj_idx += 1
    # print(r['rois'], r['class_ids'], r['scores'])
      
