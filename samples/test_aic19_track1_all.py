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
# IMAGE_DIR = os.path.join(ROOT_DIR, "data")
DATA_BASE_DIR = "/mnt/disk2/AIC19/aic19-track1-mtmc/mrcnn/aic2019_track1_reid"
IMAGE_FOLDER = os.path.join(DATA_BASE_DIR, "train")
FOLDERLIST = os.listdir(IMAGE_FOLDER)  # object ids folders
print('total number of folders:', len(FOLDERLIST))
FOLDERLIST.sort()
OUTPUT_PATH = os.path.join(DATA_BASE_DIR, "train_mrcnn_results", "output_masks")
OUTPUT_DETS_PATH = os.path.join(DATA_BASE_DIR, "train_mrcnn_results", "output_dets")
if not os.path.exists(OUTPUT_PATH):
  os.makedirs(OUTPUT_PATH)
if not os.path.exists(OUTPUT_DETS_PATH):
  os.makedirs(OUTPUT_DETS_PATH)

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.6

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

for folder in FOLDERLIST:
  cam_dir = os.path.join(IMAGE_FOLDER, folder)
  camlist = os.listdir(cam_dir)
  camlist.sort()
  print(cam_dir)
  print(camlist)

  for camid in camlist:
    # Load images from the images folder
    image_dir = os.path.join(cam_dir, camid)
    imglist = os.listdir(image_dir)
    print('total number of images:', len(imglist))
    imglist.sort()

    output_folder = os.path.join(OUTPUT_PATH, folder, camid)
    if not os.path.exists(output_folder):
      os.makedirs(output_folder)
    print('output_folder:', output_folder)

    with open(OUTPUT_DETS_PATH + "/dets_%s_%s.txt" % (folder, camid), "w") as f:
      f.seek(0)
      f.truncate()
      for imgname in imglist:
        # if imgname.lstrip('0').split('.')[0] == "":
        #   frame_id = 0
        # else:
        #   frame_id = imgname.lstrip('0').split('.')[0]
        imgname_woext = imgname.split('.')[0]
        image = skimage.io.imread(os.path.join(image_dir, imgname))
        # Run detection
        results = model.detect([image], verbose=1)
        
        # Visualize results
        r = results[0]

        max_area = 0
        max_id = None
        max_roi = None
        max_score = None
        max_mask = None

        for roi, id, score, mask in zip(r['rois'], r['class_ids'], r['scores'], r['masks'].transpose(2,0,1)):
          # if int(id) == 3 or int(id) == 6 or int(id) == 8:
            y,x,h,w = roi[0], roi[1], roi[2]-roi[0], roi[3]-roi[1]
            if max_area < h * w:
              max_area = h * w
              max_id = id
              max_roi = roi
              max_score = score
              max_mask = mask

        # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],  class_names)

        if max_id is not None:
          y,x,h,w = max_roi[0], max_roi[1], max_roi[2]-max_roi[0], max_roi[3]-max_roi[1]
          np.save(output_folder + "/%s.npy"%imgname_woext, max_mask)
          visualize.display_instances(image, max_roi[np.newaxis], max_mask[:,:,np.newaxis], max_id[np.newaxis], class_names)
          f.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % (imgname_woext, -1, x, y, w, h, str("%.4f" % max_score), class_names[max_id], -1, -1))

        
