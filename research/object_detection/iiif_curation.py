
# coding: utf-8

# Adapted from the Object Detection Demo Jupyter notebook
# Generates a IIIF Curation API JSON document that records the bounding
# boxes of prominent objects detected in each image in the input set,
# which can be used to populate an interactive photo collection explorer
# based on the IIIF Presentation layer.

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import json
import uuid

from collections import defaultdict
from io import StringIO
from PIL import Image

sys.path.append("..")
from object_detection.utils import ops as utils_ops

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

# This is needed to display the images -- but only in the Jupyter notebook
#%matplotlib inline


# ## Object detection imports
# Here are the imports from the object detection module.

# In[3]:


from utils import label_map_util

from utils import visualization_utils as vis_util

# # Model preparation 

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  

from matplotlib import pyplot as plt

# What model to download.
#MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
#MODEL_FILE = MODEL_NAME + '.tar.gz'
#DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


# ## Download Model

# In[5]:

if (not os.path.isfile(PATH_TO_CKPT)):

  print("Downloading model " + MODEL_FILE)

  opener = urllib.request.URLopener()
  opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
  tar_file = tarfile.open(MODEL_FILE)
  for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
      tar_file.extract(file, os.getcwd())


# ## Load a (frozen) Tensorflow model into memory.

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# # Detection

# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = '/mnt/clab/ucla_ua'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, i) for i in os.listdir(PATH_TO_TEST_IMAGES_DIR) ]
#TEST_IMAGE_PATHS = [ 'test_images/uclalsc_uars100_1100_008a.jpg' ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


# In[10]:


def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

# MAIN

# Preamble of the JSON curation document

curationUUID = str(uuid.uuid4())

jsonFile = open('tf_curation.json', 'w')

jsonFile.write('{ "@context": [ "http://iiif.io/api/presentation/2/context.json", "http://codh.rois.ac.jp/iiif/curation/1/context.json" ], "@type": "cr:Curation", "@id": "http://164.67.17.127/images/ucla_ua/json/' + curationUUID + '", "label": "Curation list", "selections": [ { "@id": "http://164.67.17.127/images/ucla_ua_manifest.json/range/r1", "@type": "sc:Range", "label": "Objects detected by ' + MODEL_NAME + '", "members": [')

isFirstBox = True

for image_path in TEST_IMAGE_PATHS:
  print("examining " + image_path)
  if (not os.path.isfile(image_path)):
    continue

  basename = os.path.basename(image_path)
  image = Image.open(image_path)
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = load_image_into_numpy_array(image)
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  image_np_expanded = np.expand_dims(image_np, axis=0)
  # Actual detection.
  print("running detection on " + basename)
  output_dict = run_inference_for_single_image(image_np, detection_graph)

  # If a detected object satisfies certain conditions (size, class, score),
  # add it to the JSON curation document.

  # XXX Arbitrary parameters!
  min_score_thresh = .9
  min_box_proportion_thresh = .001

  image_area = image_np.shape[0] * image_np.shape[1]
  image_height = image_np.shape[0]
  image_width = image_np.shape[1]

  print("image area is",image_area)
  print("image width is",image_width)
  print("image height is",image_height)

  boxes = output_dict['detection_boxes']
  classes = output_dict['detection_classes']
  scores = output_dict['detection_scores']

  print("processing",boxes.shape[0],"boxes")

  for i in range(boxes.shape[0]):
    print("score of box",i,"is",scores[i])
    if (float(scores[i]) < min_score_thresh):
      continue

    pct_score = "{:.0%}".format(scores[i])
    
    # PMB Maybe also check whether class label is in a desired subset?
    box = tuple(boxes[i].tolist()) # box is ymin, xmin, ymax, xmax
    print("box points are",box)
    box_unit_width = box[3] - box[1]
    print("box unit width",box_unit_width)
    box_unit_height = box[2] - box[0]
    print("box unit height",box_unit_height)
    box_proportion = box_unit_width * box_unit_height
    print("proportion is",box_proportion)
    if (box_proportion < min_box_proportion_thresh):
      continue

    if (box_proportion < .01):
      proportionString = "<.01"
    else:
      proportionString = str(round(box_proportion,2))

    box_width = int(box_unit_width * image_width)
    print("box_width",box_width)
    box_height = int(box_unit_height * image_height)
    print("box_height",box_height)

    box_X = int(float(box[1]) * image_width)
    print("box_X",box_X)
    box_Y = int(float(box[0]) * image_height)
    print("box_Y",box_Y)

    xywh = ','.join([str(box_X),str(box_Y),str(box_width),str(box_height)])
    print("xywh is",xywh)

    if classes[i] in category_index.keys():
      class_name = category_index[classes[i]]['name']
    else:
      class_name = 'N/A'

    boxJSON = {
          "@id": "http://164.67.17.127/images/ucla_ua/canvas/" + basename + ".json#xywh=" + xywh,
          "type": "sc:Canvas",
          "label": basename,
          "metadata": [
            {
              "label": "tag",
              "value": class_name
            },
            {
              "label": "confidence",
              "value": pct_score
            },
            {
              "label": "proportion",
              "value": proportionString
            }
          ]
        }

    box_str = json.dumps(boxJSON)
    if (isFirstBox == True):
      jsonFile.write(box_str + "\n")
      isFirstBox = False
    else:
      jsonFile.write("," + box_str + "\n")

# Do this at the very end
jsonFile.write( '], "within": { "@id": "http://164.67.17.127/images/ucla_ua_manifest.json", "@type": "sc:Manifest", "label": "ucla_ua" } } ] }')

jsonFile.close()
