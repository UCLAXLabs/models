
# coding: utf-8

# Adapted from the Object Detection Demo Jupyter notebook
# Generates a IIIF Curation API JSON document that records the bounding
# boxes of prominent objects detected in each image in the input set,
# which can be used to populate an interactive photo collection explorer
# based on the IIIF Presentation layer.
#
# This is a modification of iiif_curation.py, with the main difference
# being that it finds the images for object detection inference entirely
# from IIIF manifests that may be hosted remotely and downloads the 
# images directly from their IIIF servers, rather than needing the images
# to be stored locally.

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

from io import BytesIO
import pickle
import requests
from slugify import slugify

sys.path.append("..")
from object_detection.utils import ops as utils_ops

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

# This is needed to display the images -- but only in the Jupyter notebook
#%matplotlib inline

# ## Object detection imports
# Here are the imports from the object detection module.
from utils import label_map_util
from utils import visualization_utils as vis_util

# # Model preparation 

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  

#from matplotlib import pyplot as plt

# What model to use.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

#MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28'
#MODEL_FILE = MODEL_NAME + '.tar.gz'
#DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
# When training/transfer-learn training a new model, must run this first:
# python3 export_inference_graph.py --input_type image_tensor --pipeline_config_path models/model/pipeline.config --trained_checkpoint_prefix models/model/model.ckpt-3922 --output_directory exported_graphs
PATH_TO_CKPT = 'exported_graphs/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'label_map.pbtxt')

# label_map_util should be able to figure this out or at least choose a reasonable
# default, but it's silly and doesn't, so it needs to be specified here
NUM_CLASSES = 16

# Set to True to create a IIIF curation manifest of the detected regions
writeManifest = True

# Set to True to write images of detected regions to a folder
saveImages = True

cacheEnabled = True
cachePath = os.path.join(os.getcwd(), 'cache/')

targetManifests = ['http://marinus.library.ucla.edu/images/kabuki/manifest.json']

outputFolder = os.path.join(os.getcwd(), 'output/')
if (saveImages):
  try:
    if (not os.path.exists(outputFolder)):
      os.makedirs(outputFolder)
  except:
    print("Unable to create clippings folder, won't save them")
    saveImages = False

# ## Download Model
if (not os.path.isfile(PATH_TO_CKPT)):

  print("Couldn't find checkpoint, downloading model " + MODEL_FILE)

  opener = urllib.request.URLopener()
  opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
  tar_file = tarfile.open(MODEL_FILE)
  for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
      tar_file.extract(file, os.getcwd())
      PATH_TO_CKPT = os.getcwd()


# ## Load the (frozen) Tensorflow model into memory.

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
print("CATEGORIES:",str(categories))
category_index = label_map_util.create_category_index(categories)


# ## Helper code

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

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
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
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
      output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

try:
  if (not os.path.exists(cachePath)):
    os.makedirs(cachePath)
except:
  print("Unable to create cache folder, will download everything")
  cacheEnaled = False

def getURL(link, useCache=cacheEnabled):
  fileSlug = slugify(link)
  filePath = os.path.join(cachePath, fileSlug)
  if (useCache and os.path.isfile(filePath)):
    data = pickle.load(open(filePath, 'rb'))
    print("fetched from cache: " + link)
  else:
    print("fetching " + link)
    data = requests.get(link)
    if (useCache):
      pickle.dump(data, open(filePath, 'wb'))
  return data

def processManifest(maniData):
  theseMappings = {}
  for sequence in maniData['sequences']:
    for canvas in sequence['canvases']:
      canvasID = canvas['@id']
      for image in canvas['images']:

        # Could explicitly save the files here, but they'll already be in the cache,
        # so why bother?
        #with open(outputPath, 'w') as outputFile:
        #  print("harvesting cropped image " + imageID)
        #  im.save(outputFile, 'jpeg')
        if (canvasID not in theseMappings):
          theseMappings[canvasID] = [image]
        else:
          theseMappings[canvasID].append(image)
  return theseMappings

# MAIN

# Preamble of the JSON curation document
if (writeManifest):

  # Should indicate the machine the curation manifest lives on, with a subdomain specific to the project
  iiifProject = 'edo_illustrations'
  iiifDomain = 'http://marinus.library.ucla.edu/images/' + iiifProject

  curationUUID = str(uuid.uuid4())

  jsonFile = open('tf_curation.json', 'w')

  jsonFile.write('{ "@context": [ "http://iiif.io/api/presentation/2/context.json", "http://codh.rois.ac.jp/iiif/curation/1/context.json" ], "@type": "cr:Curation", "@id": "' + iiifDomain + '/json/' + curationUUID + '", "label": "Curation list", "selections": [ { "@id": "' + iiifDomain + '.json/range/r1", "@type": "sc:Range", "label": "Objects detected by ' + MODEL_NAME + '", "members": [')

isFirstBox = True

# This is where relevant data from each manifest will be stored
maniMappings = {}

# Parse each of the specified manifests
for srcManifest in targetManifests:
  if (srcManifest not in maniMappings):
    print("Processing manifest",srcManifest)
    manifestData = getURL(srcManifest).json()
    newMappings = processManifest(manifestData)
    maniMappings[srcManifest] = newMappings

for srcManifest in maniMappings:
  for canvasID in maniMappings[srcManifest]:
    for image in maniMappings[srcManifest][canvasID]: 

      fullURL = image['resource']['@id']
      # IIIF presentation always returns a .jpg
      #imageID = canvasID.split('/')[-1].replace('.json','').replace('.tif','').replace('.png','').replace('.jpg','').replace('.jpeg','') + ".jpg"
      imageID = image['resource']['service']['@id'].split('/')[-1].replace('.tif','').replace('.png','').replace('.jpg','').replace('.jpeg','') + ".jpg"

      fullWidth = image['resource']['width']
      fullHeight = image['resource']['height']

      # Files will be cached as byte streams by default
      resizedURL = fullURL.replace('full/full','full/!1000,1000')
      imageResponse = getURL(resizedURL)
      im = Image.open(BytesIO(imageResponse.content))
      resizedWidth, resizedHeight = im.size

      #image = Image.open(image_path).convert('RGB')
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.

      image_np = load_image_into_numpy_array(im)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      # Actual detection.
      print("running detection on " + imageID)
      output_dict = run_inference_for_single_image(image_np, detection_graph)

      # If a detected object satisfies certain conditions (size, class, score),
      # add it to the JSON curation document.
  
      # XXX Arbitrary parameters!
      min_score_thresh = .9
      min_box_proportion_thresh = .001

      # These should be the same as resizedWidth and resizedHeight
      image_height = image_np.shape[0]
      image_width = image_np.shape[1]
      image_area = image_height * image_width

      # Compute the ratio between resized and full-sized image
      # Multiple the resized (smaller) size by this factor to get
      # the desired value for the full-sized image
      heightRatio = fullHeight / image_height
      widthRatio = fullWidth / image_width

      print("image area is",image_area)
      print("image width is",image_width)
      print("image height is",image_height)

      boxes = output_dict['detection_boxes']
      classes = output_dict['detection_classes']
      scores = output_dict['detection_scores']

      print("processing",boxes.shape[0],"boxes")
  
      for i in range(boxes.shape[0]):
        print("score of box",i,scores[i])

        if (float(scores[i]) < min_score_thresh):
          continue
  
        print("score of box",i,"is",scores[i])
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

        box_width = box_unit_width * image_width
        print("box_width",box_width)
        box_height = box_unit_height * image_height
        print("box_height",box_height)

        box_X = float(box[1]) * image_width
        print("box_X",box_X)
        box_Y = float(box[0]) * image_height
        print("box_Y",box_Y)

        xywh = list(map(int, (box_X, box_Y, box_width, box_height)))
        xywhString = ','.join(list(map(str,xywh)))
        print("box xywh on resized image is",xywhString)
 
        fullXYWH = list(map(int, (box_X * widthRatio, box_Y * heightRatio, box_width * widthRatio, box_height * heightRatio)))
        fullXYWHstring = ','.join(list(map(str,fullXYWH)))
        print("box xywh on full-sized image is",fullXYWHstring)

        if classes[i] in category_index.keys():
          class_name = category_index[classes[i]]['name']
        else:
          class_name = 'NA'

        if (saveImages):
          croppedImageID = imageID + '.' + xywhString + '_' + class_name + '.png'
          # Format of cropBox: xmin, ymin, xmax, ymax
          cropBox = (xywh[0], xywh[1], xywh[0] + xywh[2], xywh[1] + xywh[3])
          croppedImage = im.crop(cropBox)
          croppedPath = os.path.join(outputFolder, croppedImageID)

          #with open(croppedPath, 'w') as croppedFile:
          print("saving cropped image " + croppedImageID)
          croppedImage.save(croppedPath, 'png') 

        if (writeManifest):
          boxJSON = {
            "@id": canvasID + "#xywh=" + fullXYWHstring,
            "type": "sc:Canvas",
            "label": imageID,
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


if (writeManifest):
  # Do this at the very end
  jsonFile.write( '], "within": { "@id": "' + iiifDomain + '_manifest.json", "@type": "sc:Manifest", "label": "' + iiifProject + '" } } ] }')

  jsonFile.close()
