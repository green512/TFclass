# Copyright 2015 The TensorFlow Authors. All Rights Reserved.

#

# Licensed under the Apache License, Version 2.0 (the "License");

# you may not use this file except in compliance with the License.

# You may obtain a copy of the License at

#

#     http://www.apache.org/licenses/LICENSE-2.0

#

# Unless required by applicable law or agreed to in writing, software

# distributed under the License is distributed on an "AS IS" BASIS,

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# See the License for the specific language governing permissions and

# limitations under the License.

# ==============================================================================



"""Simple image classification with Inception.

Run image classification with Inception trained on ImageNet 2012 Challenge data

set.

This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. It outputs human readable
strings of the top 5 predictions along with their probabilities.

Change the --image_file argument to any jpg image to compute a

classification of that image.

Please see the tutorial and website for a detailed description of how
to use this script to perform image recognition.

https://tensorflow.org/tutorials/image_recognition/

"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import re
import sys

import numpy as np

import PIL
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import tensorflow as tf
from tensorflow.python.platform import gfile

FLAGS = None

current_dir_path = os.path.dirname(os.path.realpath(__file__))

def create_graph():

  """Creates a graph from saved GraphDef file and returns a saver."""

  # Creates graph from saved graph_def.pb.

  with tf.gfile.FastGFile(os.path.join(

      FLAGS.model_dir, 'output_graph4.pb'), 'rb') as f:

    graph_def = tf.GraphDef()

    graph_def.ParseFromString(f.read())

    _ = tf.import_graph_def(graph_def, name='')





def run_inference_on_image(image_path):

  """Runs inference on an image.

  Args:

    image: Image file name.

  Returns:

    Nothing

  """

  # Creates graph from saved GraphDef.

  create_graph()

  with tf.Session() as sess:

    # Some useful tensors:

    # 'softmax:0': A tensor containing the normalized prediction across

    #   1000 labels.

    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048

    #   float description of the image.

    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG

    #   encoding of the image.

    # Runs the softmax tensor by feeding the image_data as input to the graph.
	
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
    file_list = []
    for extension in extensions:
      file_glob = os.path.join(image_path, '*.' + extension)
      file_list.extend(gfile.Glob(file_glob))
    if not file_list:
      tf.logging.fatal('No files found in  %s', image_path)

    for imageFile in file_list:  
      if not tf.gfile.Exists(imageFile):
        tf.logging.fatal('File does not exist %s', imageFile)

      image_data = tf.gfile.FastGFile(imageFile, 'rb').read()
      predictions = sess.run(softmax_tensor,
						   {'DecodeJpeg/contents:0': image_data})

      predictions = np.squeeze(predictions)

	  # Creates node ID --> English string lookup.

      a=open(os.path.join(FLAGS.model_dir, 'output_labels4.txt'))
      lines=a.readlines()

      top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
      im1 = Image.open(imageFile)
      draw = ImageDraw.Draw(im1)
      newfont=ImageFont.truetype('/usr/share/fonts/truetype/ubuntu-font-family/Ubuntu-B.ttf',40)

      for node_id in top_k:
        human_string = lines[node_id].strip('\n')
        score = predictions[node_id]
        print('%s (score = %.5f)' % (human_string, score))		
		
        draw.text((100, 0), human_string, (255, 0, 0), font=newfont)    #postion/content/color/font
        draw = ImageDraw.Draw(im1)                         #Just draw it!
      base_name = os.path.basename(imageFile)
      im1.save(os.path.join('result',human_string, base_name))


def main(_):

  run_inference_on_image(FLAGS.image_path)


if __name__ == '__main__':

  parser = argparse.ArgumentParser()

  # classify_image_graph_def.pb:

  #   Binary representation of the GraphDef protocol buffer.

  # imagenet_synset_to_human_label_map.txt:

  #   Map from synset ID to a human readable string.

  # imagenet_2012_challenge_label_map_proto.pbtxt:

  #   Text representation of a protocol buffer mapping a label to synset ID.

  parser.add_argument(

      '--model_dir',

      type=str,

      default= '/home/cropwatch/green',

      help="""\

      Path to classify_image_graph_def.pb,

      imagenet_synset_to_human_label_map.txt, and

      imagenet_2012_challenge_label_map_proto.pbtxt.\

      """

  )

  parser.add_argument(

      '--image_path',

      type=str,

      default='/home/cropwatch/dockerdir/gvg/val/rice',

      help='Absolute path to image path.'

  )

  parser.add_argument(

      '--num_top_predictions',

      type=int,

      default=1,

      help='Display this many predictions.'

  )

  FLAGS, unparsed = parser.parse_known_args()

  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)