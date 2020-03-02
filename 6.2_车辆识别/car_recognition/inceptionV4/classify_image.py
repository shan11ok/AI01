from __future__ import absolute_import, division, print_function

import argparse
import os.path
import re
import sys
import tarfile

import os

import numpy as np
import tensorflow as tf
from six.moves import urllib

r'''tf.app.flags.DEFINE_string(
    'model_file', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')'''

FLAGS = tf.app.flags.FLAGS

#env_dist = os.environ
#PROJECT_DIR = env_dist.get('PROJECT_DIR')
#print(PROJECT_DIR)

class In4Pre:
    def __init__(self, model_file=None):
        if not model_file:
            tf.logging.fatal('please specify the model file.')
            return
        create_graph(model_file)
        self.model_file = model_file
        self.sess = tf.Session()
        self.node_lookup = NodeLookup('./inceptionV4/outputfile/freezed.label')
    
    def run(self, image=None):
        if not tf.gfile.Exists(image):
            tf.logging.fatal('File does not exist %s', image)
        image_data = open(image, 'rb').read()

        softmax_tensor = self.sess.graph.get_tensor_by_name('final_probs:0')
        predictions = self.sess.run(softmax_tensor, {'input:0': image_data})
        predictions = np.squeeze(predictions)

        # Creates node ID --> English string lookup.
        #node_lookup = NodeLookup(PROJECT_DIR + '/inceptionV4/outputfile/freezed.label')
        #node_lookup = NodeLookup('./inceptionV4/outputfile/freezed.label')

        top_k = predictions.argsort()[-5:][::-1]
        top_names = []
        for node_id in top_k:
            human_string = self.node_lookup.id_to_string(node_id)
            top_names.append(human_string)
            score = predictions[node_id]
            print('id:[%d] name:[%s] (score = %.5f)' %
                  (node_id, human_string, score))
        return predictions, top_k, top_names
    
    def __del__(self):
        self.sess.close()

class NodeLookup(object):
    """Converts integer node ID's to human readable labels."""

    def __init__(self,
                 label_path=None):
        if not label_path:
            tf.logging.fatal('please specify the label file.')
            return
        self.node_lookup = self.load(label_path)

    def load(self, label_path):
        """Loads a human readable English name for each softmax node.
        Args:
          label_lookup_path: string UID to integer node ID.
          uid_lookup_path: string UID to human-readable string.
        Returns:
          dict from integer node ID to human-readable string.
        """
        if not tf.gfile.Exists(label_path):
            tf.logging.fatal('File does not exist %s', label_path)

        # Loads mapping from string UID to human-readable string
        proto_as_ascii_lines = tf.gfile.GFile(label_path).readlines()
        id_to_human = {}
        for line in proto_as_ascii_lines:
            if line.find(':') < 0:
                continue
            _id, human = line.rstrip('\n').split(':')
            id_to_human[int(_id)] = human

        return id_to_human

    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]


def create_graph(model_file=None):
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    if not model_file:
        model_file = FLAGS.model_file
    with open(model_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image_backup(image, model_file=None):
    """Runs inference on an image.
    Args:
      image: Image file name.
    Returns:
      Nothing
    """
    if not tf.gfile.Exists(image):
        tf.logging.fatal('File does not exist %s', image)
    image_data = open(image, 'rb').read()
    # image_data = image
    # Creates graph from saved GraphDef.
    create_graph(model_file)

    with tf.Session() as sess:
        # Some useful tensors:
        # 'softmax:0': A tensor containing the normalized prediction across
        #   1000 labels.
        # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
        #   float description of the image.
        # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
        #   encoding of the image.
        # Runs the softmax tensor by feeding the image_data as input to the graph.
        softmax_tensor = sess.graph.get_tensor_by_name('final_probs:0')
        predictions = sess.run(softmax_tensor,
                               {'input:0': image_data})
        predictions = np.squeeze(predictions)

        # Creates node ID --> English string lookup.
        #node_lookup = NodeLookup(PROJECT_DIR + '/inceptionV4/outputfile/freezed.label')
        node_lookup = NodeLookup('./inceptionV4/outputfile/freezed.label')

        top_k = predictions.argsort()[-5:][::-1]
        top_names = []
        for node_id in top_k:
            human_string = node_lookup.id_to_string(node_id)
            top_names.append(human_string)
            score = predictions[node_id]
            print('id:[%d] name:[%s] (score = %.5f)' %
                  (node_id, human_string, score))
    return predictions, top_k, top_names


def main(_):
    image = (FLAGS.image_file if FLAGS.image_file else
             os.path.join(FLAGS.model_dir, 'test.jpg'))
    run_inference_on_image(image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # classify_image_graph_def.pb:
    #   Binary representation of the GraphDef protocol buffer.
    # imagenet_synset_to_human_label_map.txt:
    #   Map from synset ID to a human readable string.
    # imagenet_2012_challenge_label_map_proto.pbtxt:
    #   Text representation of a protocol buffer mapping a label to synset ID.
    parser.add_argument(
        '--model_file',
        type=str,
        default='./outputfile/freezed.pb',
        help="""\
      Path to the .pb file that contains the frozen weights. \
      """
    )
    parser.add_argument(
        '--label_file',
        type=str,
        default='./outputfile/freezed.label',
        help='Absolute path to label file.'
    )
    parser.add_argument(
        '--image_file',
        type=str,
        default='',
        help='Absolute path to image file.'
    )
    parser.add_argument(
        '--num_top_predictions',
        type=int,
        default=5,
        help='Display this many predictions.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
