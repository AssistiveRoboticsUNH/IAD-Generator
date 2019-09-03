# convert a caffe model to a tensorflow ckpt file

# setup file arguments 
import argparse
parser = argparse.ArgumentParser(description='Generate IADs from input files')
parser.add_argument('model_location', help='where the caffemodel file can be located')
parser.add_argument('num_classes', help='how many classes in the dataset')
parser.add_argument('dst_location', help='where the output ckpt file should go')
FLAGS = parser.parse_args()

# setup caffe imports
import tensorflow as tf
import numpy as np
import sys, os
caffe_root = '../../../'
sys.path.insert(0, caffe_root + 'python')
print(sys.path)
import caffe

def net_to_py_readable(prototxt_filename, caffemodel_filename):
  net = caffe.Net(prototxt_filename, caffemodel_filename, caffe.TEST) # read the net + weights
  pynet_ = {} 

  for li in xrange(len(net.layers)):  # for each layer in the net
    layer = {}  # store layer's information
    layer['name'] = net._layer_names[li]
    # for each input to the layer (aka "bottom") store its name and shape
    layer['bottoms'] = [(net._blob_names[bi], net.blobs[net._blob_names[bi]].data.shape) 
                         for bi in list(net._bottom_ids(li))] 
    # for each output of the layer (aka "top") store its name and shape
    layer['tops'] = [(net._blob_names[bi], net.blobs[net._blob_names[bi]].data.shape) 
                      for bi in list(net._top_ids(li))]
    layer['type'] = net.layers[li].type  # type of the layer
    # the internal parameters of the layer. not all layers has weights.
    layer['weights'] = [net.layers[li].blobs[bi].data[...] 
                        for bi in xrange(len(net.layers[li].blobs))]
    pynet_[layer['name']] = layer
  
  return pynet_

def load_conv_3d_weight(ori_name, shape, caffe_name, caffe_dict):
  values = tf.constant(caffe_dict[caffe_name]['weights'][0], name="filter")
  values = tf.transpose(values, [2,3,4,1,0])

  return tf.get_variable(name=ori_name, initializer=values)

def load_bias(ori_name, shape, caffe_name, caffe_dict):
  values = tf.constant(caffe_dict[caffe_name]['weights'][1], name="filter", dtype=tf.float32)

  return tf.get_variable(name=ori_name, initializer=values)

def load_fc_weight(ori_name, shape, caffe_name, caffe_dict):
  values = tf.constant(caffe_dict[caffe_name]['weights'][0], name="filter", dtype=tf.float32)
  values = tf.transpose(values, [1,0])

  return tf.get_variable(name=ori_name, initializer=values)

if __name__ == '__main__':
  prototxt_filename = "c3d_model_only.prototxt"

  # find the most recent caffemodel in the provided directory
  directory = FLAGS.model_location
  caffemodel_filename = [x for x in os.listdir(directory) if x.find(".caffemodel") > 0][0]
  caffemodel_filename = os.path.join(directory, caffemodel_filename)
  
  print("Generating model from: "+caffemodel_filename)

  #open the caffe model in python
  pynet = net_to_py_readable(prototxt_filename, caffemodel_filename)
  
  for k in pynet.keys():
    print(pynet[k]['name'])
    for n in range(len(pynet[k]['weights'])):
      print(pynet[k]['weights'][n].shape)
  
  # the python model definition, using teh values from pynet instead of typical initalizers
  with tf.variable_scope('var_name') as var_scope:
    NUM_CLASSES = FLAGS.num_classes
   
    weights = {
            'wc1': load_conv_3d_weight('wc1', [3, 3, 3, 3, 64], 'conv1a', pynet),
            'wc2': load_conv_3d_weight('wc2', [3, 3, 3, 64, 128], 'conv2a', pynet),
            'wc3a': load_conv_3d_weight('wc3a', [3, 3, 3, 128, 256], 'conv3a', pynet),
            'wc4a': load_conv_3d_weight('wc4a', [3, 3, 3, 256, 256], 'conv4a', pynet),
            'wc5a': load_conv_3d_weight('wc5a', [3, 3, 3, 256, 256], 'conv5a', pynet),
            'wd1': load_fc_weight('wd1', [4096, 2048], 'fc6', pynet),
            'wd2': load_fc_weight('wd2', [2048, 2048], 'fc7', pynet),
            'out': load_fc_weight('wout', [2048, NUM_CLASSES], 'fc8', pynet)
            }
    biases = {
            'bc1': load_bias('bc1', [64], 'conv1a', pynet),
            'bc2': load_bias('bc2', [128], 'conv2a', pynet),
            'bc3a': load_bias('bc3a', [256], 'conv3a', pynet),
            'bc4a': load_bias('bc4a', [256], 'conv4a', pynet),
            'bc5a': load_bias('bc5a', [256], 'conv5a', pynet),
            'bd1': load_bias('bd1', [2048], 'fc6', pynet),
            'bd2': load_bias('bd2', [2048], 'fc7', pynet),
            'out': load_bias('bout', [NUM_CLASSES], 'fc8', pynet),
            }

  # generate the output directory
  model_save_dir = FLAGS.dst_location
  print("Saving model to: "+model_save_dir)
  if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

  # save the model to the output directory
  saver = tf.train.Saver(weights.values() + biases.values())
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.save(sess, os.path.join(model_save_dir, 'c3d_model'))