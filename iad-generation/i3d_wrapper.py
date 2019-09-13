import tensorflow as tf
import numpy as np

import os, cv2

import PIL.Image as Image

###################
# FILE IO         #
###################

def obtain_files(directory_file):
  # read a *.list file and get the file names and labels
  ifile = open(directory_file, 'r')
  line = ifile.readline()
  
  filenames, labels, max_length = [],[],0
  while len(line) != 0:

    line = line.split()

    filename, label = line
    filenames.append(filename)
    labels.append(label)
    file_length = len(os.listdir(filename))-1

    if(file_length > max_length):
      max_length = file_length

    line = ifile.readline()

  return zip(filenames, labels), max_length

def read_file(file, input_placeholder):
  # read a file and concatenate all of the frames
  # pad or prune the video to the given length

  print("reading: "+ file)

  # input_shape
  _, num_frames, h, w, ch = input_placeholder.get_shape()

  # read available frames in file
  img_data = []
  for r, d, f in os.walk(file):
    f.sort()
    limit = min(num_frames, len(f))
    
    for i in range(limit):
      filename = os.path.join(r, f[i])
      img = Image.open(filename)

      # resize and crop to fit input size
      #print(img.height, img.width)
      img = np.array(cv2.resize(np.array(img),(int((256.0/img.height) * img.width+1), 256))).astype(np.float32)
      #print(" after resize", img.shape)
      crop_x = int((img.shape[0] - h)/2)
      crop_y = int((img.shape[1] - w)/2)
      img = img[crop_x:crop_x+w, crop_y:crop_y+h,:] 

      img_data.append(np.array(img))

  img_data = np.array(img_data).astype(np.float32)
  length_ratio = len(img_data) / float(int(limit))

  # pad file to appropriate length
  buffer_len = int(num_frames) - len(img_data)
  img_data = np.pad(np.array(img_data), 
        ((0,buffer_len), (0,0), (0,0),(0,0)), 
        'constant', 
        constant_values=(0,0))
  img_data = np.expand_dims(img_data, axis=0)



  return img_data, length_ratio 


###################
# MODEL FUNCTIONS #
###################

import i3d

INPUT_DATA_SIZE = {"t": 64, "h":224, "w":224, "c":3}
CNN_FEATURE_COUNT = [64, 192, 480, 832, 1024]

def get_input_placeholder(batch_size, num_frames=INPUT_DATA_SIZE["t"]):
  # returns a placeholder for the C3D input
  return tf.placeholder(tf.float32, 
      shape=(batch_size, num_frames, INPUT_DATA_SIZE["h"], INPUT_DATA_SIZE["w"], INPUT_DATA_SIZE["c"]),
      name="c3d_input_ph")


def get_output_placeholder(batch_size):
  # returns a placeholder for the C3D output (currently unused)
  return tf.placeholder(tf.int32, 
      shape=(batch_size),
      name="c3d_label_ph")


def get_variables(num_classes=-1):
  '''Define all of the variables for the convolutional layers of the C3D model. 
  We ommit the FC layers as these layers are used to perform reasoning and do 
  not contain feature information '''

  variable_map = {}
  for variable in tf.global_variables():
    if variable.name.split('/')[0] == 'RGB' and 'Adam' not in variable.name.split('/')[-1] and variable.name.split('/')[2] != 'Logits':
      variable_map[variable.name.replace(':0', '')] = variable

  return variable_map

def generate_activation_map(input_ph):
  '''Generates the activation map for a given input from a specific depth
        -input_ph: the input placeholder, should have been defined using the 
          "get_input_placeholder" function
        -_weights: weights used to convolve the input, defined in the 
          "get_variables" function
        -_biases: biases used to convolve the input, defined in the 
          "get_variables" function
        -depth: the depth at which the activation map should be extracted (an 
          int between 0 and 4)
  '''

  # build I3D model
  is_training = tf.placeholder_with_default(False, shape=(), name="is_training_ph")
  with tf.variable_scope('RGB'):
    _, _, target_layers = i3d.InceptionI3d( num_classes=101,
                                  spatial_squeeze=True,
                                  final_endpoint='Logits')(input_ph, is_training)

  return target_layers


def load_model(input_ph):
  activation_maps = generate_activation_map(input_ph)

  variable_name_list = get_variables()
  saver = tf.train.Saver(variable_name_list.values())

  return activation_maps, saver
