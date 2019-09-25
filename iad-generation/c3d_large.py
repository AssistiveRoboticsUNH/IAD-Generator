import tensorflow as tf 
import numpy as np
import os, cv2
import PIL.Image as Image

############
# FILE IO 
############

def obtain_files(directory_file):
  # read a *.list file and get the file names and labels
  ifile = open(directory_file, 'r')
  line = ifile.readline()
  
  filenames, labels, max_length = [],[],0
  while len(line) != 0:
    filename, start_frame, label = line.split()

    if(filename not in filenames):
      filenames.append(filename)
      labels.append(int(label))

      if(len(os.listdir(filename)) > max_length):
        max_length = len(os.listdir(filename))

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
    limit = min(int(num_frames), len(f))
    
    for i in range(limit):
      filename = os.path.join(r, f[i])
      img = Image.open(filename)

      # resize and crop to fit input size
      img = np.array(cv2.resize(np.array(img),(171, 128))).astype(np.float32)
      crop_x = int((img.shape[0] - h)/2)
      crop_y = int((img.shape[1] - w)/2)
      img = img[crop_x:crop_x+w, crop_y:crop_y+h,:] 

      #subtract the mean value expression from the data and convert to RGB
      img -= np.array([90, 98, 102]) # mean values
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

      img_data.append(np.array(img))

  img_data = np.array(img_data).astype(np.float32)
  length_ratio = float(limit) / int(num_frames)

  # pad file to appropriate length
  buffer_len = int(num_frames) - len(img_data)
  img_data = np.pad(np.array(img_data), 
        ((0,buffer_len), (0,0), (0,0),(0,0)), 
        'constant', 
        constant_values=(0,0))
  img_data = np.expand_dims(img_data, axis=0)



  return img_data, length_ratio 

###########
# Model
###########


INPUT_DATA_SIZE = {"t": 16, "h":112, "w":112, "c":3}
CNN_FEATURE_COUNT = [64, 128, 256, 256, 256]

def get_input_placeholder(batch_size, num_frames=INPUT_DATA_SIZE["t"]):
  # returns a placeholder for the C3D input
  return tf.placeholder(tf.float32, 
      shape=(batch_size, num_frames, INPUT_DATA_SIZE["h"], INPUT_DATA_SIZE["w"], INPUT_DATA_SIZE["c"]),
      name="c3d_input_ph")


def get_output_placeholder(batch_size):
  # returns a placeholder for the C3D output (currently unused)
  return tf.placeholder(tf.float32, 
      shape=(batch_size),
      name="c3d_label_ph")


def get_variables(num_classes=-1):
  '''Define all of the variables for the convolutional layers of the C3D model. 
  We ommit the FC layers as these layers are used to perform reasoning and do 
  not contain feature information '''

  def gen_var(ori_name, shape):
    return tf.get_variable(name=ori_name, shape=shape, initializer=tf.zeros_initializer)

  with tf.variable_scope('var_name') as var_scope:
    weights = {
              'wc1': gen_var('wc1', [3, 3, 3, 3, 64]),
              'wc2': gen_var('wc2', [3, 3, 3, 64, 128]),
              'wc3a': gen_var('wc3a', [3, 3, 3, 128, 256]),
              'wc3b': gen_var('wc3b', [3, 3, 3, 256, 256]),
              'wc4a': gen_var('wc4a', [3, 3, 3, 256, 512]),
              'wc4b': gen_var('wc4b', [3, 3, 3, 512, 512]),
              'wc5a': gen_var('wc5a', [3, 3, 3, 512, 512]),
              'wc5b': gen_var('wc5b', [3, 3, 3, 512, 512])
              }
    biases = {
              'bc1': gen_var('bc1', [64]),
              'bc2': gen_var('bc2', [128]),
              'bc3a': gen_var('bc3a', [256]),
              'bc3b': gen_var('bc3b', [256]),
              'bc4a': gen_var('bc4a', [512]),
              'bc4b': gen_var('bc4b', [512]),
              'bc5a': gen_var('bc5a', [512]),
              'bc5b': gen_var('bc5b', [512])
              }

    if(num_classes > 0):
      weights['wd1'] = gen_var('wd1', [4096, 2048])
      weights['wd2'] = gen_var('wd2', [2048, 2048])
      weights['out'] = gen_var('wout', [2048, num_classes])

      biases['bd1'] = gen_var('bd1', [2048])
      biases['bd2'] = gen_var('bd2', [2048])
      biases['out'] = gen_var('bout', [num_classes])

  return weights, biases

def generate_activation_map(input_ph, _weights, _biases, depth=4, separate_conv_layers=True):
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

  def conv3d(name, l_input, w, b):
    # performs a 3d convolution
    return tf.nn.bias_add(tf.nn.conv3d(l_input, w, strides=[1, 1, 1, 1, 1], padding='SAME'),b)

  def max_pool(name, l_input, k):
    # performs a 2x2 max pool operation in 3 dimensions
    return tf.nn.max_pool3d(l_input, ksize=[1, k, 2, 2, 1], strides=[1, k, 2, 2, 1], padding='SAME', name=name)

  # Convolution Layer
  conv1 = conv3d('conv1', input_ph, _weights['wc1'], _biases['bc1'])
  conv1 = tf.nn.relu(conv1, 'relu1')
  pool1 = max_pool('pool1', conv1, k=1)

  # Convolution Layer
  conv2 = conv3d('conv2', pool1, _weights['wc2'], _biases['bc2'])
  conv2 = tf.nn.relu(conv2, 'relu2')
  pool2 = max_pool('pool2', conv2, k=2)

  # Convolution Layer
  conv3 = conv3d('conv3a', pool2, _weights['wc3a'], _biases['bc3a'])
  conv3 = tf.nn.relu(conv3, 'relu3a')
  conv3 = conv3d('conv3b', conv3, _weights['wc3b'], _biases['bc3b'])
  conv3 = tf.nn.relu(conv3, 'relu3b')
  pool3 = max_pool('pool3', conv3, k=2)

  # Convolution Layer
  conv4 = conv3d('conv4a', pool3, _weights['wc4a'], _biases['bc4a'])
  conv4 = tf.nn.relu(conv4, 'relu4a')
  conv4 = conv3d('conv4b', conv4, _weights['wc4b'], _biases['bc4b'])
  conv4 = tf.nn.relu(conv4, 'relu4b')
  pool4 = max_pool('pool4', conv4, k=2)

  # Convolution Layer
  conv5 = conv3d('conv5a', pool4, _weights['wc5a'], _biases['bc5a'])
  conv5 = tf.nn.relu(conv5, 'relu5a')
  conv5 = conv3d('conv5b', conv5, _weights['wc5b'], _biases['bc5b'])
  conv5 = tf.nn.relu(conv5, 'relu5b')
  pool5 = max_pool('pool5', conv5, k=2)

  if(separate_conv_layers):

    # an array of convolution layers to select from
    layers = [conv1, conv2, conv3, conv4, conv5]

    return layers
  return pool5

def load_model(input_ph):
  weights, biases = get_variables()
  variable_name_list = list( set(weights.values() + biases.values()))
  saver = tf.train.Saver(variable_name_list)

  activation_maps = generate_activation_map(input_ph, weights, biases)

  return activation_maps, saver


def generate_full_model(input_ph, _weights, _biases):

  pool5 = generate_activation_map(input_ph, _weights, _biases, separate_conv_layers=False)[-1]

  #flatten actviation map
  flat_am = tf.layers.flatten(pool5)

  #dense layers
  dense1 = tf.nn.bias_add(tf.matmul(flat_am, _weights['wd1']), _biases['bd1'])
  dense2 = tf.nn.bias_add(tf.matmul(dense1, _weights['wd2']), _biases['bd2'])
  out = tf.nn.bias_add(tf.matmul(dense2, _weights['out']), _biases['out'])

  softmax = tf.nn.softmax(out)
  classifcation = tf.argmax(softmax, axis = 1)

  return classifcation, softmax, out

