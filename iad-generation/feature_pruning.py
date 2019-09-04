import tensorflow as tf 
import numpy as np 

import c3d as model
from file_reader import obtain_files, read_file

import argparse
parser = argparse.ArgumentParser(description='Generate IADs from input files')
#required command line args
parser.add_argument('model_file', help='the tensorflow ckpt file used to generate the IADs')
parser.add_argument('dataset_file', help='the *.list file than contains the ')

FLAGS = parser.parse_args()




#update model with ranking 

batch_size=1
input_placeholder = model.get_input_placeholder(batch_size)
label_ph = model.get_output_placeholder(batch_size)
	
# define model
weights, biases = model.get_variables(num_classes=13)
variable_name_dict = list( set(weights.values() + biases.values()))
saver = tf.train.Saver(variable_name_dict)

#https://jacobgil.github.io/deeplearning/pruning-deep-learning
#https://stackoverflow.com/questions/43839431/tensorflow-how-to-replace-or-modify-gradient/43948872

ranks_out = []
def generate_full_model(input_ph, _weights, _biases, depth=4, separate_conv_layers=True):
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

	@tf.custom_gradient
	def rank_layer(x):
		def grad(dy):

			#calculate the rank
			ranks = tf.math.multiply(x, dy)
			ranks = tf.reduce_sum(ranks, axis=(0, 1, 2, 3)) #combine spatial and temporal points together

			#normalize the rank by the input size
			norm_term = tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), tf.float32)
			ranks = tf.math.divide(ranks, norm_term) 

			ranks_out.append(ranks)
			return dy
		return tf.identity(x), grad

	# Convolution Layer
	conv1 = conv3d('conv1', input_ph, _weights['wc1'], _biases['bc1'])
	conv1 = rank_layer(conv1)
	conv1 = tf.nn.relu(conv1, 'relu1')
	pool1 = max_pool('pool1', conv1, k=1)

	# Convolution Layer
	conv2 = conv3d('conv2', pool1, _weights['wc2'], _biases['bc2'])
	conv2 = rank_layer(conv2)
	conv2 = tf.nn.relu(conv2, 'relu2')
	pool2 = max_pool('pool2', conv2, k=2)

	# Convolution Layer
	conv3 = conv3d('conv3a', pool2, _weights['wc3a'], _biases['bc3a'])
	conv3 = rank_layer(conv3)
	conv3 = tf.nn.relu(conv3, 'relu3a')
	pool3 = max_pool('pool3', conv3, k=2)

	# Convolution Layer
	conv4 = conv3d('conv4a', pool3, _weights['wc4a'], _biases['bc4a'])
	conv4 = rank_layer(conv4)
	conv4 = tf.nn.relu(conv4, 'relu4a')
	pool4 = max_pool('pool4', conv4, k=2)

	# Convolution Layer
	conv5 = conv3d('conv5a', pool4, _weights['wc5a'], _biases['bc5a'])
	conv5 = rank_layer(conv5)
	conv5 = tf.nn.relu(conv5, 'relu5a')
	pool5 = max_pool('pool5', conv5, k=2) 

	#flatten actviation map
	flat_am = tf.layers.flatten(pool5)

	#dense layers
	dense1 = tf.nn.bias_add(tf.matmul(flat_am, _weights['wd1']), _biases['bd1'])
	dense2 = tf.nn.bias_add(tf.matmul(dense1, _weights['wd2']), _biases['bd2'])
	out = tf.nn.bias_add(tf.matmul(dense2, _weights['out']), _biases['out'])

	softmax = tf.nn.softmax(out)
	classifcation = tf.argmax(softmax, axis = 1)

	return classifcation, softmax, out

ranks_out = ranks_out[::-1]


class_op, softmax_op, pred_op = generate_full_model(input_placeholder, weights, biases)
gradients = tf.gradients(pred_op, input_placeholder)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	saver.restore(sess, FLAGS.model_file)


	list_of_files_and_labels, max_frame_length = obtain_files(FLAGS.dataset_file)




	file, label = list_of_files_and_labels[0]
	raw_data, length_ratio = read_file(file, input_placeholder)

	r = sess.run([ranks_out], feed_dict={input_placeholder: raw_data})#, label_ph:np.array([label])})
	

	print("printing gradients:")
	print(r)
	#print(gr)
	#for g in gr:
	#	print(g.shape)



#perform a forward and backward pass on our entire training dataset


