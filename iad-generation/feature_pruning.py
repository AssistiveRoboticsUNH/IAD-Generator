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


conv_variables = tf.trainable_variables()#[v for v in tf.trainable_variables() if len(v.shape) == 5]

#https://jacobgil.github.io/deeplearning/pruning-deep-learning
#https://stackoverflow.com/questions/43839431/tensorflow-how-to-replace-or-modify-gradient/43948872

'''
class Model:
	def __init__(self):
		self.activations = []
'''
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
			dy = tf.Print(dy, [dy], message="---->>>>dy_shape", summarize=10)
			dy = tf.Print(dy, [x], message="---->>>>x_shape", summarize=10)

			ranks = tf.math.multiply(x, dy)#tf.reduce_sum()
			ranks = tf.reduce_sum(ranks, axis=(0, 1, 2, 3))
			#ranks /= (tf.shape(x)[:-1])

			dy = tf.Print(dy, [tf.shape(x)[:-1]], message="norm_value", summarize=10)
			

			#values = \
			#values / (activation.size(0) * activation.size(2) * activation.size(3))


			dy = tf.Print(dy, [ranks], message="---->>>>rank_shape", summarize=10)

			
			#values = sum( activation * dy , dim=0 ) 
			return dy

		#self.activations.append(x)
		return tf.identity(x), grad

	# Convolution Layer
	conv1 = conv3d('conv1', input_ph, _weights['wc1'], _biases['bc1'])
	conv1 = rank_layer(conv1)
	conv1 = tf.nn.relu(conv1, 'relu1')
	pool1 = max_pool('pool1', conv1, k=1)

	# Convolution Layer
	conv2 = conv3d('conv2', pool1, _weights['wc2'], _biases['bc2'])
	#conv2 = rank_layer(conv2)
	conv2 = tf.nn.relu(conv2, 'relu2')
	pool2 = max_pool('pool2', conv2, k=2)

	# Convolution Layer
	conv3 = conv3d('conv3a', pool2, _weights['wc3a'], _biases['bc3a'])
	conv3 = tf.nn.relu(conv3, 'relu3a')
	pool3 = max_pool('pool3', conv3, k=2)

	# Convolution Layer
	conv4 = conv3d('conv4a', pool3, _weights['wc4a'], _biases['bc4a'])
	conv4 = tf.nn.relu(conv4, 'relu4a')
	pool4 = max_pool('pool4', conv4, k=2)

	# Convolution Layer
	conv5 = conv3d('conv5a', pool4, _weights['wc5a'], _biases['bc5a'])
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










'''


activations = model.generate_activation_map(input_placeholder, weights, biases)
class_op, softmax_op, pred_op = model.generate_full_model(input_placeholder, weights, biases)


'''
class_op, softmax_op, pred_op = generate_full_model(input_placeholder, weights, biases)

loss = tf.losses.sparse_softmax_cross_entropy(label_ph, pred_op)
opt = tf.train.AdamOptimizer()







for v in tf.all_variables():
	print(v)#print(v.name, v.shape)

gradients = tf.gradients(pred_op, input_placeholder)#opt.compute_gradients(loss, tf.all_variables())#tf.gradients(loss, conv_variables)
for g in gradients:
	print("g:", g)
'''
def compute_rank(activation, gradient):

	#point wise multiplication of each activation in the batch and it's gradient
	#for each actvation (that is an output of a convolution) we sum in all dimensions except the dimension of the outpu
	values = sum( activation * gradient , dim=0 ) 

	
	#normalize the rank by filter dimensions
	values = values / (activation.size(0) * activation.size(2) * activation.size(3))


	if activation_index not in filter_ranks?
		filter_ranks[activation_index] = float(activation.size(1).zero_like)

	filter_ranks[activation_index] += values

	return ranks
'''

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	saver.restore(sess, FLAGS.model_file)


	list_of_files_and_labels, max_frame_length = obtain_files(FLAGS.dataset_file)




	file, label = list_of_files_and_labels[0]
	raw_data, length_ratio = read_file(file, input_placeholder)

	gr = sess.run(gradients, feed_dict={input_placeholder: raw_data, label_ph:np.array([label])})
	

	print("printing gradients:")
	#print(gr)
	#for g in gr:
	#	print(g.shape)



#perform a forward and backward pass on our entire training dataset


