# ensemble.py
#
# This program takes as input a set of classified IADS...
#
# Usage:
# ensemble.py <train|test> <model name> <dataset name> <num_classes> <dataset size>
#   if train is specified, the model name will be the name of the saved model
#   if test is specified, the model name will be the name of the loaded model






import os, random

import tensorflow as tf
import numpy as np

import sys
sys.path.append("../iad-generation/")
from feature_rank_utils import get_top_n_feature_indexes
from csv_utils import read_csv
import tf_utils

#import c3d as model
#import c3d_large as model
#import i3d_wrapper as model
import rank_i3d as model

get_input_shape = lambda num_features, pad_length: \
				   [(min(  64, num_features), pad_length/2), 
					(min( 192, num_features), pad_length/2), 
					(min( 480, num_features), pad_length/2), 
					(min( 832, num_features), pad_length/4), 
					(min(1024, num_features), pad_length/8)]

def get_data(ex, layer, pruning_indexes, window_size):

	# open the IAD
	f = np.load(ex['iad_path_'+str(layer)])
	d, z = f["data"], f["length"]

	# prune unused indexes
	if(pruning_indexes != None):
		idx = pruning_indexes[layer]
		d = d[idx]

	# modify data to desired window size
	pading_length = window_size - (z%window_size)
	d = np.pad(d, [[0,0],[0,pading_length]], 'constant', constant_values=0)

	# split the input into chunks of the given window size
	d = np.split(d, d.shape[1]/window_size, axis=1)
	d = np.stack(d)

	return d, ex['label']

def get_batch_data(dataset, model_num, pruning_indexes, input_shape, batch_size, batch_indexes=None, sliding_window=False):

	def get_batch_at_layer(layer, batch_indexes):
		data, labels = [],[]

		for b_idx in batch_indexes:

			# open example and prepare data
			d, l = get_data(dataset[b_idx], layer, pruning_indexes, input_shape[layer][1])

			# randomly select one of the windows in the data
			if(sliding_window):
				#print("d.shape:", d.shape)
				w_idx = random.randint(0, d.shape[0]-1)
			else:
				w_idx = 0 # replace if using sliding window: 

			# add values to list
			#print("w_idx:", w_idx)
			data.append(d[w_idx])
			labels.append(l)

		return np.array(data), np.array(labels)

	if(batch_indexes == None):
		batch_indexes = np.random.randint(0, len(dataset), size=batch_size)

	if (model_num < 5):
		return get_batch_at_layer(model_num, batch_indexes)
	else:
		data = []
		for layer in range(5):
			d, labels = get_batch_at_layer(layer, batch_indexes)
			w_idx = 0
			d = d.reshape(batch_size, -1, 1)
			data.append(d)
		data = np.concatenate(data, axis=1)

	return data, labels

def get_stack_data(dataset, model_num, pruning_indexes, input_shape, batch_size, batch_indexes=None, sliding_window=False):
	def get_batch_at_layer(layer, batch_indexes):
		data, labels = [],[]

		for b_idx in batch_indexes:

			# open example and prepare data
			d, l = get_data(dataset[b_idx], layer, pruning_indexes, input_shape[layer][1])

			# add values to list
			data.append(d)
			labels.append(l)

		return np.array(data), np.array(labels)

	if(batch_indexes == None):
		batch_indexes = np.random.randint(0, len(dataset), size=batch_size)

	if (model_num < 5):
		return get_batch_at_layer(model_num, batch_indexes)
	else:
		data = []
		for layer in range(5):
			d, labels = get_batch_at_layer(layer, batch_indexes)
			num_win = d.shape[1]

			d = d.reshape(batch_size, num_win, -1, 1)
			data.append(d)

		for i in data:
			print(i.shape)
		data = np.concatenate(data, axis=2)

	return data, labels

##############################################
# Model Structure
##############################################

def model_def(num_classes, input_shape, model_num, alpha):
	"""Create the tensor operations to be used in training and testing, stored in a dictionary."""

	def conv_model(input_ph):
		"""Return a convolutional model."""
		top = input_ph

		# hidden layers
		num_filters = 32
		filter_width = 4
		conv1 = tf.layers.conv2d(
			inputs=input_layer,
			filters=num_filters,
			kernel_size=[1, filter_width],
			padding="valid", 
			activation=tf.nn.leaky_relu)
		top = tf.layers.flatten(top)
		top = tf.layers.dense(inputs=top, units=2048, activation=tf.nn.leaky_relu)
		top = tf.layers.dropout(top, rate=0.5, training=ph["train"])

		# output layers
		return tf.layers.dense(inputs=top, units=num_classes)

	def dense_model(input_ph):
		"""Return a single layer softmax model."""
		top = input_ph

		# hidden layers
		top = tf.layers.flatten(top)
		top = tf.layers.dense(inputs=top, units=2048, activation=tf.nn.leaky_relu)
		top = tf.layers.dropout(top, rate=0.5, training=ph["train"])

		# output layers
		return tf.layers.dense(inputs=top, units=num_classes)

	# Placeholders
	print("Model Shape:", model_num, input_shape[model_num][0], input_shape[model_num][1])

	ph = {
		"x_"+str(model_num): tf.placeholder(tf.float32, 
			shape=(None, input_shape[model_num][0], input_shape[model_num][1])),
		"y": tf.placeholder(tf.int32, shape=(None)),
		"train": tf.placeholder(tf.bool)
	}

	# Logits
	# input layers [batch_size, h, w, num_channels]
	input_layer = tf.reshape(ph["x_" + str(model_num)], [-1, input_shape[model_num][0], input_shape[model_num][1], 1])
	if(model_num < 3):
		logits = conv_model(input_layer)
	else:
		logits = dense_model(input_layer)

	# Predict
	softmax = tf.nn.softmax(logits, name="softmax_tensor")
	class_pred = tf.argmax(input=logits, axis=1, output_type=tf.int32)

	# Train
	loss = tf.losses.sparse_softmax_cross_entropy(labels=ph["y"], logits=logits)
	train_op = tf.train.AdamOptimizer(learning_rate=alpha).minimize(
		loss=loss,
		global_step=tf.train.get_global_step()
	)

	# Test
	correct_pred = tf.equal(class_pred, ph["y"])
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	# the class predictions across all of the models
	all_pred = tf.squeeze(tf.argmax(softmax, axis=1, output_type=tf.int32))

	ops = {
		'train': [train_op, loss, accuracy],
		'model_sftmx': softmax,
		'model_preds': all_pred
	}
	
	return ph, ops

def model_consensus(confidences):
	"""Generate a weighted average over the composite models"""
	confidence_discount_layer = [0.5, 0.7, 0.9, 0.9, 0.9, 1.0]

	confidences = confidences * confidence_discount_layer

	confidences = np.sum(confidences, axis=(2,3))

	return np.argmax(confidences, axis=1)

##############################################
# Train/Test Functions
##############################################

def train_model(model_dirs, num_classes, train_data, test_data, pruning_indexes, num_features, window_size, batch_size, alpha, epochs, sliding_window):

	# get the shape of the flattened and merged IAD and append
	input_shape = get_input_shape(num_features, window_size)
	input_shape += [(np.sum([shape[0]*shape[1] for shape in input_shape]), 1)]

	for model_num in range(6):

		#define network
		ph, ops = model_def(num_classes, input_shape, model_num, alpha)
		saver = tf.train.Saver()
		
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			sess.run(tf.local_variables_initializer())

			sess.graph.finalize()

			# train the network
			num_iter = epochs * len(train_data) / batch_size
			for i in range(num_iter):
			
			# setup training batch

				data, label = get_batch_data(train_data, model_num, pruning_indexes, input_shape, batch_size, sliding_window=sliding_window)
				feed_dict = { ph["x_"+str(model_num)]: data, ph["y"]: label,  ph["train"]: True }

				out = sess.run(ops["train"], feed_dict=feed_dict)

				# print out every 2K iterations
				if i % 2000 == 0:
					print("step: ", str(i) + '/' + str(num_iter))
					
					# evaluate test network
					data, label = get_batch_data(test_data, model_num, pruning_indexes, input_shape, batch_size, sliding_window=sliding_window)
					feed_dict = { ph["x_"+str(model_num)]: data, ph["y"]: label,  ph["train"]: False }

					correct_prediction = sess.run([ops['model_preds']], feed_dict=feed_dict)
					correct, total = np.sum(correct_prediction[0] == label), len(correct_prediction[0] == label)

					print("model_num: {0}, test_accuracy: {1}, correct: {2}, total: {3}".format(model_num, correct / float(total), correct, total))

			# save the model
			save_name = model_dirs[model_num]+'/model'
			saver.save(sess, save_name)
			print("Final model saved in %s" % save_name)
		tf.reset_default_graph()

def test_model(iad_model_path, model_dirs, num_classes, test_data, pruning_indexes, num_features, window_size, sliding_window):

	# get the shape of the flattened and merged IAD and append
	input_shape = get_input_shape(num_features, window_size)
	input_shape += [(np.sum([shape[0]*shape[1] for shape in input_shape]), 1)]



	#variables for determining model accuracy
	model_accuracy = np.zeros([ 6, 2], dtype=np.int32)

	#variables for determining class accuracy
	class_accuracy = np.zeros([ num_classes, 2], dtype=np.int32)

	#variables used to generate consensus value
	aggregated_confidences, aggregated_labels = [], []
	for i in range(len(test_data)):
		aggregated_confidences.append([])



	for model_num in range(6):

		#define network
		ph, ops = model_def(num_classes, input_shape, model_num, 1e-4)
		saver = tf.train.Saver()

		with tf.Session() as sess:
			# restore the model
			tf_utils.restore_model(sess, saver, model_dirs[model_num])

			num_iter = len(test_data)
			for i in range(num_iter):

				data, label = get_stack_data(test_data, model_num, pruning_indexes, input_shape, 1, batch_indexes=[i], sliding_window=sliding_window)
				data = data[0]

				#print("data_shape:", data.shape)

				if(sliding_window):
					num_win = len(data)
				else:
					num_win = 1

				for w_idx in range(num_win): # replace with len(data[0]) if using sliding window
					print("input:", data[w_idx].shape, label.shape)
					feed_dict = { ph["x_"+str(model_num)]: np.expand_dims(data[w_idx], axis = 0), ph["y"]: label,  ph["train"]: False }

					confidences, predictions = sess.run([ 
							ops['model_sftmx'], ops['model_preds']], 
							feed_dict=feed_dict)
					print("output:", confidences.shape)

					# append confidences for evaluating consensus model
					aggregated_confidences[i].append(confidences)
					if(model_num == 0):
						aggregated_labels.append(label)

					# update model accuracy
					if(predictions == label):
						model_accuracy[model_num, 0] += 1
					model_accuracy[model_num, 1] += 1

		tf.reset_default_graph()

	for ag in aggregated_confidences:
		print(np.array(ag).shape)


	# generate wighted sum for ensemble of models 
	print("aggregated_confidences", np.array(aggregated_confidences).shape)
	aggregated_confidences = np.transpose(np.array(aggregated_confidences), [0, 3, 2, 1])
	ensemble_prediction = model_consensus(aggregated_confidences)
	aggregated_labels = np.array(aggregated_labels).reshape(-1)

	#print("ensemble_prediction", ensemble_prediction)
	#print("aggregated_labels", aggregated_labels)
	#print(np.sum(ensemble_prediction == aggregated_labels), len(aggregated_labels))

	for i, label in enumerate(aggregated_labels):
		if(ensemble_prediction[i] == label):
			class_accuracy[label, 0] += 1
		class_accuracy[label, 1] += 1    
				
	# print partial model's cummulative accuracy
	ofile = open(os.path.join(iad_model_path, "model_accuracy.txt"), 'w')
	print("Model accuracy: ")
	for model_num in range(6):
		print("{:d}\t{:4.6f}".format(model_num, model_accuracy[model_num, 0] / float(model_accuracy[model_num, 1])) )
		ofile.write("{:d}\t{:4.6f}\n".format(model_num, model_accuracy[model_num, 0] / float(model_accuracy[model_num, 1])) )

	# print ensemble cummulative accuracy
	print("FINAL\t{:4.6f}".format( np.sum(ensemble_prediction == aggregated_labels) / float(len(aggregated_labels)) ) )
	ofile.write("FINAL\t{:4.6f}\n".format( np.sum(ensemble_prediction == aggregated_labels) / float(len(aggregated_labels)) ) )
	ofile.close()

	# save per-class accuracy
	np.save(os.path.join(iad_model_path, "class_accuracy.npy"),  class_accuracy[:, 0] / class_accuracy[:, 1] )


def main(model_type, dataset_dir, csv_filename, num_classes, operation, dataset_id, model_filename, 
		window_size, epochs, batch_size, alpha, 
		feature_retain_count, gpu, sliding_window):

	# optional - specify the CUDA device to use for GPU computation
	# comment this line out if you wish to use all CUDA-capable devices
	os.environ["CUDA_VISIBLE_DEVICES"] = gpu

	# Setup file IO
	iad_data_path = os.path.join(dataset_dir, 'iad')
	model_id_path = os.path.join('iad_model_'+str(window_size), 'model_'+str(25*dataset_id))
	iad_model_path = os.path.join(dataset_dir, model_id_path)

	model_dirs = []
	for model_num in range(6):
		separate_model_dir = os.path.join(iad_model_path, 'model_'+str(model_num))
		model_dirs.append(separate_model_dir)

		if(not os.path.exists(separate_model_dir)):
			os.makedirs(separate_model_dir)

	

	try:
		csv_contents = read_csv(csv_filename)
	except:
		print("Cannot open CSV file: "+ csv_filename)

	train_data = [ex for ex in csv_contents if ex['dataset_id'] >  0]
	test_data  = [ex for ex in csv_contents if ex['dataset_id'] == 0] 

	for ex in csv_contents:
		file_location = os.path.join(ex['label_name'], ex['example_id'])
		for layer in range(5):
			iad_file = os.path.join(iad_data_path, file_location+"_"+str(layer)+".npz")
			assert os.path.exists(iad_file), "Cannot locate IAD file: "+ iad_file
			ex['iad_path_'+str(layer)] = iad_file

	train_data = train_data[:5]
	test_data = test_data[:5]

	# Determine features to prune
	pruning_keep_indexes = None
	if(feature_retain_count and dataset_id):
		ranking_file = os.path.join(iad_data_path, "feature_ranks_"+str(dataset_id * 25)+".npz")
		assert os.path.exists(ranking_file), "Cannot locate Feature Ranking file: "+ ranking_file
		pruning_keep_indexes = get_top_n_feature_indexes(ranking_file, feature_retain_count)

	# Begin Training/Testing
	if(operation == "train"):
		#model_filename, num_classes, train_data, test_data, pruning_indexes, window_size, batch_size
		train_model(model_dirs, num_classes, train_data, test_data, pruning_keep_indexes, feature_retain_count, window_size, batch_size, alpha, epochs, sliding_window)
	elif(operation == "test"):
		test_model (iad_model_path, model_dirs, num_classes, test_data, pruning_keep_indexes, feature_retain_count, window_size, sliding_window)
	else:
		print('Operation parameter must be either "train" or "test"')



if __name__ == "__main__":
	"""Determine if the user has specified training or testing and run the appropriate function."""
	
	import argparse
	parser = argparse.ArgumentParser(description='Generate IADs from input files')

	#required command line args
	parser.add_argument('model_type', help='the type of model to use: I3D')

	parser.add_argument('dataset_dir', help='the directory where the dataset is located')
	parser.add_argument('csv_filename', help='a csv file denoting the files in the dataset')

	parser.add_argument('num_classes', type=int, help='the number of classes in the dataset')
	parser.add_argument('operation', help='"train" or "test"')
	parser.add_argument('dataset_id', nargs='?', type=int, help='the dataset_id used to train the network. Is used in determing feature rank file')
	parser.add_argument('window_size', nargs='?', type=int, help='the maximum length video to convert into an IAD')

	parser.add_argument('--sliding_window', type=bool, default=False, help='.list file containing the test files')
	parser.add_argument('--model_filename', default="model", help='the checkpoint file to use with the model')
	parser.add_argument('--epochs', nargs='?', type=int, default=30, help='the maximum length video to convert into an IAD')
	parser.add_argument('--batch_size', nargs='?', type=int, default=15, help='the maximum length video to convert into an IAD')
	parser.add_argument('--alpha', nargs='?', type=int, default=1e-4, help='the maximum length video to convert into an IAD')
	parser.add_argument('--feature_retain_count', nargs='?', type=int, default=10000, help='the number of features to remove')
	
	parser.add_argument('--gpu', default="0", help='gpu to run on')

	FLAGS = parser.parse_args()

	main(FLAGS.model_type, 
		FLAGS.dataset_dir, 
		FLAGS.csv_filename, 
		FLAGS.num_classes, 
		FLAGS.operation, 
		FLAGS.dataset_id, 
		FLAGS.model_filename, 
		FLAGS.window_size, 
		FLAGS.epochs,
		FLAGS.batch_size,
		FLAGS.alpha,
		FLAGS.feature_retain_count,
		FLAGS.gpu,
		FLAGS.sliding_window)

	

