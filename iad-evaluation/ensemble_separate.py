# ensemble.py
#
# This program takes as input a set of classified IADS...
#
# Usage:
# ensemble.py <train|test> <model name> <dataset name> <num_classes> <dataset size>
#   if train is specified, the model name will be the name of the saved model
#   if test is specified, the model name will be the name of the loaded model


'''
import argparse
import csv
import numpy as np
import os
import sys
import tensorflow as tf

import time
import random

parser = argparse.ArgumentParser(description="Ensemble model processor")
parser.add_argument('model', help='model to save (when training) or to load (when testing)')
parser.add_argument('num_classes', type=int, help='the number of classes in the dataset')
parser.add_argument('iad_dir', help='location of the generated IADs')
parser.add_argument('prefix', help='"train" or "test"')

parser.add_argument('window_length', type=int, help='the size of the window. If left unset then the entire IAD is fed in at once. \
																	If the window is longer than the video then we pad to the IADs to that length')

#feature pruning command line args
parser.add_argument('--feature_rank_file', nargs='?', type=str, default=None, help='a file containing the rankings of the features')
parser.add_argument('--feature_retain_count', nargs='?', type=int, default=-1, help='the number of features to remove')

parser.add_argument('--gpu', default="0", help='gpu to run on')
parser.add_argument('--v', default=False, help='verbose')

args = parser.parse_args()
'''
from csv_utils import read_csv
import tf_utils

#import c3d as model
#import c3d_large as model
#import i3d_wrapper as model
import rank_i3d as model

import os, sys

import tensorflow as tf
import numpy as np



'''
input_shape_c3d = [(64, args.window_length), (128, args.window_length), (256, args.window_length/2), (256, args.window_length/4), (256, args.window_length/8)]
input_shape_c3d_large = [(64, args.window_length), (128, args.window_length), (256, args.window_length/2), (512, args.window_length/4), (512, args.window_length/8)]
input_shape_i3d = [(64, args.window_length/2), (192, args.window_length/2), (480, args.window_length/2), (832, args.window_length/4), (1024, args.window_length/8)]
'''

get_input_shape = lambda num_features, pad_length: \
				   [(min(  64, num_features), pad_length/2), 
					(min( 192, num_features), pad_length/2), 
					(min( 480, num_features), pad_length/2), 
					(min( 832, num_features), pad_length/4), 
					(min(1024, num_features), pad_length/8)]

##############################################
# Parameters
##############################################

# trial-specific parameters
# EPOCHS is the number of training epochs to complete
# ALPHA is the learning rate

#TRAIN_PREFIX = "train"
#TEST_PREFIX = "test"

##############################################
# File IO
##############################################
"""
def parse_iadlist(iad_dir, prefix):
	'''Opena dn parse a .iadlist file'''

	iadlist_filename = os.path.join(iad_dir, prefix+".iadlist")

	try:
		ifile = open (iadlist_filename, 'r')
	except:
		print("File doesn't exist: "+ iadlist_filename)
		sys.exit(1)
	
	iad_groups = []

	line = ifile.readline()
	while(len(line) > 0):
		filename_group = [os.path.join(iad_dir, f) for f in line.split()]
		iad_groups.append(filename_group)
		line = ifile.readline()
	return iad_groups
"""

def get_data(ex, layer, pruning_indexes, window_size):

	# open the IAD
	f = np.load(ex['iad_path_'+str(layer)])
	d, z = f["data"], f["length"]

	# prune unused indexes
	if(pruning_keep_indexes != None):
		idx = pruning_keep_indexes[layer]
		d = d[idx]

	# modify data to desired window size
	pading_length = window_size - (z%window_size)
	d = np.pad(d, [[0,0],[0,pading_length]], 'constant', constant_values=0)

	# split the input into chunks of the given window size
	d = np.split(d, d.shape[1]/window_size, axis=1)
	d = np.stack(d)

	return d, ex['label']

def get_batch_data(dataset, model_num, pruning_indexes, window_size, batch_size):

	def get_batch_at_layer(layer, batch_indexes):
		data, labels = [],[]
		layer = model_num

		for b_idx in batch_indexes:

			# open example and prepare data
			d, l = get_data(dataset[b_idx], layer, pruning_indexes, window_size)

			# randomly select one of the windows in the data
			w_idx = random.randint(0, d[0].shape[0]-1)

			# add values to list
			data.append(d[w_idx])
			labels.append(l)

		return np.array(data), np.array(labels)



	batch_indexes = np.random.randint(0, len(dataset), size=batch_size)

	if (model_num < 5):
		return get_batch_at_layer(model_num, batch_indexes)
	else:
		data = []
		for layer in range(5):
			d, labels = get_batch_at_layer(layer, batch_indexes)
			data.append(d)
		data = [x.reshape(batch_size, -1, 1) for x in file_data]

	return data, labels

"""
def open_and_org_file(filename_group, pruning_keep_indexes=None):
	'''Open all of the files in the filename_group (an array of filenames). Then format and shape the IADs within'''
	file_data = []

	#join the separate IAD layers
	for layer, filename in enumerate(filename_group):
		
		f = np.load(filename)
		d, label, z = f["data"], f["label"], f["length"]

		# prune irrelevant features
		if(pruning_keep_indexes != None):
			idx = pruning_keep_indexes[layer]
			d = d[idx]

		#break d in to chuncks of window size
		window_size = input_shape[layer][1]
		pad_length = window_size - (z%window_size)
		d = np.pad(d, [[0,0],[0,pad_length]], 'constant', constant_values=0)
		d = np.split(d, d.shape[1]/window_size, axis=1)
		d = np.stack(d)
		file_data.append(d)

	#append the flattened and merged IAD
	flat_data = np.concatenate([x.reshape(x.shape[0], -1, 1) for x in file_data], axis = 1)
	file_data.append(flat_data)

	return file_data, np.array([int(label)])

def get_data_train(iad_list, pruning_keep_indexes=None):
	'''Randomly select a batch of IADs, if using windows smaller than the input the select a window that will capture the data'''
	
	batch_data = []
	for i in range(6):
		batch_data.append([])
	batch_label = []

	#select files randomly
	batch_indexs = np.random.randint(0, len(iad_list), size=BATCH_SIZE)

	for index in batch_indexs:
		file_data, label = open_and_org_file(iad_list[index], pruning_keep_indexes)
		
		#randomly select a window from the example
		win_index = random.randint(0, file_data[0].shape[0]-1)
		for layer in range(len(file_data)):
			batch_data[layer].append(file_data[layer][win_index])

		batch_label.append(label)

	for i in range(6):
		batch_data[i] = np.array(batch_data[i])

	return batch_data, np.array(batch_label).reshape(-1)

def get_data_test(iad_list, index, pruning_keep_indexes=None):
	return open_and_org_file(iad_list[index], pruning_keep_indexes)
"""
##############################################
# Model Structure
##############################################

def model_def(num_classes, data_shapes, layer=-1):
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
	ph = {
		"y": tf.placeholder(tf.int32, shape=(None)),
		"train": tf.placeholder(tf.bool)
	}

	for c3d_depth in range(6):
		ph["x_" + str(c3d_depth)] = tf.placeholder(
			tf.float32, shape=(None, data_shapes[c3d_depth][0], data_shapes[c3d_depth][1])
		)

	# for each model generate tensor ops

	# Logits
	# input layers [batch_size, h, w, num_channels]
	input_layer = tf.reshape(ph["x_" + str(layer)], [-1, data_shapes[layer][0], data_shapes[layer][1], 1])
	if(c3d_depth < 3):
		logits = conv_model(input_layer)
	else:
		logits = dense_model(input_layer)

	# Predict
	softmax = tf.nn.softmax(logits, name="softmax_tensor")
	class_pred = tf.argmax(input=logits, axis=1, output_type=tf.int32)

	# Train
	loss = tf.losses.sparse_softmax_cross_entropy(labels=ph["y"], logits=logits)
	train_op = tf.train.AdamOptimizer(learning_rate=ALPHA).minimize(
		loss=loss,
		global_step=tf.train.get_global_step()
	)

	# Test
	correct_pred = tf.equal(class_pred, ph["y"])
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	# the softmax values across all of the models
	print("softmax.get_shape(): ", softmax.get_shape())
	#all_sftmx = tf.transpose(softmax, [1, 2, 0])

	# the class predictions across all of the models
	#all_pred = tf.transpose(all_sftmx, [0, 2, 1])
	all_pred = tf.squeeze(tf.argmax(softmax, axis=1, output_type=tf.int32))

	ops = {
		'train': [train_op , loss, accuracy],
		'model_sftmx': softmax,
		'model_preds': all_pred
	}
	
	return ph, ops

def model_consensus(confidences):
	"""Generate a weighted average over the composite models"""
	confidence_discount_layer = [0.5, 0.7, 0.9, 0.9, 0.9, 1.0]
	print("conf_strt:", confidences.shape)

	confidences = confidences * confidence_discount_layer
	print("conf_shape:", confidences.shape)

	confidences = np.sum(confidences, axis=(2,3))
	print("conf_sum:", confidences.shape)

	maxes = np.argmax(confidences, axis=1)
	print("conf_max:", maxes.shape)
	return maxes

##############################################
# Train/Test Functions
##############################################

def train_model(model_filename, num_classes, train_data, test_data, pruning_indexes, window_size, batch_size):

	# get the shape of the flattened and merged IAD and append
	input_shape = get_input_shape(FLAGS.feature_retain_count, FLAGS.pad_length)
	input_shape += [(np.sum([shape[0]*shape[1] for shape in input_shape]), 1)]

	for model_num in range(6):

		#define network
		ph, ops = model_def(num_classes, input_shape, layer=model_num)
		saver = tf.train.Saver()
		
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			sess.run(tf.local_variables_initializer())

			sess.graph.finalize()

			# train the network
			num_iter = FLAGS.EPOCHS * len(train_data) / FLAGS.BATCH_SIZE
			for i in range(num_iter):
			# setup training batch

				data, label = get_batch_data(train_data, model_num, pruning_indexes, window_size, batch_size)
				feed_dict = { ph["x_"+str(model_num)]: data[d], ph["y"]: label,  ph["train"]: True }

				out = sess.run(ops["train"], feed_dict=batch_data)

				# print out every 2K iterations
				if i % 2000 == 0:
					print("step: ", str(i) + '/' + str(num_iter))
					
					# evaluate test network
					data, label = get_batch_data(test_data, model_num, pruning_indexes, window_size, batch_size)
					feed_dict = { ph["x_"+str(model_num)]: data[d], ph["y"]: label,  ph["train"]: False }

					correct_prediction = sess.run([ops['model_preds']], feed_dict=batch_data)
					correct, total = np.sum(correct_prediction[0] == label), len(correct_prediction[0] == label)

					print("test_accuracy: {0}, correct: {1}, total: {2}".format(correct / float(total), correct, total))

			# save the model
			save_name = model_name+'_'+str(layer)
			saver.save(sess, save_name)
			print("Final model saved in %s" % save_name)
		tf.reset_default_graph()

def test_model(model_name, num_classes, test_data, pruning_keep_indexes=None):

	# get the shape of the flattened and merged IAD and append
	test_batch_size = 1
	data_shape = input_shape + [(np.sum([shape[0]*shape[1] for shape in input_shape]), 1)]

	correct, total = 0, 0
	model_correct, model_total = [0]*6, [0]*6

	correct_class = np.zeros(num_classes, dtype=np.float32)
	total_class = np.zeros(num_classes, dtype=np.float32)

	aggregated_confidences = []
	aggregated_labels = []

	for layer in range(6):

		#define network
		ph, ops = model_def(num_classes, data_shape, layer=layer)
		saver = tf.train.Saver()

		with tf.Session() as sess:
			# restore the model
			try:
				saver.restore(sess, model_name+"_"+str(layer))
				print("Model restored from %s" % model_name+"_"+str(layer))
			except:
				print("Failed to load model")

			num_iter = len(test_data)
			for i in range(num_iter):
				data, label = get_data_test(test_data, i, pruning_keep_indexes)
				label = int(label[0])
				
				if(layer == 0):
					aggregated_confidences.append([])

				batch_data = {}
				batch_data[ph["y"]] = label
				batch_data[ph["train"]] = False

				for j in range(1):#len(data[0])):
					for d in range(6):
						batch_data[ph["x_" + str(d)]] = np.expand_dims(data[d][j], axis = 0)

					confidences, predictions = sess.run([
						ops['model_sftmx'], 
						ops['model_preds'], 
					], feed_dict=batch_data)

					aggregated_confidences[i].append(confidences)
					if(layer == 0):
						aggregated_labels.append(label)

					#print("predictions:", predictions)

					if(predictions == label):
						model_correct[layer] += 1
					model_total[layer] += 1
		tf.reset_default_graph()

	aggregated_confidences=np.array(aggregated_confidences)
	aggregated_confidences = np.transpose(aggregated_confidences, [0, 3, 2, 1])
	ensemble_prediction = model_consensus(aggregated_confidences)

	print("aggregated_labels: ", aggregated_labels)
	print("aggregated_confidences: ", aggregated_confidences.shape)

	for i in range(len(aggregated_confidences)):
		label = aggregated_labels[i]


		if(ensemble_prediction[i] == label):
			correct_class[label] += 1
		total_class[label] += 1    
				
	# print partial model's cummulative accuracy
	print("Model accuracy: ")
	for i in range(6):
		print("%s: %s" % (i, model_correct[i] / float(model_total[i])))
		   



	# print ensemble cummulative accuracy
	print("FINAL - accuracy:", np.sum(correct_class) / np.sum(total_class))

	total_class[np.where(total_class == 0)] = 1

	np.save("classes.npy",  correct_class / total_class)


if __name__ == "__main__":
	"""Determine if the user has specified training or testing and run the appropriate function."""
	
	import argparse
	parser = argparse.ArgumentParser(description='Generate IADs from input files')

	#required command line args
	parser.add_argument('model_type', help='the type of model to use: I3D')
	parser.add_argument('model_filename', help='the checkpoint file to use with the model')

	parser.add_argument('dataset_dir', help='the directory whee the dataset is located')
	parser.add_argument('csv_filename', help='a csv file denoting the files in the dataset')

	parser.add_argument('num_classes', type=int, help='the number of classes in the dataset')
	parser.add_argument('operation', help='"train" or "test"')
	parser.add_argument('dataset_id', nargs='?', type=int, default=4, help='the dataset_id used to train the network. Is used in determing feature rank file')

	parser.add_argument('--pad_length', nargs='?', type=int, default=-1, help='the maximum length video to convert into an IAD')
	parser.add_argument('--epochs', nargs='?', type=int, default=-30, help='the maximum length video to convert into an IAD')
	parser.add_argument('--batch_size', nargs='?', type=int, default=-30, help='the maximum length video to convert into an IAD')
	parser.add_argument('--alpha', nargs='?', type=int, default=1e-4, help='the maximum length video to convert into an IAD')
	parser.add_argument('--feature_retain_count', nargs='?', type=int, default=-1, help='the number of features to remove')
	


	parser.add_argument('--gpu', default="0", help='gpu to run on')

	FLAGS = parser.parse_args()

	# optional - specify the CUDA device to use for GPU computation
	# comment this line out if you wish to use all CUDA-capable devices
	os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu











	# Setup file IO
	iad_data_path = os.path.join(FLAGS.dataset_dir, 'iad')

	try:
		csv_contents = read_csv(csv_file)
	except:
		print("Cannot open CSV file: "+ csv_file)

	for ex in csv_contents:
		file_location = os.path.join(ex['label_name'], ex['example_id'])
		for layer in range(5):
			iad_file = os.path.join(iad_data_path, file_location+"_"+str(layer)+".npz")
			assert os.path.exists(iad_file), "Cannot locate IAD file: "+ iad_file
			ex['iad_path_'+str(layer)] = iad_file

	train_data = [ex for ex in csv_contents if ex['dataset_id'] <= dataset_id and ex['dataset_id'] > 0]
	test_data  = [ex for ex in csv_contents if ex['dataset_id'] == 0]

	# Determine features to prune
	pruning_keep_indexes = None
	if(FLAGS.feature_retain_count and FLAGS.dataset_id):
		ranking_file = os.path.join(iad_data_path, "feature_ranks_"+str(dataset_id * 25)+".npz")
		assert os.path.exists(ranking_file), "Cannot locate Feature Ranking file: "+ ranking_file
		pruning_keep_indexes = get_top_n_feature_indexes(ranking_file, feature_retain_count)

	# Begin Training/Testing
	if(FLAGS.operation == "train"):
		train_model(FLAGS.model_filename, FLAGS.num_classes, train_data, test_data, pruning_keep_indexes)
	elif(FLAGS.operation == "train"):
		test_model (FLAGS.model_filename, FLAGS.num_classes, test_data, pruning_keep_indexes)
	else:
		print('Operation parameter must be either "train" or "test"')

