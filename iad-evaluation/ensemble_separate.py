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

def get_data_merged(ex, pruning_indexes, input_shape):
	flat_d = []

	for layer in range(5):
		window_size = input_shape[layer][1]
		d, l = get_data(ex, layer, pruning_indexes, window_size)
		flat_d.append(d.reshape(d.shape[0], -1, 1))


	return np.concatenate(flat_d, axis=1), l

def get_batch_data(dataset, model_num, pruning_indexes, input_shape, batch_size, sliding_window):


	batch_data, batch_label = [], []

	for b_idx in np.random.randint(0, len(dataset), size=batch_size):
		file_ex = dataset[b_idx]

		if model_num < 5:
			window_size = input_shape[model_num][1]
			d, l = get_data(file_ex, model_num, pruning_indexes, window_size)
		else:
			d, l = get_data_merged(file_ex, pruning_indexes, input_shape)

		w_idx = random.randint(0, d.shape[0]-1) if sliding_window else 0

		batch_data.append(d[w_idx])
		batch_label.append(l)

	return np.array(batch_data), np.array(batch_label)
		

def get_test_data(dataset, model_num, pruning_indexes, input_shape, idx):

	file_ex = dataset[idx]

	if model_num < 5:
		window_size = input_shape[model_num][1]
		return get_data(file_ex, model_num, pruning_indexes, window_size)
	else:
		return get_data_merged(file_ex, pruning_indexes, input_shape)

##############################################
# Model Structure
##############################################

def model_def(num_classes, input_shape, model_num, alpha):
	"""Create the tensor operations to be used in training and testing, stored in a dictionary."""

	def conv_model(input_ph):
		"""Return a convolutional model."""
		top = input_ph

		# input layers [batch_size, h, w, num_channels]
		top = tf.reshape(top, [-1, input_shape[model_num][0], input_shape[model_num][1], 1])

		'''
		# hidden layers (1x1 conv)
		num_filters = 16
		top = tf.layers.conv1d(
			inputs=top,
			filters=num_filters,
			padding="valid", 
			activation=tf.nn.leaky_relu)

		
		# hidden layers (1x4 conv)
		num_filters = 8
		filter_width = 4
		top = tf.layers.conv2d(
			inputs=top,
			filters=num_filters,
			kernel_size=[1, filter_width],
			padding="valid", 
			activation=tf.nn.leaky_relu)
		'''
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
	input_layer = ph["x_"+str(model_num)]
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
				data, label = get_batch_data(train_data, model_num, pruning_indexes, input_shape, batch_size, sliding_window)
				feed_dict = { ph["x_"+str(model_num)]: data, ph["y"]: label,  ph["train"]: True }

				sess.run(ops["train"], feed_dict=feed_dict)

				# print out every 2K iterations
				if i % 2000 == 0:
					print("step: ", str(i) + '/' + str(num_iter))
					
					# evaluate test network
					data, label = get_batch_data(test_data, model_num, pruning_indexes, input_shape, batch_size, sliding_window)
				
					feed_dict = { ph["x_"+str(model_num)]: data, ph["y"]: label,  ph["train"]: False }

					correct_prediction = sess.run([ops['model_preds']], feed_dict=feed_dict)
					correct, total = np.sum(correct_prediction[0] == label), len(correct_prediction[0] == label)

					print("model_num: {0}, test_accuracy: {1}, correct: {2}, total: {3}".format(model_num, correct / float(total), correct, total))

			# save the model
			save_name = model_dirs[model_num]+'/model'
			saver.save(sess, save_name)
			print("Final model saved in %s" % save_name)
		tf.reset_default_graph()


def test_model(iad_model_path, model_dirs, num_classes, test_data, pruning_indexes, num_features, window_size, sliding_window, dataset_type):

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

				aggregated_confidences[i].append([])


				data, label = get_test_data(test_data, model_num, pruning_indexes, input_shape, i)

				num_win = len(data) if sliding_window else 1

				for w_idx in range(num_win): 
					feed_dict = { ph["x_"+str(model_num)]: np.expand_dims(data[w_idx], axis = 0), ph["y"]: label,  ph["train"]: False }

					confidences, predictions = sess.run([ 
							ops['model_sftmx'], ops['model_preds']], 
							feed_dict=feed_dict)

					# append confidences for evaluating consensus model
					aggregated_confidences[i][model_num].append(confidences)
					if(model_num == 0 and w_idx ==0):
						aggregated_labels.append(label)

					# update model accuracy
					if(predictions == label):
						model_accuracy[model_num, 0] += 1
					model_accuracy[model_num, 1] += 1

		tf.reset_default_graph()

	for i in range(len(aggregated_confidences)):
		aggregated_confidences[i] = np.mean(aggregated_confidences[i], axis = 1)

	# generate wighted sum for ensemble of models 
	aggregated_confidences = np.transpose(np.array(aggregated_confidences), [0, 3, 2, 1])
	ensemble_prediction = model_consensus(aggregated_confidences)
	aggregated_labels = np.array(aggregated_labels).reshape(-1)

	for i, label in enumerate(aggregated_labels):
		if(ensemble_prediction[i] == label):
			class_accuracy[label, 0] += 1
		class_accuracy[label, 1] += 1    
				
	# print partial model's cummulative accuracy
	ofile = open(os.path.join(iad_model_path, "model_accuracy_"+dataset_type+".txt"), 'w')
	print("Model accuracy: ")
	for model_num in range(6):
		print("{:d}\t{:4.6f}".format(model_num, model_accuracy[model_num, 0] / float(model_accuracy[model_num, 1])) )
		ofile.write("{:d}\t{:4.6f}\n".format(model_num, model_accuracy[model_num, 0] / float(model_accuracy[model_num, 1])) )

	# print ensemble cummulative accuracy
	print("FINAL\t{:4.6f}".format( np.sum(ensemble_prediction == aggregated_labels) / float(len(aggregated_labels)) ) )
	ofile.write("FINAL\t{:4.6f}\n".format( np.sum(ensemble_prediction == aggregated_labels) / float(len(aggregated_labels)) ) )
	ofile.close()

	# save per-class accuracy
	np.save(os.path.join(iad_model_path, "class_accuracy_"+dataset_type+".npy"),  class_accuracy[:, 0] / class_accuracy[:, 1] )

	return aggregated_confidences, aggregated_labels

def prepare_filenames(dataset_dir, dataset_type, dataset_id, file_list):
	iad_data_path_frames = os.path.join(dataset_dir, 'iad_frames_'+str(dataset_id))
	iad_data_path_flow   = os.path.join(dataset_dir, 'iad_flow_'+str(dataset_id))

	for ex in file_list:
		file_location = os.path.join(ex['label_name'], ex['example_id'])
		for layer in range(5):

			if(dataset_type == 'frames' or dataset_type == 'both'):
				iad_frames = os.path.join(iad_data_path_frames, file_location+"_"+str(layer)+".npz")
				assert os.path.exists(iad_frames), "Cannot locate IAD file: "+ iad_frames
				ex['iad_path_'+str(layer)] = iad_frames

			if(dataset_type == 'flow' or dataset_type == 'both'):
				iad_flow = os.path.join(iad_data_path_flow, file_location+"_"+str(layer)+".npz")
				assert os.path.exists(iad_flow), "Cannot locate IAD file: "+ iad_flow
				ex['iad_path_'+str(layer)] = iad_flow

def main(model_type, dataset_dir, csv_filename, num_classes, operation, dataset_id, dataset_type,
		window_size, epochs, batch_size, alpha, 
		feature_retain_count, gpu, sliding_window):

	# optional - specify the CUDA device to use for GPU computation
	# comment this line out if you wish to use all CUDA-capable devices
	os.environ["CUDA_VISIBLE_DEVICES"] = gpu

	# Setup file IO
	model_id_path_frames = os.path.join('iad_model_'+str(window_size), 'model_frames'+str(dataset_id))
	iad_model_path_frames = os.path.join(dataset_dir, model_id_path_frames)

	model_id_path_flow = os.path.join('iad_model_'+str(window_size), 'model_flow'+str(dataset_id))
	iad_model_path_flow = os.path.join(dataset_dir, model_id_path_flow)

	model_dirs_frames, model_dirs_flow = [], []
	for model_num in range(6):

		# setup directory for frames
		separate_model_dir = os.path.join(iad_model_path_frames, 'model_'+str(model_num))
		model_dirs_frames.append(separate_model_dir)

		if(not os.path.exists(separate_model_dir)):
			os.makedirs(separate_model_dir)

		# setup directory for flow
		separate_model_dir = os.path.join(iad_model_path_flow, 'model_'+str(model_num))
		model_dirs_flow.append(separate_model_dir)

		if(not os.path.exists(separate_model_dir)):
			os.makedirs(separate_model_dir)

	# setup train and test file lists
	try:
		csv_contents = read_csv(csv_filename)
	except:
		print("Cannot open CSV file: "+ csv_filename)

	train_data = [ex for ex in csv_contents if ex['dataset_id'] >= dataset_id and ex['dataset_id'] != 0]
	test_data  = [ex for ex in csv_contents if ex['dataset_id'] == 0]
	
	print("Number Training Examples:", len(train_data))
	print("Number Testing Examples:",  len(test_data))

	#train_data = train_data[:5]
	#test_data = test_data[:5]

	# Determine features to prune
	pruning_keep_indexes_frame, pruning_keep_indexes_flow = None, None
	if(feature_retain_count and dataset_id):
		if(dataset_type == 'frames' or dataset_type == 'both'):
			iad_data_path_frames = os.path.join(dataset_dir, 'iad_frames_'+str(dataset_id))
			ranking_file = os.path.join(iad_data_path_frames, "feature_ranks_"+str(dataset_id)+".npz")
			assert os.path.exists(ranking_file), "Cannot locate Feature Ranking file: "+ ranking_file
			pruning_keep_indexes_frame = get_top_n_feature_indexes(ranking_file, feature_retain_count)

		if(dataset_type == 'flow' or dataset_type == 'both'):
			iad_data_path_flow = os.path.join(dataset_dir, 'iad_flow_'+str(dataset_id))
			ranking_file = os.path.join(iad_data_path_flow, "feature_ranks_"+str(dataset_id)+".npz")
			assert os.path.exists(ranking_file), "Cannot locate Feature Ranking file: "+ ranking_file
			pruning_keep_indexes_flow = get_top_n_feature_indexes(ranking_file, feature_retain_count)

	# Begin Training/Testing
	if(operation == "train"):
		prepare_filenames(dataset_dir, dataset_type, dataset_id, train_data)
		prepare_filenames(dataset_dir, dataset_type, dataset_id, test_data)

		if(dataset_type == 'frames'):
			train_model(model_dirs_frames, num_classes, train_data, test_data, pruning_keep_indexes_frame, feature_retain_count, window_size, batch_size, alpha, epochs, sliding_window)
		elif(dataset_type == 'flow'):
			train_model(model_dirs_flow,   num_classes, train_data, test_data, pruning_keep_indexes_flow, feature_retain_count, window_size, batch_size, alpha, epochs, sliding_window)
	
	elif(operation == "test"):
		if(dataset_type == 'frames'):
			prepare_filenames(dataset_dir, 'frames', dataset_id, test_data)
			print("\n\n\nframes_out:", test_data[0]['iad_path_0'])
			test_model (iad_model_path_frames, model_dirs_frames, num_classes, test_data, pruning_keep_indexes_frame, feature_retain_count, window_size, sliding_window, dataset_type)
			
		elif(dataset_type == 'flow'):
			prepare_filenames(dataset_dir, 'flow',   dataset_id, test_data)
			test_model (iad_model_path_flow,   model_dirs_flow,   num_classes, test_data, pruning_keep_indexes_flow, feature_retain_count, window_size, sliding_window, dataset_type)
		
		elif(dataset_type == 'both'):
			prepare_filenames(dataset_dir, 'frames', dataset_id, test_data)
			print("\n\n\nframes_out:", test_data[0]['iad_path_0'])
			frame_results, frame_labels = test_model (iad_model_path_frames, model_dirs_frames, num_classes, test_data, pruning_keep_indexes_frame, feature_retain_count, window_size, sliding_window, "frames")
			
			prepare_filenames(dataset_dir, 'flow',   dataset_id, test_data)
			print("\n\n\nflow_out:",test_data[0]['iad_path_0'])
			flow_results,  flow_labels  = test_model (iad_model_path_flow,   model_dirs_flow,   num_classes, test_data, pruning_keep_indexes_flow, feature_retain_count, window_size, sliding_window, "flow")
	
			#Get Individual accuracy
			results = np.stack((frame_results, flow_results))
			results = np.mean(results, axis = 0)
			results = np.squeeze(results)
			pred = np.argmax(results, axis=1)

			ofile = open(os.path.join(iad_model_path_frames, "model_accuracy_both.txt"), 'w')
			for model_num in range(6):
				pred = np.argmax(results[:, :, model_num], axis=1)
				print("{:d}\t{:4.6f}".format(model_num, np.sum(pred == frame_labels) / float(len(frame_labels)) ))
				ofile.write("{:d}\t{:4.6f}\n".format(model_num, np.sum(pred == frame_labels) / float(len(frame_labels)) ))

			#Get Ensemble accuracy
			confidence_discount_layer = [0.5, 0.7, 0.9, 0.9, 0.9, 1.0]

			frame_results = frame_results * confidence_discount_layer
			frame_results = np.sum(frame_results, axis=(2,3))

			flow_results = flow_results * confidence_discount_layer
			flow_results = np.sum(flow_results, axis=(2,3))

			results = np.stack((frame_results, flow_results))
			results = np.mean(results, axis = 0)
			pred = np.argmax(results, axis=1)

			print("FINAL\t{:4.6f}".format( np.sum(pred == frame_labels) / float(len(frame_labels)) ))
			ofile.write("FINAL\t{:4.6f}\n".format( np.sum(pred == frame_labels) / float(len(frame_labels)) ) )
			ofile.close()
			

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
	parser.add_argument('operation', help='"train" or "test"', choices=['train', 'test'])
	parser.add_argument('dataset_id', type=int, help='the dataset_id used to train the network. Is used in determing feature rank file')
	parser.add_argument('dataset_type', help='"frames", "flow", or "both" (only usable for test)', choices=['frames', 'flow', 'both'])
	parser.add_argument('window_size', type=int, help='the maximum length video to convert into an IAD')

	parser.add_argument('--sliding_window', type=bool, default=False, help='.list file containing the test files')
	parser.add_argument('--epochs', type=int, default=30, help='the maximum length video to convert into an IAD')
	parser.add_argument('--batch_size', type=int, default=15, help='the maximum length video to convert into an IAD')
	parser.add_argument('--alpha', type=int, default=1e-4, help='the maximum length video to convert into an IAD')
	parser.add_argument('--feature_retain_count', type=int, default=10000, help='the number of features to remove')
	
	parser.add_argument('--gpu', default="0", help='gpu to run on')

	FLAGS = parser.parse_args()

	main(FLAGS.model_type, 
		FLAGS.dataset_dir, 
		FLAGS.csv_filename, 
		FLAGS.num_classes, 
		FLAGS.operation, 
		FLAGS.dataset_id, 
		FLAGS.dataset_type,
		FLAGS.window_size, 
		FLAGS.epochs,
		FLAGS.batch_size,
		FLAGS.alpha,
		FLAGS.feature_retain_count,
		FLAGS.gpu,
		FLAGS.sliding_window)
