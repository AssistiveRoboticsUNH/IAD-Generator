# Madison Clark-Turner
# iad_generator.py
# 8/29/2019

import c3d as model
from file_reader import obtain_files, read_file
from feature_rank_utils import order_feature_ranks

import os

import tensorflow as tf
import numpy as np

import argparse
parser = argparse.ArgumentParser(description='Generate IADs from input files')
#required command line args
parser.add_argument('model_file', help='the tensorflow ckpt file used to generate the IADs')
parser.add_argument('dataset_file', help='the *.list file than contains the ')

#optional command line args
parser.add_argument('--prefix', nargs='?', type=str, default="complete", help='the prefix to place infront of finished files <prefix>_<layer>.npz')
parser.add_argument('--dst_directory', nargs='?', type=str, default='generated_iads/', help='where the IADs should be stored')
#test dataset command line args
parser.add_argument('--min_max_file', nargs='?', type=str, default=None, help='max and minimum values')
parser.add_argument('--pad_length', nargs='?', type=int, default=-1, help='length to pad/prune the videos to, default is padd to the longest file in the dataset')
#feature pruning command line args
parser.add_argument('--feature_rank_file', nargs='?', type=str, default=None, help='a file containing the rankings of the features')
parser.add_argument('--feature_remove_count', nargs='?', type=int, default=0, help='the number of features to remove')

FLAGS = parser.parse_args()

batch_size = 1

def output_filename(file, layer):
	return os.path.join(FLAGS.dst_directory, file.split(os.path.sep)[-1]+"_"+str(layer)+".npz")

def convert_to_iad(data, label, file, min_max_vals, update_min_maxes, length_ratio):
	#converts file to iad and extracts the max and min values for the given IAD

	#update max and min values
	if(min_max_vals):
		for layer in range(len(data)):
			local_max_values = np.max(data[layer], axis=1)
			local_min_values = np.min(data[layer], axis=1)

			for i in range(len(local_max_values)):
				if(local_max_values[i] > min_max_vals["max"][layer][i]):
					min_max_vals["max"][layer][i] = local_max_values[i]

				if(local_min_values[i] < min_max_vals["min"][layer][i]):
					min_max_vals["min"][layer][i] = local_min_values[i]

	#save to disk
	for i in range(len(data)):
		filename = output_filename(file, i)
		np.savez(filename, data=data[i], label=label, length=int(data[i].shape[1]*length_ratio))

def convert_dataset_to_iad(list_of_files, min_max_vals, update_min_maxes):
	
	# define placeholder
	input_placeholder = model.get_input_placeholder(batch_size, num_frames=FLAGS.pad_length)
	
	# define model
	weights, biases = model.get_variables()
	variable_name_dict = list( set(weights.values() + biases.values()))
	saver = tf.train.Saver(variable_name_dict)

	activation_map = model.generate_activation_map(input_placeholder, weights, biases)
	
	#collapse the spatial dimensions of the activation map
	for layer in range(len(activation_map)):
		activation_map[layer] = tf.reduce_max(activation_map[layer], axis = (2,3))
		activation_map[layer] = tf.squeeze(activation_map[layer])
		activation_map[layer] = tf.transpose(activation_map[layer])

	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=.9)#.25
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

		# initialize model variables to those in the described checkpoint file
		saver.restore(sess, FLAGS.model_file)

		# prevent further modification to the graph
		sess.graph.finalize()

		# process files
		for i in range(len(list_of_files)):
			file, label = list_of_files[i]

			print("converting video to IAD: {:6d}/{:6d}".format(i, len(list_of_files)))

			# read data into placeholders
			raw_data, length_ratio = read_file(file, input_placeholder)

			# generate activation map from model
			iad_data = sess.run(activation_map, feed_dict={input_placeholder: raw_data})

			# write the am_layers to file and get the minimum and maximum values for each feature row
			convert_to_iad(iad_data, label, file, min_max_vals, update_min_maxes, length_ratio)

	#save min_max_vals
	if(update_min_maxes):
		np.savez(os.path.join(FLAGS.dst_directory, "min_maxes.npz"), min=np.array(min_max_vals["min"]), max=np.array(min_max_vals["max"]))
	
def normalize_dataset(list_of_files, min_max_vals):
	for i in range(len(list_of_files)):
		file, _ = list_of_files[i]

		print("normalizing IAD: {:6d}/{:6d}".format(i, len(list_of_files)))

		for layer in range(len(model.CNN_FEATURE_COUNT)):

			filename = output_filename(file, layer)
			f = np.load(filename)
			data, label, length = f["data"], f["label"], f["length"]

			for row in range(data.shape[0]):
				if(min_max_vals["max"][layer][row] - min_max_vals["min"][layer][row] == 0):
					data[row] = np.zeros_like(data[row])
				else:
					data[row] = (data[row] - min_max_vals["min"][layer][row]) / (min_max_vals["max"][layer][row] - min_max_vals["min"][layer][row])
			np.savez(filename, data=data, label=label, length=length)

def get_features_to_prune(feature_rank_file, num_features_to_remove):
	#initalize array
	prune_locs = []
	for i in range(len(model.CNN_FEATURE_COUNT)):
		prune_locs.append([])
	
	if(feature_rank_file != None):
	
		#open file
		depth, index, rank = order_feature_ranks(feature_rank_file)

		#get the worst N features
		for c in range(num_features_to_remove):
			prune_locs[depth[c]].append(index[c])

	#arrays to np arrays
	for i in range(len(model.CNN_FEATURE_COUNT)):
		prune_locs[i] = np.array(prune_locs[i])

	return prune_locs

def combine_npy_files(list_of_files, prune_locs=None):
	# combine all of the IADs from a specific depth together
	for layer in range(len(model.CNN_FEATURE_COUNT)):
		data_all, label_all, length_all = [],[],[]
		for i in range(len(list_of_files)):
			file, _ = list_of_files[i]

			filename = output_filename(file, layer)
			f = np.load(filename)

			data, label, length = f["data"], f["label"], f["length"]

			data_all.append(data)
			label_all.append(label)
			length_all.append(length)

		num_features = np.array(data_all).shape[1]
		keep_locs = np.arange(num_features)
		if(prune_locs != None):  
			keep_locs = np.delete(keep_locs, prune_locs[layer])

		print(np.array(data_all)[:, keep_locs, :].shape)

		np.savez(os.path.join(FLAGS.dst_directory, FLAGS.prefix+"_"+str(layer)+".npz"), 
				data=np.array(data_all)[:, keep_locs, :], 
				label=np.array(label_all), 
				length=np.array(length_all))

def clean_up_npy_files(list_of_files):
	for i in range(len(list_of_files)):
		file, _ = list_of_files[i]
		for layer in range(len(model.CNN_FEATURE_COUNT)):
			os.remove(output_filename(file, layer))


if __name__ == '__main__':
	
	list_of_files_and_labels, max_frame_length = obtain_files(FLAGS.dataset_file)

	if(not os.path.exists(FLAGS.dst_directory)):
		os.makedirs(FLAGS.dst_directory)

	if (FLAGS.pad_length < 0):
		FLAGS.pad_length = max_frame_length
		print("padding iads to a length of {0} frames".format(max_frame_length))

	# generate arrays to store the min and max values of each feature
	update_min_maxes = (FLAGS.min_max_file == None)
	if(update_min_maxes):
		min_max_vals = {"max": [],"min": []}
		for layer in range(len(model.CNN_FEATURE_COUNT)):
			min_max_vals["max"].append([float("-inf")] * model.CNN_FEATURE_COUNT[layer])
			min_max_vals["min"].append([float("inf")] * model.CNN_FEATURE_COUNT[layer])
	else:
		f = np.load(FLAGS.min_max_file, allow_pickle=True)
		min_max_vals = {"max": f["max"],"min": f["min"]}

	convert_dataset_to_iad(list_of_files_and_labels, min_max_vals, update_min_maxes)
	normalize_dataset(list_of_files_and_labels, min_max_vals)

	prune_locs = get_features_to_prune(FLAGS.feature_rank_file, FLAGS.feature_remove_count)

	combine_npy_files(list_of_files_and_labels, prune_locs)
	clean_up_npy_files(list_of_files_and_labels)

	#summarize operations
	print("--------------")
	print("Summary")
	print("--------------")
	print("Number of videos into IADs: {0}".format(len(list_of_files_and_labels)))
	print("IADs are padded/pruned to a length of: {0}".format(FLAGS.pad_length))
	print("Longest video sequence in file list: {0}".format(max_frame_length))
	print("Files place in: {0}".format(FLAGS.dst_directory))
	print("Min/Max File was Saved: {0}".format(update_min_maxes))

	print("Removed Features:")
	for i in range(len(prune_locs)):
		print("\tremoved {0} features from layer {1}".format(len(prune_locs[i]), i))