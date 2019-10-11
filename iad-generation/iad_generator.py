# Madison Clark-Turner
# iad_generator.py
# 10/10/2019

from csv_utils import read_csv
import tf_utils

#import c3d as model
#import c3d_large as model
#import i3d_wrapper as model
import rank_i3d as model

import os, sys

import tensorflow as tf
import numpy as np

import argparse
parser = argparse.ArgumentParser(description='Generate IADs from input files')

#required command line args
parser.add_argument('model_type', help='the type of model to use: I3D')
parser.add_argument('model_filename', help='the checkpoint file to sue with the model')

parser.add_argument('dataset_dir', help='the directory whee the dataset is located')
parser.add_argument('csv_filename', help='a csv file denoting the files in the dataset')

parser.add_argument('--pad_length', nargs='?', type=int, default=-1, help='the maximum length video to convert into an IAD')
parser.add_argument('--min_max_file', nargs='?', default=None, help='a .npz file containing min and max values to normalize by')
parser.add_argument('--gpu', default="0", help='gpu to run on')

FLAGS = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

batch_size = 1


RAW_DATA_PATH = os.path.join(FLAGS.dataset_dir, 'imgFiles')
IAD_DATA_PATH = os.path.join(FLAGS.dataset_dir, 'iad')

UPDATE_MIN_MAXES = (FLAGS.min_max_file == None)


def convert_to_iad(data, meta_data, min_max_vals, length_ratio):
	#converts file to iad and extracts the max and min values for the given IAD

	#update max and min values
	if(UPDATE_MIN_MAXES):
		for layer in range(len(data)):
			local_max_values = np.max(data[layer], axis=1)
			local_min_values = np.min(data[layer], axis=1)

			for i in range(len(local_max_values)):
				if(local_max_values[i] > min_max_vals["max"][layer][i]):
					min_max_vals["max"][layer][i] = local_max_values[i]

				if(local_min_values[i] < min_max_vals["min"][layer][i]):
					min_max_vals["min"][layer][i] = local_min_values[i]

	#save to disk
	for layer in range(len(data)):
		label_path = os.path.join(IAD_DATA_PATH, meta_data['label_name'])
		if(not os.path.exists(label_path)):
			os.makedirs(label_path)

		meta_data['iad_path_'+str(layer)] = os.path.join(label_path, meta_data['example_id'])+"_"+str(layer)+".npz"

		data[layer] = data[layer][:, :int(data[layer].shape[1]*length_ratio)]

		np.savez(meta_data['iad_path_'+str(layer)], data=data[layer], label=meta_data['label'], length=data[layer].shape[1])

def convert_dataset_to_iad(csv_contents, min_max_vals):
	
	# define placeholder
	input_placeholder = model.get_input_placeholder(batch_size, num_frames=FLAGS.pad_length)
	
	# define model
	activation_map, rankings, saver = model.load_model(input_placeholder)
	print("rank3", rankings[0].get_shape())
	
	#collapse the spatial dimensions of the activation map
	for layer in range(len(activation_map)):
		activation_map[layer] = tf.reduce_max(activation_map[layer], axis = (2,3))
		activation_map[layer] = tf.squeeze(activation_map[layer])
		activation_map[layer] = tf.transpose(activation_map[layer])

	with tf.Session() as sess:

		# Restore model
		sess.run(tf.global_variables_initializer())
		tf_utils.restore_model(sess, saver, FLAGS.model_filename)

		# prevent further modification to the graph
		sess.graph.finalize()

		summed_ranks = [None]*4

		# process files
		for i in range(len(csv_contents)):
			file, label = csv_contents[i]['raw_path'], csv_contents[i]['label']

			print("converting video to IAD: {:6d}/{:6d}".format(i, len(csv_contents)))

			# read data into placeholders
			raw_data, length_ratio = model.read_file(file, input_placeholder)

			# generate activation map from model
			iad_data, rank_data = sess.run([activation_map, rankings], feed_dict={input_placeholder: raw_data})

			# write the am_layers to file and get the minimum and maximum values for each feature row
			convert_to_iad(iad_data, csv_contents[i], min_max_vals, length_ratio)

			# add new ranks to cummulative sum
			for j in range(csv_contents[i]['dataset_id']):
				summed_ranks[j] = r if summed_ranks[j] == None else np.add(summed_ranks[j], r)

	# save ranking files
	for j in range(4):
		depth, index, rank = [],[],[] 

		for i in range(len(summed_ranks[j])):
			depth.append(np.full(len(summed_ranks[j][i]), i))
			index.append(np.arange(len(summed_ranks[j][i])))
			rank.append(summed_ranks[j][i])

		filename = os.path.join(IAD_DATA_PATH, "feature_ranks_"+str(i)+".npz")
		np.savez(filename, 
			depth=np.concatenate(depth), 
			index=np.concatenate(index), 
			rank=np.concatenate(rank))


	#save min_max_vals
	if(UPDATE_MIN_MAXES):
		np.savez(os.path.join(IAD_DATA_PATH, "min_maxes.npz"), min=np.array(min_max_vals["min"]), max=np.array(min_max_vals["max"]))
	
def normalize_dataset(csv_contents, min_max_vals):
	for i in range(len(csv_contents)):
		print("normalizing IAD: {:6d}/{:6d}".format(i, len(csv_contents)))

		for layer in range(len(model.CNN_FEATURE_COUNT)):

			filename = csv_contents[i]['iad_path_'+str(layer)]

			# open .npz file
			f = np.load(filename)
			data, label, length = f["data"], f["label"], f["length"]

			# normalize IAD
			for row in range(data.shape[0]):
				if(min_max_vals["max"][layer][row] - min_max_vals["min"][layer][row] == 0):
					data[row] = np.zeros_like(data[row])
				else:
					data[row] = (data[row] - min_max_vals["min"][layer][row]) / (min_max_vals["max"][layer][row] - min_max_vals["min"][layer][row])

			# re-save file
			np.savez(filename, data=data, label=label, length=length)

if __name__ == '__main__':
	
	csv_contents = read_csv(FLAGS.csv_filename)

	# get the maximum frame length among the dataset and add the 
	# full path name to the dict
	max_frame_length = 0
	filenames, labels = [],[]
	for ex in csv_contents:
		file_location = os.path.join(ex['label_name'], ex['example_id'])
		ex['raw_path'] = os.path.join(RAW_DATA_PATH, file_location)

		if(ex['length'] > max_frame_length):
			max_frame_length = ex['length']

	csv_contents = csv_contents[:3]

	print("numIADs:", len(csv_contents))
	print("max_frame_length:", max_frame_length)

	if (FLAGS.pad_length < 0):
		FLAGS.pad_length = max_frame_length
	print("padding iads to a length of {0} frames".format(max_frame_length))

	if(not os.path.exists(IAD_DATA_PATH)):
		os.makedirs(IAD_DATA_PATH)

	# generate arrays to store the min and max values of each feature
	UPDATE_MIN_MAXES = (FLAGS.min_max_file == None)
	if(UPDATE_MIN_MAXES):
		min_max_vals = {"max": [],"min": []}
		for layer in range(len(model.CNN_FEATURE_COUNT)):
			min_max_vals["max"].append([float("-inf")] * model.CNN_FEATURE_COUNT[layer])
			min_max_vals["min"].append([float("inf")] * model.CNN_FEATURE_COUNT[layer])
	else:
		f = np.load(FLAGS.min_max_file, allow_pickle=True)
		min_max_vals = {"max": f["max"],"min": f["min"]}

	convert_dataset_to_iad(csv_contents, min_max_vals)
	normalize_dataset(csv_contents, min_max_vals)

	#summarize operations
	print("--------------")
	print("Summary")
	print("--------------")
	print("Number of videos into IADs: {0}".format(len(csv_contents)))
	print("IADs are padded/pruned to a length of: {0}".format(FLAGS.pad_length))
	print("Longest video sequence in file list: {0}".format(max_frame_length))
	print("Files place in: {0}".format(IAD_DATA_PATH))
	print("Min/Max File was Saved: {0}".format(UPDATE_MIN_MAXES))
