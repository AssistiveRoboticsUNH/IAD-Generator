# Madison Clark-Turner
# iad_generator.py
# 10/10/2019

from csv_utils import read_csv
import tf_utils

#import c3d as model
#import c3d_large as model

#import i3d_wrapper as model



import os, sys

import tensorflow as tf
import numpy as np

batch_size = 1

def convert_to_iad(data, meta_data, min_max_vals, length_ratio, update_min_maxes, iad_data_path):
	#converts file to iad and extracts the max and min values for the given IAD

	#update max and min values
	#print(meta_data['dataset_id'])
	#print(type(data[0]))
	if(update_min_maxes and meta_data['dataset_id'] != 0):
		for layer in range(len(data)):
			local_max_values = np.max(data[layer], axis=1)
			local_min_values = np.min(data[layer], axis=1)

			print(local_max_values.shape)
			print(local_min_values.shape)

			for i in range(len(local_max_values)):
				if(local_max_values[i] > min_max_vals["max"][layer][i]):
					min_max_vals["max"][layer][i] = local_max_values[i]

				if(local_min_values[i] < min_max_vals["min"][layer][i]):
					min_max_vals["min"][layer][i] = local_min_values[i]

	#save to disk
	for layer in range(len(data)):
		label_path = os.path.join(iad_data_path, meta_data['label_name'])
		if(not os.path.exists(label_path)):
			os.makedirs(label_path)

		meta_data['iad_path_'+str(layer)] = os.path.join(label_path, meta_data['example_id'])+"_"+str(layer)+".npz"

		data[layer] = data[layer][:, :int(data[layer].shape[1]*length_ratio)]

		np.savez(meta_data['iad_path_'+str(layer)], data=data[layer], label=meta_data['label'], length=data[layer].shape[1])

def convert_dataset_to_iad(csv_contents, min_max_vals, model, pad_length, dataset_size, update_min_maxes, iad_data_path):
	
	# set to None initiially and then accumulates over time
	summed_ranks = None

	# process files
	for i in range(len(csv_contents)):
		print("converting video to IAD: {:6d}/{:6d}".format(i, len(csv_contents)))

		iad_data, length_ratio = model.process(csv_contents[i], max_length=20)

		# generate activation map and rankings from model
		#iad_data, rank_data, length_ratio = model.process(csv_contents[i])

		# write the am_layers to file and get the minimum and maximum values for each feature row
		convert_to_iad(iad_data, csv_contents[i], min_max_vals, length_ratio, update_min_maxes, iad_data_path)

		# add new ranks to cummulative taylor sum

	# save ranking files
	depth, index, rank = [],[],[] 

	#save min_max_vals
	if(update_min_maxes):
		np.savez(os.path.join(iad_data_path, "min_maxes.npz"), min=np.array(min_max_vals["min"]), max=np.array(min_max_vals["max"]))

def main(model_type, model_filename, dataset_dir, csv_filename, num_classes, dataset_id, pad_length, min_max_file, gpu, dtype):

	os.environ["CUDA_VISIBLE_DEVICES"] = gpu

	file_loc = 'frames' if dtype else 'flow'

	raw_data_path = os.path.join(dataset_dir, file_loc)
	iad_data_path = os.path.join(dataset_dir, 'iad_'+file_loc+'_'+str(dataset_id))

	csv_contents = read_csv(csv_filename)
	csv_contents = [ex for ex in csv_contents if ex['dataset_id'] == dataset_id][:5]

	# get the maximum frame length among the dataset and add the 
	# full path name to the dict
	max_frame_length = 0
	filenames, labels = [],[]
	for ex in csv_contents:
		file_location = os.path.join(ex['label_name'], ex['example_id'])
		ex['raw_path'] = os.path.join(raw_data_path, file_location)

		if(ex['length'] > max_frame_length):
			max_frame_length = ex['length']

	#csv_contents = csv_contents[:5]

	print("numIADs:", len(csv_contents))
	print("max_frame_length:", max_frame_length)

	if (pad_length < 0):
		pad_length = max_frame_length
	print("padding iads to a length of {0} frames".format(max_frame_length))

	if(not os.path.exists(iad_data_path)):
		os.makedirs(iad_data_path)

	#define the model
	if(model_type == 'i3d'):
		from i3d_wrapper import I3DBackBone as bb
	if(model_type == 'tsm'):
		from tsm_wrapper import TSMBackBone as bb
	model = bb(model_filename, num_classes)

	# generate arrays to store the min and max values of each feature
	update_min_maxes = (min_max_file == None)
	if(update_min_maxes):
		min_max_vals = {"max": [],"min": []}
		for layer in range(len(model.CNN_FEATURE_COUNT)):
			min_max_vals["max"].append([float("-inf")] * model.CNN_FEATURE_COUNT[layer])
			min_max_vals["min"].append([float("inf")] * model.CNN_FEATURE_COUNT[layer])
	else:
		f = np.load(min_max_file, allow_pickle=True)
		min_max_vals = {"max": f["max"],"min": f["min"]}

	#generate IADs
	convert_dataset_to_iad(csv_contents, min_max_vals, model, pad_length, dataset_id, update_min_maxes, iad_data_path)
	#normalize_dataset(csv_contents, min_max_vals, model)

	#summarize operations
	print("--------------")
	print("Summary")
	print("--------------")
	print("Dataset ID: {0}".format(dataset_id))
	print("Number of videos into IADs: {0}".format(len(csv_contents)))
	print("IADs are padded/pruned to a length of: {0}".format(pad_length))
	print("Longest video sequence in file list: {0}".format(max_frame_length))
	print("Files place in: {0}".format(iad_data_path))
	print("Min/Max File was Saved: {0}".format(update_min_maxes))


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='Generate IADs from input files')
	#required command line args
	parser.add_argument('model_type', help='the type of model to use', choices=['i3d', 'tsm'])
	parser.add_argument('model_filename', help='the checkpoint file to use with the model')

	parser.add_argument('dataset_dir', help='the directory whee the dataset is located')
	parser.add_argument('csv_filename', help='a csv file denoting the files in the dataset')
	parser.add_argument('num_classes', type=int, help='number of classes')

	parser.add_argument('dataset_id', type=int, help='a csv file denoting the files in the dataset')
	parser.add_argument('num_features', type=int, default=128, help='the number of features to retain')

	parser.add_argument('--pad_length', nargs='?', type=int, default=-1, help='the maximum length video to convert into an IAD')
	parser.add_argument('--min_max_file', nargs='?', default=None, help='a .npz file containing min and max values to normalize by')
	parser.add_argument('--gpu', default="0", help='gpu to run on')
	parser.add_argument('--dtype', default="frames", help='run on RGB as opposed to flow data', choices=['frames', 'flow'])

	FLAGS = parser.parse_args()

	main(FLAGS.model_type, 
		FLAGS.model_filename, 
		FLAGS.dataset_dir, 
		FLAGS.csv_filename, 
		FLAGS.num_classes,
		FLAGS.dataset_id,
		FLAGS.pad_length, 
		FLAGS.min_max_file, 
		FLAGS.gpu,
		FLAGS.dtype)

	
