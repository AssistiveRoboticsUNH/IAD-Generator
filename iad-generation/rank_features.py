# Madison Clark-Turner
# iad_generator.py
# 10/10/2019

from csv_utils import read_csv
import tf_utils

import os, sys

import tensorflow as tf
import numpy as np

batch_size = 1

def rank_dataset(csv_contents, min_max_vals, model, pad_length, dataset_size, update_min_maxes, iad_data_path):
	
	# set to None initiially and then accumulates over time
	summed_ranks = []

	# process files
	for i in range(len(csv_contents)):
		print("converting video to IAD: {:6d}/{:6d}".format(i, len(csv_contents)))

		# rank files
		rank_data = model.rank(csv_contents[i])

		# add new ranks to cummulative taylor sum
		for j, rd in enumerate(rank_data):
			rd = rd.numpy()
			print(rd.shape, type(rd))


			if(i == 0):
				summed_ranks.append(rd)
			else:
				summed_ranks = np.add(summed_ranks[j], rd)

		#summed_ranks = rank_data if i == 0 else np.add(summed_ranks, rank_data)

	# save ranking files
	depth, index, rank = [],[],[] 

	for layer in range(len(summed_ranks)):
		depth.append(np.full(len(summed_ranks[layer]), layer))
		index.append(np.arange(len(summed_ranks[layer])))
		rank.append(summed_ranks[layer])

	filename = os.path.join(iad_data_path, "feature_ranks_"+str(dataset_size)+".npz")
	np.savez(filename, 
		depth=np.concatenate(depth), 
		index=np.concatenate(index), 
		rank=np.concatenate(rank))

def main(model_type, model_filename, dataset_dir, csv_filename, num_classes, dataset_id, pad_length, min_max_file, gpu, dtype):

	os.environ["CUDA_VISIBLE_DEVICES"] = gpu

	file_loc = 'frames' if dtype else 'flow'

	raw_data_path = os.path.join(dataset_dir, file_loc)
	iad_data_path = os.path.join(dataset_dir, 'iad_'+file_loc+'_'+str(dataset_id))

	csv_contents = read_csv(csv_filename)
	csv_contents = [ex for ex in csv_contents if ex['dataset_id'] == dataset_id][:5]
	
	#csv_contents = csv_contents[:3]

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
	rank_dataset(csv_contents, min_max_vals, model, pad_length, dataset_id, update_min_maxes, iad_data_path)

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

	
