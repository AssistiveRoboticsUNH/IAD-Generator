# Madison Clark-Turner
# rank_features.py
# 2/13/2020

from csv_utils import read_csv

import os, sys, random

import tensorflow as tf
import numpy as np

def rank_dataset(csv_contents, model, dataset_id, iad_data_path):
	
	# set to None initiially and then accumulates over time
	summed_ranks = []

	# process files
	for i in range(len(csv_contents)):
		print("Ranking features for file: {:6d}/{:6d}".format(i, len(csv_contents)))

		# rank files
		rank_data = model.rank(csv_contents[i])

		# add new ranks to cummulative sum
		for j, rd in enumerate(rank_data):
			if(i == 0):
				summed_ranks.append(rd)
			else:
				summed_ranks[j] = np.add(summed_ranks[j], rd)

	# save ranking files
	depth, index, rank = [],[],[] 
	for layer in range(len(summed_ranks)):
		depth.append(np.full(len(summed_ranks[layer]), layer))
		index.append(np.arange(len(summed_ranks[layer])))
		rank.append(summed_ranks[layer])

	filename = os.path.join(iad_data_path, "feature_ranks_"+str(dataset_id)+".npz")
	np.savez(filename, 
		depth=np.concatenate(depth), 
		index=np.concatenate(index), 
		rank=np.concatenate(rank))

def main(
	model_type, model_filename, 
	dataset_dir, csv_filename, num_classes, dataset_id, 
	dataset_size, dtype, gpu
	):

	os.environ["CUDA_VISIBLE_DEVICES"] = gpu

	file_loc = 'frames' if dtype else 'flow'

	raw_data_path = os.path.join(dataset_dir, file_loc)
	iad_data_path = os.path.join(dataset_dir, 'iad_'+model_type+'_'+file_loc+'_'+str(dataset_id))

	# parse CSV file
	csv_contents = read_csv(csv_filename)
	csv_contents = [ex for ex in csv_contents if ex['dataset_id'] == dataset_id]
	#random.shuffle(csv_contents)
	csv_contents = csv_contents[:dataset_size]

	# get the maximum frame length among the dataset and add the 
	# full path name to the dict
	max_frame_length = 0
	filenames, labels = [],[]
	for ex in csv_contents:
		file_location = os.path.join(ex['label_name'], ex['example_id'])
		ex['raw_path'] = os.path.join(raw_data_path, file_location)

		if(ex['length'] > max_frame_length):
			max_frame_length = ex['length']

	if(not os.path.exists(iad_data_path)):
		os.makedirs(iad_data_path)

	#define the model
	if(model_type == 'i3d'):
		from gi3d_wrapper import I3DBackBone as bb
	if(model_type == 'trn'):
		from trn_wrapper import TRNBackBone as bb
	if(model_type == 'tsm'):
		from tsm_wrapper import TSMBackBone as bb
	model = bb(model_filename, num_classes)

	#generate IADs
	rank_dataset(csv_contents, model, dataset_id, iad_data_path)

	#summarize operations
	print("--------------")
	print("Summary")
	print("--------------")
	print("Dataset ID: {0}".format(dataset_id))
	print("Longest video sequence in file list: {0}".format(max_frame_length))
	print("Files placed in: {0}".format(iad_data_path))

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='Generate IADs from input files')
	
	# model command line args
	parser.add_argument('model_type', help='the type of model to use', choices=['i3d', 'trn', 'tsm'])
	parser.add_argument('model_filename', help='the checkpoint file to use with the model')

	# dataset command line args
	parser.add_argument('dataset_dir', help='the directory whee the dataset is located')
	parser.add_argument('csv_filename', help='a csv file denoting the files in the dataset')
	parser.add_argument('num_classes', type=int, help='number of classes')
	parser.add_argument('dataset_id', type=int, help='a csv file denoting the files in the dataset')

	# optional command line args
	parser.add_argument('--dataset_size', default=2000, type=int, help='number of examples to base choice on')
	parser.add_argument('--dtype', default="frames", help='run on RGB as opposed to flow data', choices=['frames', 'flow'])
	parser.add_argument('--gpu', default="0", help='gpu to run on')

	FLAGS = parser.parse_args()

	main(
		FLAGS.model_type, 
		FLAGS.model_filename, 

		FLAGS.dataset_dir, 
		FLAGS.csv_filename, 
		FLAGS.num_classes,
		FLAGS.dataset_id,

		FLAGS.dataset_size, 
		FLAGS.dtype,
		FLAGS.gpu,
		)

	
