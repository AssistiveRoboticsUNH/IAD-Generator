# Madison Clark-Turner
# iad_generator.py
# 2/13/2019

from csv_utils import read_csv
from feature_rank_utils import get_top_n_feature_indexes
import tf_utils

import os, sys, time

import tensorflow as tf
import numpy as np

from multiprocessing import Pool

batch_size = 1

def convert_to_iad(data, csv_input, length_ratio, iad_data_path):
	#save to disk
	for layer in range(len(data)):
		label_path = os.path.join(iad_data_path, csv_input['label_name'])
		if(not os.path.exists(label_path)):
			os.makedirs(label_path)

		csv_input['iad_path_'+str(layer)] = os.path.join(label_path, csv_input['example_id'])+"_"+str(layer)+".npz"

		data[layer] = data[layer][:, :int(data[layer].shape[1]*length_ratio)]
		#print(data[layer].shape)

		np.savez_compressed(csv_input['iad_path_'+str(layer)], data=data[layer], label=csv_input['label'], length=data[layer].shape[1])

def convert_dataset_to_iad(csv_contents, model, iad_data_path):
	
	# set to None initiially and then accumulates over time
	summed_ranks = None

	# process files
	for i, csv_ex in enumerate(csv_contents):
		t_s = time.time()

		# generate activation map
		iad_data, length_ratio = model.process(csv_ex)

		# write the am_layers to file and get the minimum and maximum values for each feature row
		convert_to_iad(iad_data, csv_ex, length_ratio, iad_data_path)

		print("converted video to IAD: {:6d}/{:6d}, time: {:8.2}".format(i, len(csv_contents), time.time()-t_s))

def convert_csv_chunk(inputs):
	csv_contents, model_type, model_filename, iad_data_path, num_classes, max_length, feature_idx = inputs
	#print([ex['example_id'] for ex in csv_contents])

	#define the model
	if(model_type == 'i3d'):
		from gi3d_wrapper import I3DBackBone as bb
	if(model_type == 'trn'):
		from trn_wrapper import TRNBackBone as bb
	if(model_type == 'tsm'):
		from tsm_wrapper import TSMBackBone as bb
	model = bb(model_filename, num_classes, max_length=max_length, feature_idx=feature_idx)
	
	#generate IADs
	convert_dataset_to_iad(csv_contents, model, iad_data_path)

def main(
	model_type, model_filename, 
	dataset_dir, csv_filename, num_classes, dataset_id, 
	feature_rank_file, max_length, 
	num_features=128, dtype="frames", gpu=0, num_procs=1, single=""
	):

	os.environ["CUDA_VISIBLE_DEVICES"] = gpu

	file_loc = 'frames' if dtype else 'flow'

	raw_data_path = os.path.join(dataset_dir, file_loc)
	iad_data_path = os.path.join(dataset_dir, 'iad_{0}_{1}_{2}'.format(model_type,file_loc,dataset_id))

	csv_contents = read_csv(csv_filename)
	
	if(single == ""):
		csv_contents = [ex for ex in csv_contents if ex['dataset_id'] >= dataset_id or ex['dataset_id'] == 0]
	else:
		csv_contents = [ex for ex in csv_contents if ex['example_id'] == single]
	#print([ex["example_id"] for ex in csv_contents])
	#csv_contents = csv_contents[:50]
	
	if (gpu == 0):
		print("set 1")
		csv_contents = csv_contents[:len(csv_contents)/2]
	else:
		print("set 2")
		csv_contents = csv_contents[len(csv_contents)/2:]
	

	# get the maximum frame length among the dataset and add the 
	# full path name to the dict
	max_frame_length = 0
	for ex in csv_contents:

		file_location = os.path.join(ex['label_name'], ex['example_id'])
		ex['raw_path'] = os.path.join(raw_data_path, file_location)

		if(ex['length'] > max_frame_length):
			max_frame_length = ex['length']

	if(not os.path.exists(iad_data_path)):
		os.makedirs(iad_data_path)

	feature_idx = get_top_n_feature_indexes(feature_rank_file, num_features)

	p = Pool(num_procs)

	inputs = []
	chunk_size = len(csv_contents)/float(num_procs)
	last = 0.0

	while last < len(csv_contents):
		inputs.append(
			(
			csv_contents[int(last):int(last+chunk_size)], 
			model_type, 
			model_filename, 
			iad_data_path,
			num_classes, 
			max_length, 
			feature_idx
			)
		)
		last += chunk_size

	#convert files to IAD in parallel
	#convert_csv_chunk(inputs[0])
	p.map(convert_csv_chunk, inputs)

	#summarize operations
	print("--------------")
	print("Summary")
	print("--------------")
	print("Dataset ID: {0}".format(dataset_id))
	print("Number of videos into IADs: {0}".format(len(csv_contents)))
	print("IADs are padded/pruned to a length of: {0}".format(max_length))
	print("Files place in: {0}".format(iad_data_path))
	#print("Min/Max File was Saved: {0}".format(update_min_maxes))

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

	# IAD gen command line args
	parser.add_argument('feature_rank_file', help='a .npz file containing min and max values to normalize by')
	parser.add_argument('max_length', type=int, help='the maximum length video to convert into an IAD')

	# optional command line args
	parser.add_argument('--num_features', type=int, default=128, help='the number of features to retain')
	parser.add_argument('--dtype', default="frames", help='run on RGB as opposed to flow data', choices=['frames', 'flow'])
	parser.add_argument('--gpu', default="0", help='gpu to run on')
	parser.add_argument('--num_procs', default=1, type=int, help='number of process to split IAD generation over')
	parser.add_argument('--single', default="", help='process a singular file at given path')


	FLAGS = parser.parse_args()

	main(
		FLAGS.model_type, 
		FLAGS.model_filename,

		FLAGS.dataset_dir, 
		FLAGS.csv_filename, 
		FLAGS.num_classes,
		FLAGS.dataset_id,

		FLAGS.feature_rank_file,
		FLAGS.max_length,

		FLAGS.num_features, 
		FLAGS.dtype,
		FLAGS.gpu,
		FLAGS.num_procs,
		FLAGS.single,
		)
	
	
