# Madison Clark-Turner
# rank_features.py
# 2/13/2020

from csv_utils import read_csv

import os, sys, random

#import tensorflow as tf
import numpy as np


def train(csv_contents, model_type, model_filename, num_classes, dataset_id, iad_data_path):

	#csv_contents, model_type, model_filename, num_classes, dataset_id, iad_data_path = inp

	#define the model
	if(model_type == 'i3d'):
		from gi3d_wrapper import I3DBackBone as bb
	if(model_type == 'trn'):
		from trn_wrapper import TRNBackBone as bb
	if(model_type == 'tsm'):
		from tsm_wrapper import TSMBackBone as bb
	model = bb(model_filename, num_classes)


	# wrapper should take care of the bottlenecking steps
	train(csv_contents, model, dataset_id, iad_data_path)

def main(
	model_type, model_filename, 
	dataset_dir, csv_filename, num_classes, dataset_id, 
	dataset_size, dtype, gpu, num_procs
	):

	os.environ["CUDA_VISIBLE_DEVICES"] = gpu

	file_loc = 'frames' if dtype else 'flow'

	raw_data_path = os.path.join(dataset_dir, file_loc)
	iad_data_path = os.path.join(dataset_dir, 'iad_'+model_type+'_'+file_loc+'_'+str(dataset_id))

	# parse CSV file
	csv_contents = read_csv(csv_filename)
	csv_contents = [ex for ex in csv_contents if ex['dataset_id'] == dataset_id]
	random.shuffle(csv_contents)
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



	
	train(
		csv_contents, 
		model_type, 
		model_filename, 
		num_classes, 
		dataset_id, 
		iad_data_path)


	# save model


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
	parser.add_argument('--num_procs', default=1, type=int, help='number of process to split IAD generation over')
	

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
		FLAGS.num_procs,
		)

	
