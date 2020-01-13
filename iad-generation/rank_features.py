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

batch_size = 1

def taylor_expansion(csv_contents, model_filename, pad_length, dataset_size, iad_data_path, isRGB):
	
	# define placeholder
	input_placeholder = model.get_input_placeholder(isRGB, batch_size, num_frames=pad_length)
	
	# define model
	activation_map, rankings, saver = model.load_model(input_placeholder, isRGB)
	
	#collapse the spatial dimensions of the activation map
	for layer in range(len(activation_map)):
		activation_map[layer] = tf.reduce_max(activation_map[layer], axis = (2,3))
		activation_map[layer] = tf.squeeze(activation_map[layer])
		activation_map[layer] = tf.transpose(activation_map[layer])

	with tf.Session() as sess:

		# Restore model
		sess.run(tf.global_variables_initializer())
		tf_utils.restore_model(sess, saver, model_filename)

		# prevent further modification to the graph
		sess.graph.finalize()

		summed_ranks = None

		# process files
		for i in range(len(csv_contents)):
			file = csv_contents[i]['raw_path']

			print("converting video to IAD: {:6d}/{:6d}".format(i, len(csv_contents)))
			raw_data, length_ratio = model.read_file(file, input_placeholder, isRGB)

			# generate activation map from model
			iad_data, rank_data = sess.run([activation_map, rankings], feed_dict={input_placeholder: raw_data})

			for row in rank_data:
				print("row:", row.shape)

			#summed_ranks = rank_data if summed_ranks == None else np.add(summed_ranks, rank_data)

	# save ranking files
	depth, index, rank = [],[],[] 

	for layer in range(len(summed_ranks)):
		depth.append(np.full(len(summed_ranks[layer]), layer))
		index.append(np.arange(len(summed_ranks[layer])))
		rank.append(summed_ranks[layer])

	filename = os.path.join(iad_data_path, "feature_ranks2_"+str(dataset_size)+".npz")
	np.savez(filename, 
		depth=np.concatenate(depth), 
		index=np.concatenate(index), 
		rank=np.concatenate(rank))


def group_vars(group):
	ends = ['Branch_0/Conv3d_0a_1x1', 'Branch_1/Conv3d_0b_3x3', 'Branch_2/Conv3d_0b_3x3', 'Branch_3/Conv3d_0b_1x1']
	return [group+'/'+e+'/conv_3d/w' for e in ends]



def weight_magnitudes(model_type, model_filename, dataset_dir, csv_filename, dataset_id, pad_length, gpu, isRGB):

	input_placeholder = model.get_input_placeholder(isRGB, batch_size, num_frames=pad_length)
	
	# define model
	activation_map, rankings, saver = model.load_model(input_placeholder, isRGB)
	variables = model.get_variables(isRGB)

	weights = [ ['Conv3d_1a_7x7/conv_3d/w'],
				['Conv3d_2c_3x3/conv_3d/w'],
				group_vars('Mixed_3c'), #['Branch_0/Conv3d_0a_1x1', 'Branch_1/Conv3d_0b_3x3', 'Branch_2/Conv3d_0b_3x3', 'Branch_3/Conv3d_0b_1x1'],
				group_vars('Mixed_4f'), #['Branch_0/Conv3d_0a_1x1', 'Branch_1/Conv3d_0b_3x3', 'Branch_2/Conv3d_0b_3x3', 'Branch_3/Conv3d_0b_1x1'],
				group_vars('Mixed_5c')] #['Branch_0/Conv3d_0a_1x1', 'Branch_1/Conv3d_0b_3x3', 'Branch_2/Conv3d_0b_3x3', 'Branch_3/Conv3d_0b_1x1'],

	all_w = []

	for i, end_point in enumerate(weights):
		all_w.append([])

		for element in end_point:
			for v in variables:
				#print(str(v), element)
				if(element in str(v)):
					all_w[i].append(variables[v])

		all_w[i] = [tf.reduce_sum(tf.math.abs( v ), axis=[0,1,2,3]) for v in all_w[i]]
		all_w[i] = tf.concat(all_w[i], axis=0)

		print(all_w[i])

	with tf.Session() as sess:

		# Restore model
		sess.run(tf.global_variables_initializer())
		tf_utils.restore_model(sess, saver, model_filename)

		# prevent further modification to the graph
		sess.graph.finalize()

		depth, index, rank = [],[],[] 

		for layer, w in enumerate(all_w):

			w0 = sess.run(w)
			#print(w0, w0[0].shape)

			depth.append(np.full(len(w0[0]), layer))
			index.append(np.arange(len(w0[0])))
			rank.append(w0[0])



	# save ranking files
	iad_data_path = os.path.join(dataset_dir, 'iad_'+file_loc+'_'+str(dataset_id))
	filename = os.path.join(iad_data_path, "feature_ranks_l1_"+str(dataset_size)+".npz")
	np.savez(filename, 
		depth=np.concatenate(depth), 
		index=np.concatenate(index), 
		rank=np.concatenate(rank))
	


	



def main(model_type, model_filename, dataset_dir, csv_filename, dataset_id, pad_length, gpu, isRGB):

	os.environ["CUDA_VISIBLE_DEVICES"] = gpu

	file_loc = 'frames' if isRGB else 'flow'

	raw_data_path = os.path.join(dataset_dir, file_loc)
	iad_data_path = os.path.join(dataset_dir, 'iad_'+file_loc+'_'+str(dataset_id))

	csv_contents = read_csv(csv_filename)
	csv_contents = [ex for ex in csv_contents if ex['dataset_id'] >= dataset_id or ex['dataset_id'] == 0]
	
	csv_contents = csv_contents[:3]

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

	# generate arrays to store the min and max values of each feature
	
	taylor_expansion(csv_contents, model_filename, pad_length, dataset_id, iad_data_path, isRGB)

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='Generate IADs from input files')
	#required command line args
	parser.add_argument('model_type', help='the type of model to use: I3D')
	parser.add_argument('model_filename', help='the checkpoint file to use with the model')

	parser.add_argument('dataset_dir', help='the directory whee the dataset is located')
	parser.add_argument('csv_filename', help='a csv file denoting the files in the dataset')

	parser.add_argument('dataset_id', type=int, help='a csv file denoting the files in the dataset')

	parser.add_argument('--pad_length', nargs='?', type=int, default=-1, help='the maximum length video to convert into an IAD')
	parser.add_argument('--gpu', default="0", help='gpu to run on')
	parser.add_argument('--rgb', type=bool, default=False, help='run on RGB as opposed to flow data')

	FLAGS = parser.parse_args()


	weight_magnitudes(FLAGS.model_type, 
		FLAGS.model_filename, 
		FLAGS.dataset_dir, 
		FLAGS.csv_filename, 
		FLAGS.dataset_id,
		FLAGS.pad_length, 
		FLAGS.gpu,
		FLAGS.rgb)
	'''
	main(FLAGS.model_type, 
		FLAGS.model_filename, 
		FLAGS.dataset_dir, 
		FLAGS.csv_filename, 
		FLAGS.dataset_id,
		FLAGS.pad_length, 
		FLAGS.gpu,
		FLAGS.rgb)
	'''
	
