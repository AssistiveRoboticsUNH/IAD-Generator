# Madison Clark-Turner
# iad_generator.py
# 8/29/2019

import c3d as model
from file_reader import obtain_files, read_file

import os

import tensorflow as tf
import numpy as np

import argparse
parser = argparse.ArgumentParser(description='Generate IADs from input files')
#required command line args
parser.add_argument('model_file', help='the tensorflow ckpt file used to generate the IADs')
parser.add_argument('dataset_file', help='the *.list file than contains the ')
#optional command line args
parser.add_argument('--prefix', nargs='?', default="complete", help='the prefix to place infront of finished files <prefix>_<layer>.npz')
parser.add_argument('--min_max_file', nargs='?', default=None, help='max and minimum values')
parser.add_argument('--features_file', nargs='?', default=None, help='which features to keep')
parser.add_argument('--dst_directory', nargs='?', default='generated_iads/', help='where the IADs should be stored')
parser.add_argument('--pad_length', nargs='?', default=-1, help='length to pad/prune the videos to, default is padd to the longest file in the dataset')
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
	input_placeholder = model.get_input_placeholder(batch_size)
	
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

def combine_npy_files(list_of_files):
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

		np.savez(os.path.join(FLAGS.dst_directory, FLAGS.prefix+"_"+str(layer)+".npz"), 
				data=np.array(data_all), 
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
	combine_npy_files(list_of_files_and_labels)
	clean_up_npy_files(list_of_files_and_labels)

	#summarize operations
	print("Summary")
	print("--------------")
	print("Number of videos into IADs: {0}".format(len(list_of_files_and_labels)))
	print("IADs are padded/pruned to a length of: {0}".format(FLAGS.pad_length))
	print("Longest video sequence in file list: {0}".format(max_frame_length))
	print("Files place in: {0}".format(FLAGS.dst_directory))
	print("Min/Max File was Saved: {0}".format(update_min_maxes))


































'''


from iad_writer_json import write_json_to_disk, read_json_file_entire
import c3d#c3d_mod_model as c3d
from file_reader import FileReader, FileProperties

import os
from multiprocessing import RawArray, Queue
from threading import Thread, Semaphore

import tensorflow as tf
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import json
import cv2

import argparse

import time

parser = argparse.ArgumentParser(description='Generate IADs from input files')
parser.add_argument('src_directory', help='where the input files can be located')
parser.add_argument('bag_type', help='indicate the data type of the input')
parser.add_argument('model_name', help='the name of the model to build from')
parser.add_argument('--dst_directory', nargs='?', default='generated_iads/', help='where the IADs should be stored')
parser.add_argument('--num_threads', nargs='?', default=1, help='number of concurrent python thredas to run')

FLAGS = parser.parse_args()

batch_size = 1

class Record:
	def __init__(self, filename, file_properties):
		self.filename = filename
		self.file_properties = file_properties

def read_files_in_dir(directory, file_ext=".bag"):
	#Get all of the files that end with file_id and are located in the provided directory
	#	- directory - directory to investigate
	#	- file_id - a string that denotes the file_extention
	contents = [os.path.join(directory, f) for f in os.listdir(directory)]

	all_files = []
	for f in contents:
		if os.path.isfile(f) and f.find(file_ext) >=0:
			all_files += [f]
		elif os.path.isdir(f):
			all_files += read_files_in_dir(f)
	
	return all_files

def write_min_max_to_file(min_vals, max_vals):

	print("begin writing min-max values")
	filename = os.path.join(FLAGS.dst_directory, "min-max.json") # replace with actual name, write into file
	
	data = {"depth": len(min_vals)}
	#c3d_activation_map
	for c3d_depth in range(len(min_vals)):

		data["num_features_"+str(c3d_depth)] = len(min_vals[c3d_depth])

		min_dump, max_dump = [],[]
		for x in min_vals[c3d_depth]:
			min_dump.append(x)
		for x in max_vals[c3d_depth]:
			max_dump.append(x)

		data["min_"+str(c3d_depth)] = min_dump
		data["max_"+str(c3d_depth)] = max_dump

	json.dump(data, open(filename, 'w'))

	print("finished writing min-max values")

######## Parse the Input through the CNN ######## 

def update_min_maxes(c3d_activation_map, max_vals, min_vals):
	#get the minimum and maximum values for each row. Update the standard

	for c3d_depth in range(len(c3d_activation_map)):
		local_max_values = np.max(c3d_activation_map[c3d_depth], axis=1)
		local_min_values = np.min(c3d_activation_map[c3d_depth], axis=1)

		for i in range(len(local_max_values)):

			if(local_max_values[i] > max_vals[c3d_depth][i]):
				max_vals[c3d_depth][i] = local_max_values[i]

			if(local_min_values[i] < min_vals[c3d_depth][i]):
				min_vals[c3d_depth][i] = local_min_values[i]

def store_and_get_min_max(iad_map, file_properties, max_vals, min_vals, records):
	#get the minimum and maximum values for each row in the activation_map and then write the file to 
	#disk temporarily.
	update_min_maxes(iad_map, max_vals, min_vals)

	for i in range(len(iad_map)):
		iad_map[i] = iad_map[i][:,:int(iad_map[i].shape[1]*file_properties.data_ratio)]

	print(file_properties.original_name, file_properties.original_name.split('/'))
	filename = os.path.join(FLAGS.dst_directory, file_properties.original_name.split('/')[-1]+".json")
	write_json_to_disk(filename, iad_map, file_properties)
	records.put(Record(filename, file_properties))

def convert_videos_to_IAD(model_name, file_reader, records):
	#opens an unthreshodled IAD and thresholds given the new values
	#	- records - providing a records variable indicates that the function is 
	#		meant to be run as global_norm not local_norm
	print("Parsing files through Model")
	
	# generate arrays to store the min and max values of each feature
	max_vals, min_vals = [],[]
	for c3d_depth in range(len(model.CNN_FEATURE_COUNT)):
		max_vals.append(RawArray('d', model.CNN_FEATURE_COUNT[c3d_depth])) 
		min_vals.append(RawArray('d', model.CNN_FEATURE_COUNT[c3d_depth]))
		for i in range(model.CNN_FEATURE_COUNT[c3d_depth]):
			max_vals[c3d_depth][i] = float("-inf")
			min_vals[c3d_depth][i] = float("inf")


	# define model and tensorflow parameters
	placeholders = model.get_input_placeholder(batch_size)
	weights, biases = model.get_variables()
	variable_name_dict = list( set(weights.values() + biases.values()))
	
	# wrap each activation map in a maximum function
	c3d_model = model.generate_activation_map(placeholders, weights, biases, depth=2)
	
	for layer in range(len(c3d_model)):
		
		# collapse the hieght and width dimensions of the activation map
		c3d_model[layer] = tf.reduce_max(c3d_model[layer], axis = (2,3))
		#c3d_model[layer] = tf.Print(c3d_model[layer], [c3d_model[layer]], message="max: ", summarize=10)

		# orient data to be features x time instances
		c3d_model[layer] = tf.squeeze(c3d_model[layer])
		c3d_model[layer] = tf.transpose(c3d_model[layer])
	

	#config = tf.ConfigProto()
	#config.gpu_options.allow_growth = True

	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=.9)#.25

	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

		# initialize model variables to those in the described checkpoint file
		saver = tf.train.Saver(variable_name_dict)
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		saver.restore(sess, model_name)

		# if file_reader takes TFRecords as input we need to setup those tensors
		file_reader.prime()

		# prevent further modification to the graph
		sess.graph.finalize()

		# setup file readers in case input is a TF Record
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord, sess=sess)

		# limit the number of threads running at once
		all_procs = []
		
		# process files
		while(not file_reader.isEmpty()):
			print("file_reader:", file_reader.filename_tracker, len(file_reader.filenames))

			if(file_reader.filename_tracker %1000 == 0 ):
				print("Converted "+str(file_reader.filename_tracker)+" files")

			# read data into placeholders
			print("get data")
			ph_values, file_properties = file_reader.generate_model_input(placeholders, sess, pad_length=model.INPUT_DATA_SIZE["t"])
			# generate activation map from model
			print("run_model")
			t_s = time.time()
			iad_layers = sess.run(c3d_model, feed_dict=ph_values)

			print("activation_map_layers[0].shape:", iad_layers[0].shape, " exec_time: ", time.time() - t_s)

			# write the am_layers to file and get the minimum and maximum values for each feature row
			store_and_get_min_max(iad_layers, file_properties, max_vals, min_vals, records)
			p=None
			all_procs.append(p)
				
		#for p in all_procs:
		#	p.join()
			
		coord.request_stop()
		coord.join(threads)

	return max_vals, min_vals

######## Normalize the Data ######## 

def normalize_data(record, max_vals, min_vals):
	filename = record.filename
	file_properties = record.file_properties

	normalized_data = []

	data_dict = read_json_file_entire(filename)

	for c3d_depth in range(data_dict['depth']):
		data_values = np.array(data_dict["data"][c3d_depth]).reshape(data_dict["num_rows"][c3d_depth], data_dict["num_columns"][c3d_depth])

		for row in range(data_values.shape[0]):

			if(max_vals[c3d_depth][row] - min_vals[c3d_depth][row] == 0):
				data_values[row] = np.zeros_like(data_values[row])
			else:
				data_values[row] = (data_values[row] - min_vals[c3d_depth][row]) / (max_vals[c3d_depth][row] - min_vals[c3d_depth][row])
		normalized_data.append(data_values)

	print(filename)

	img = normalized_data[3].copy()
	img = cv2.blur(img,(5,5))
	img *= 255
	img = img.astype(np.uint8)
	cv2.imwrite("pics/sanity_iad.png", img)

	write_json_to_disk(filename, normalized_data, file_properties)

def global_norm(records, max_vals, min_vals):
	#opens an unthreshodled IAD and thresholds given the new values
	print("Globally Normalizing IADs...")

	file_count = 0
	while(not records.empty()):

		if(file_count %1000 == 0 ):
			print("normalized "+str(file_count)+" files")

		normalize_data(records.get(), max_vals, min_vals)
		file_count+=1

def generate_normalized_iads(file_reader):
	
	#make destination directory if it doesn't exist
	if not os.path.exists(FLAGS.dst_directory):
		os.makedirs(FLAGS.dst_directory)

	records = Queue()
	# get the maximum and mimimum activation values for each iad row
	max_vals, min_vals = convert_videos_to_IAD(FLAGS.model_name, file_reader, records)
	write_min_max_to_file(min_vals, max_vals)

	# generate IADs using the earlier identified values
	global_norm(records, max_vals, min_vals)	

if __name__ == '__main__':
	#locate all of the files to be converted
	file_reader = None

	if(FLAGS.bag_type == "bag"):

		from rosbag_reader import RosBagFileReader

		filenames = read_files_in_dir(FLAGS.src_directory)
		file_reader = RosBagFileReader(filenames[:3], batch_size=1)

	elif(FLAGS.bag_type == "ucf"):

		from ucf_reader import UCF101FileReader

		filenames = FLAGS.src_directory
		file_reader = UCF101FileReader(filenames, batch_size=1)#, limit_read=5)

	assert file_reader != None

	generate_normalized_iads(file_reader)
'''