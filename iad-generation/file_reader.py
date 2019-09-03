import tensorflow as tf
import numpy as np

import os, cv2

import PIL.Image as Image


def obtain_files(directory_file):
	# read a *.list file and get the file names and labels
	ifile = open(directory_file, 'r')
	line = ifile.readline()
	
	filenames, labels, max_length = [],[],0
	while len(line) != 0:
		filename, start_frame, label = line.split()

		if(filename not in filenames):
			filenames.append(filename)
			labels.append(int(label))

		if(int(start_frame) + 16 > max_length):
			max_length = int(start_frame) + 16

		line = ifile.readline()

	return zip(filenames, labels), max_length

def read_file(file, input_placeholder):
	# read a file and concatenate all of the frames
	# pad or prune the video to the given length

	print("reading: "+ file)

	# input_shape
	_, num_frames, h, w, ch = input_placeholder.get_shape()

	# read available frames in file
	img_data = []
	for r, d, f in os.walk(file):
		f.sort()
		limit = min(num_frames, len(f))
		
		for i in range(limit):
			filename = os.path.join(r, f[i])
			img = Image.open(filename)

			# resize and crop to fit input size
			img = np.array(cv2.resize(np.array(img),(171, 128))).astype(np.float32)
			crop_x = int((img.shape[0] - h)/2)
			crop_y = int((img.shape[1] - w)/2)
			img = img[crop_x:crop_x+w, crop_y:crop_y+h,:] 

			#subtract the mean value expression from the data and convert to RGB
			img -= np.array([90, 98, 102]) # mean values
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

			img_data.append(np.array(img))

	img_data = np.array(img_data).astype(np.float32)
	length_ratio = len(img_data) / int(num_frames)

	# pad file to appropriate length
	buffer_len = int(num_frames) - len(img_data)
	img_data = np.pad(np.array(img_data), 
				((0,buffer_len), (0,0), (0,0),(0,0)), 
				'constant', 
				constant_values=(0,0))
	img_data = np.expand_dims(img_data, axis=0)



	return img_data, length_ratio 










'''
class FileProperties:
	def __init__(self, original_name, uid, label, oiginal_length, data_ratio):
		self.original_name = original_name
		self.uid = uid
		self.label = label
		self.original_length = oiginal_length
		self.data_ratio = data_ratio


class FileReader:
	
	def __init__(self, filenames, batch_size=1):
		self.filenames = filenames
		self.batch_size = batch_size
		self.filename_tracker = 0

	def generate_model_input(self, placeholders, sess, pad_length):
		#Obtain the contents from one of the files in self.filenames. 
		#Store relevant information into the provided placeholders. Sess is provided incase
		#data needs to be read from a TFRecord
		ph_values = {}
		info_values = FileProperties("file_name", 0, 0, 0, 0)

		return ph_values, info_values

	def prime(self):
		#Use to setup any tensorflow commands for reading TF Records, etc.
		pass

	def isEmpty(self):
		return self.filename_tracker == len(self.filenames)

	def getCurrentFile():
		return self.filenames[self.filename_tracker]
'''