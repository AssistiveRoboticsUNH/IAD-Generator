from multiprocessing import Process
from ensemble_separate import main






def f(model_type, dataset_dir, csv_filename, num_classes, operation, dataset_id, model_filename, 
		window_size, epochs, batch_size, alpha, 
		feature_retain_count, gpu):
	main(model_type, dataset_dir, csv_filename, num_classes, operation, dataset_id, model_filename, 
		window_size, epochs, batch_size, alpha, 
		feature_retain_count, gpu)

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
	parser.add_argument('--min_max_file', nargs='?', default=None, help='a .npz file containing min and max values to normalize by')
	parser.add_argument('--gpu', default="0", help='gpu to run on')

	FLAGS = parser.parse_args()

	procs = []
	for dataset_id in range(4, 0, -1):
		p = Process(target=f, args=(FLAGS.model_type, 
		FLAGS.model_filename, 
		FLAGS.dataset_dir, 
		FLAGS.csv_filename, 
		dataset_id,
		FLAGS.pad_length, 
		FLAGS.min_max_file, 
		FLAGS.gpu,
		 ))
		p.start()
		p.join()
