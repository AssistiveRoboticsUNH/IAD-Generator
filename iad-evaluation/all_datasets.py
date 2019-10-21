from multiprocessing import Process
from ensemble_separate import main






def f(model_type, dataset_dir, csv_filename, num_classes, operation, dataset_id,  
		window_size, epochs, batch_size, alpha, 
		feature_retain_count, gpu, sliding_window):
	main(model_type, dataset_dir, csv_filename, num_classes, operation, dataset_id, 
		window_size, epochs, batch_size, alpha, 
		feature_retain_count, gpu, sliding_window)

if __name__ == '__main__':


	import argparse
	parser = argparse.ArgumentParser(description='Generate IADs from input files')
	#required command line args
	parser.add_argument('model_type', help='the type of model to use: I3D')

	parser.add_argument('dataset_dir', help='the directory where the dataset is located')
	parser.add_argument('csv_filename', help='a csv file denoting the files in the dataset')

	parser.add_argument('num_classes', type=int, help='the number of classes in the dataset')
	parser.add_argument('operation', help='"train" or "test"')
	#parser.add_argument('dataset_id', nargs='?', type=int, help='the dataset_id used to train the network. Is used in determing feature rank file')
	parser.add_argument('window_size', nargs='?', type=int, help='the maximum length video to convert into an IAD')

	parser.add_argument('--sliding_window', type=bool, default=False, help='.list file containing the test files')
	parser.add_argument('--epochs', nargs='?', type=int, default=30, help='the maximum length video to convert into an IAD')
	parser.add_argument('--batch_size', nargs='?', type=int, default=15, help='the maximum length video to convert into an IAD')
	parser.add_argument('--alpha', nargs='?', type=int, default=1e-4, help='the maximum length video to convert into an IAD')
	parser.add_argument('--feature_retain_count', nargs='?', type=int, default=10000, help='the number of features to remove')
	
	parser.add_argument('--gpu', default="0", help='gpu to run on')

	FLAGS = parser.parse_args()

	procs = []
	for dataset_id in range(4, 0, -1):
		p = Process(target=f, args=(FLAGS.model_type, 
		FLAGS.dataset_dir, 
		FLAGS.csv_filename, 
		FLAGS.num_classes, 
		FLAGS.operation, 
		dataset_id, 
		FLAGS.window_size, 
		FLAGS.epochs,
		FLAGS.batch_size,
		FLAGS.alpha,
		FLAGS.feature_retain_count,
		FLAGS.gpu,
		 ))
		p.start()
		p.join()
