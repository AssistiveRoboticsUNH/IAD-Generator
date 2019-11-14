import os, csv

def split_trainlist_to_percent_chunks(csv_filename, trainlist_filename, testlist_filename):

	with open(csv_filename, 'w') as csvfile:
		writer = csv.DictWriter(csvfile, 
			fieldnames=['label_name', 'example_id', 'label', 'dataset_id', 'length'])

		writer.writeheader()

		for i, line in enumerate( list(open(trainlist_filename, 'r')) ):

			# open each .list row
			line = line.split(' ')
			filename, label = line[0], int(line[1])

			# extract file label info
			example_id = filename.split('/')[-1]
			label_name = filename.split('/')[-2]
			
			# assign dataset_id
			dataset_id = 4-(i%4)

			# determine length
			length = len(os.listdir(filename))

			# write to CSV
			writer.writerow({'label_name': label_name, 
							'example_id': example_id, 
							'label':label, 
							'dataset_id':dataset_id,
							'length':length})

		for i, line in enumerate( list(open(testlist_filename, 'r')) ):

			# open each .list row
			line = line.split(' ')
			filename, label = line[0], int(line[1])

			# extract file label info
			example_id = filename.split('/')[-1]
			label_name = filename.split('/')[-2]
			
			# assign dataset_id
			dataset_id = 0

			# determine length
			length = len(os.listdir(filename))

			# write to CSV
			writer.writerow({'label_name': label_name, 
							'example_id': example_id, 
							'label':label, 
							'dataset_id':dataset_id,
							'length':length})

def convert_listfiles_to_csv(csv_filename, file_list):
	with open(csv_filename, 'w') as csvfile:
		writer = csv.DictWriter(csvfile, 
			fieldnames=['label_name', 'example_id', 'label', 'dataset_id', 'length'])

		writer.writeheader()

		# organize data
		data_rows = {}
		for dataset_id, filename in enumerate(file_list):
			for line in list(open(filename, 'r')):

				line = line.split(' ')
				filename, label = line[0], int(line[1])

				# extract file label info
				example_id = filename.split('/')[-1]
				label_name = filename.split('/')[-2]

				# determine length
				length = len(os.listdir(filename))

				if(line not in data_rows):
					data_rows[example_id] = {
						'label_name': label_name, 
						'example_id': example_id, 
						'label':label, 
						'length':length}
				data_rows[example_id]['dataset_id'] = dataset_id

		# write data to csv
		for k in data_rows.values():
			writer.writerow(v)

def read_csv(csv_file):
	csv_contents = []

	with open(csv_file) as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			data = {'label_name': row['label_name'], 
					'example_id': row['example_id'], 
					'label':int(row['label']), 
					'dataset_id':int(row['dataset_id']), 
					'length':int(row['length']) }
			csv_contents.append(data)
	
	return csv_contents	

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='Convert a .list file into a .csv file')

	#required command line args
	parser.add_argument('csv_filename', help='the name of the .csv file to generate')
	parser.add_argument('list_files', type=argparse.FileType('r'), nargs='+', help=
		'''a list of .list files to add to the CSV. It is recommended to start 
		with the test dataset, followed by the largest training dataset and 
		then decrementing to the smallest training dataset''')

	#parser.add_argument('trainlist_filename', help='the .list file containing the training data')
	#parser.add_argument('testlist_filename', help='the .list file containing the test data')
	#parser.add_argument('csv_filename', help='the name of the .csv file to generate')
	FLAGS = parser.parse_args()

	print("FLAGS.list_files:")
	print(FLAGS.list_files)

	convert_listfiles_to_csv(FLAGS.csv_filename, FLAGS.list_files)