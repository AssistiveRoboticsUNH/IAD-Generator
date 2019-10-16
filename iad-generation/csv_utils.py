import os, csv

def convert_list_to_csv(trainlist_filename, testlist_filename, csv_filename):

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
	parser.add_argument('trainlist_filename', help='the .list file containing the training data')
	parser.add_argument('testlist_filename', help='the .list file containing the test data')
	parser.add_argument('csv_filename', help='the name of the .csv file to generate')
	FLAGS = parser.parse_args()

	convert_list_to_csv(FLAGS.trainlist_filename, FLAGS.testlist_filename, FLAGS.csv_filename)