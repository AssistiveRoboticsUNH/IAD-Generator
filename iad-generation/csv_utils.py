import csv, sys

def convert_list_to_csv(trainlist_filename, testlist_filename, csv_filename):

	with open(csv_filename, 'w') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=['label_name', 'example_id', 'label', 'dataset_id'])

		writer.writeheader()

		for i, line in enumerate( list(open(trainlist_filename, 'r')) ):

			# open each .list row
			line = line.split(' ')
			filename, label = line[0], int(line[1])

			# extract file label info
			example_id = filename.split('/')[-1]
			label_name = filename.split('/')[-2]
			
			# assign dataset_id
			dataset_id = (i%4) + 1

			# write to CSV
			writer.writerow({'label_name': label_name, 'example_id': example_id, 'label':label, 'dataset_id':dataset_id})

		for i, line in enumerate( list(open(testlist_filename, 'r')) ):

			# open each .list row
			line = line.split(' ')
			filename, label = line[0], int(line[1])

			# extract file label info
			example_id = filename.split('/')[-1]
			label_name = filename.split('/')[-2]
			
			# assign dataset_id
			dataset_id = 0

			# write to CSV
			writer.writerow({'label_name': label_name, 'example_id': example_id, 'label':label, 'dataset_id':dataset_id})

def read_csv_file(csv_file):
	return []

if __name__ == '__main__':
	convert_list_to_csv(sys.argv[1], sys.argv[2], sys.argv[3])