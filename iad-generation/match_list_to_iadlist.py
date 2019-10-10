def match_lists(iadlist_filename, list_filename, output_filename):
	data_names = [row.split()[0] for row in list(open(list_filename, 'r'))]

	iadlist_contents = list(open(iadlist_filename, 'r'))

	ofile = open(output_filename, 'w')

	for l in iadlist_contents:
		l = l.split()
		data_name = l[0][:-6]

		if(data_name in data_names):
			ofile.write(l):
	ofile.close()

if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser(description="Ensemble model processor")
	parser.add_argument('iadlist_filename', help='A .iadlist file')
	parser.add_argument('list_filename', type=int, help='A .list file')
	parser.add_argument('output_filename', help='The .iadlist file to write to')
	FLAGS = parser.parse_args()

	match_lists(FLAGS.iadlist_filename, FLAGS.list_filename, FLAGS.output_filename)

