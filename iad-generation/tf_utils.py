import tensorflow as tf
import sys, os

def restore_model(sess, saver, model_filename):
	ckpt = tf.train.get_checkpoint_state(model_filename)

	# first check if the checkpoint file is a directory containing a 'checkpoint' file
	if ckpt and ckpt.model_checkpoint_path:
		print("loading checkpoint %s,waiting......" % ckpt.model_checkpoint_path)
		saver.restore(sess, ckpt.model_checkpoint_path)
		print("load complete!")

	# the file may be an actual .ckpt file, check that next
	elif os.path.exists(model_filename):
		print("loading checkpoint file: "+model_filename)
		saver.restore(sess, model_filename)	
		
	else:
		print("Failed to Load model: "+model_filename)
		sys.exit(1)