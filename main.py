#!/usr/bin/env python3

import tensorflow as tf

import training_data

def main():
	# numpy.set_printoptions(threshold=sys.maxsize)

	print("TensorFlow version:", tf.__version__)
	training_data.loadTrainingData()
	
if __name__ == '__main__':
	main()