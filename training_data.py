import os
import tensorflow as tf
from PIL import Image
import numpy
import sys


def loadTFData():
	mnist = tf.keras.datasets.mnist
	print(type(mnist.load_data()[0][1]))


def loadTrainingData():

	inputData = list()
	outputClassification = list()

	if os.path.exists("inputData.npy") and os.path.exists("outputClassification.npy"):
		inputData = numpy.load("inputData.npy")
		outputClassification = numpy.load("outputClassification.npy")
		return inputData, outputClassification

	data_base_path = "data/by_class"

	characters = os.listdir(data_base_path)

	for char in characters:
		char_training_path = data_base_path + "/" + char + "/train_" + char

		if os.path.isdir(char_training_path):
			training_pictures = os.listdir(char_training_path)

			for tpic in training_pictures:
				tpic_path = char_training_path + "/" + tpic

				image = Image.open(tpic_path)
				numpydata = numpy.asarray(image)

				inputData.append(numpydata)
				outputClassification.append(char)

				break
	
	numpy.save("inputData.npy", inputData)
	numpy.save("outputClassification.npy", outputClassification)

	return inputData, outputClassification
