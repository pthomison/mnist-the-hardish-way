from PIL import Image
import numpy


def hex_extractor(character):
	return hex(ord(character))[2:]


def load_image_file(path):
	image = Image.open(path)
	image_data_rgb = numpy.asarray(image)
	# only need one column for black and white pictures
	image_data_bw = numpy.compress([True, False, False], image_data_rgb, axis=2)
	return image_data_bw
