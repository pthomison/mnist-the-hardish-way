import os
import re

from PIL import Image
import numpy

import constants
import model


def hex_extractor(character):
	return hex(ord(character))[2:]


def load_image_file(path):
	image = Image.open(path)
	image_data_rgb = numpy.asarray(image)
	# only need one column for black and white pictures
	image_data_bw = numpy.compress([True, False, False], image_data_rgb, axis=2)
	return image_data_bw


def test_character(m, character):
	character_folder = constants.characters_folders[constants.characters.index(character)]

	print(f"{character}: ", end="")

	for hsf_folder in os.scandir(character_folder):
		if re.match('^hsf_\d$', hsf_folder.name):
			prediction = model.predict(m, load_image_file(f"{hsf_folder.path}/{hsf_folder.name}_00000.png"))
			print(prediction, end=",")

	print()
