import os
import re
import time

from PIL import Image
import numpy

import constants


def character_folder_lookup(character):
	return constants.characters_folders[constants.characters.index(character)]


def picture_folder_walk(character, folder_regex, func):
	character_folder = constants.characters_folders[constants.characters.index(character)]
	outputs = []
	for folder in os.scandir(character_folder):
		if re.match(folder_regex, folder.name):
			outputs.append(func(folder.path))
	return outputs


def picture_walk(character, folder_regex, func):
	character_folder = constants.characters_folders[constants.characters.index(character)]
	outputs = []
	for folder in os.scandir(character_folder):
		if re.match(folder_regex, folder.name):
			for picture in os.scandir(folder):
				outputs.append(func(picture.path))
	return outputs


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

	predictions = ""

	#
	# def load_pictures(picture_path):
	# 	total_count += 1
	# 	prediction = model.predict(m, load_image_file(hsf_picture.path))
	# 	predictions += prediction + ","
	#
	# 	if prediction == character:
	# 		correct_count += 1
	#
	# 	if total_count >= 100:
	# 		break

	# print(f"{character}: ", end="")

	total_count = 0
	correct_count = 0

	start = time.perf_counter()

	picture_data = hsf_walk(character, lambda hsf_picture_path: print(hsf_picture_path))

	# print(len(picture_data))
	# print(picture_data[0].shape)

	# for hsf_folder in os.scandir(character_folder):
	# 	if re.match(r'^hsf_\d$', hsf_folder.name):
	# 		for hsf_picture in os.scandir(hsf_folder):
	# 			total_count += 1
	# 			prediction = model.predict(m, load_image_file(hsf_picture.path))
	# 			predictions += prediction + ","
	#
	# 			if prediction == character:
	# 				correct_count += 1
	#
	# 			if total_count >= 100:
	# 				break

	# print(f"({str(time.perf_counter() - start)} seconds){character} ({str(correct_count)}/{str(total_count)}): {predictions}")


	# print()
	#
	# print(correct_count + "/" + correct_count)
