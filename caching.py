import os
import glob

import numpy as np

import utils
import constants


def ensure_caching():
    if not os.path.exists(constants.data_cache_location):
        os.makedirs(constants.data_cache_location)
    for char in constants.characters:
        cache_character(char)


def cache_character(character):
    training_image_count = len(utils.picture_walk(character, constants.training_regex, lambda x: 1))
    # validation_image_count = utils.picture_walk(character, constants.validation_regex, lambda x: 1)

    # total_image_count = len(training_image_count) + len(hsf_image_count)

    training_char_cache_path = f"{constants.data_cache_location}/{character}-training.npy"
    training_char_cache_array = np.empty((training_image_count, 128, 128, 1), dtype=np.uint8)
    
    if not os.path.exists(training_char_cache_path):
        index = 0
        for arr in utils.picture_folder_walk(character, constants.training_regex, slurp_image_folder):
            for datum in arr:
                training_char_cache_array[index] = datum
                index += 1

        np.save(training_char_cache_path, training_char_cache_array)


def slurp_image_folder(folder_path):
    image_paths = glob.glob(folder_path + "/*.png")
    output_array = np.empty((len(image_paths), 128, 128, 1), dtype=np.uint8)

    if len(image_paths) == 0:
        print(f"error (slurp_image_folder): {folder_path}")
        exit(1)

    for i in range(len(image_paths)):
        output_array[i] = utils.load_image_file(image_paths[i])

    return output_array