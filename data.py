import os
import numpy as np

import constants
import utils


def load_data():
    if saved_tuple_exists("training") and saved_tuple_exists("testing"):
        local_training_tuple = load_data_tuple("training")
        local_testing_tuple = load_data_tuple("testing")
    else:
        local_training_tuple = load_training_data_from_folder(lambda char: "train_" + char)
        save_data_tuple(local_training_tuple, "training")

        local_testing_tuple = load_training_data_from_folder(lambda char: "hsf_0")
        save_data_tuple(local_testing_tuple, "testing")

    return local_training_tuple, local_testing_tuple


def load_training_data_from_folder(folder_func):
    input_data = np.empty((0, 128, 128), dtype=np.uint8)
    output_classification = np.empty(0)
    index = 0

    for char in constants.characters_folders:
        print(char)

        folder_path = constants.characters_folders_base_path + "/" + char + "/" + folder_func(char)
        folder_contents = os.listdir(folder_path)

        for i in range(constants.char_image_count):
            picture_filename = folder_contents[i]

            image_data = utils.load_image_file(folder_path + "/" + picture_filename)
            input_data = np.insert(input_data, index, image_data, axis=0)
            output_classification = np.insert(output_classification, index, constants.characters_folders.index(char))

            index += 1

    return input_data, output_classification


def save_data_tuple(data_tuple, path_prefix):
    np.save("cache-data/" + path_prefix + "-data.npy", data_tuple[0])
    np.save("cache-data/" + path_prefix + "-classification.npy", data_tuple[1])


def load_data_tuple(path_prefix):
    data = np.load("cache-data/" + path_prefix + "-data.npy")
    classification = np.load("cache-data/" + path_prefix + "-classification.npy")

    return data, classification


def saved_tuple_exists(path_prefix):
    data_exists = os.path.exists("cache-data/" + path_prefix + "-data.npy")
    classification_exists = os.path.exists("cache-data/" + path_prefix + "-classification.npy")

    return data_exists and classification_exists
