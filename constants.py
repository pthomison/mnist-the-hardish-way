import sys
import numpy as np

numbers = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
lowercase = ["a", "b", "c", "d", "e", "f", "g", "h", "i",
             "j", "k", "l", "m", "n", "o", "p", "q", "r",
             "s", "t", "u", "v", "w", "x", "y", "z"]
uppercase = ["A", "B", "C", "D", "E", "F", "G", "H", "I",
             "J", "K", "L", "M", "N", "O", "P", "Q", "R",
             "S", "T", "U", "V", "W", "X", "Y", "Z"]

# characters = ["0", "1"]
# characters = numbers + lowercase + uppercase
characters = numbers
# characters = lowercase
# characters = uppercase
# char_image_count = 10
input_shape = (128, 128, 1)
epochs = 10

characters_folders_base_path = "source-data/by_class"
characters_folders = ["source-data/by_class/" + hex(ord(x))[2:] for x in characters]

cache_base_location = "cache-data"
model_cache_location = f"{cache_base_location}/model"
data_cache_location = f"{cache_base_location}/characters"

train_regex = r'^train_'
hsf_folder_regex = r'^hsf_\d$'
all_folder_regex = r'^(hsf_\d$)|(train_)'

training_regex = train_regex
validation_regex = hsf_folder_regex

verbose = False
persistence = True

if verbose:
    np.set_printoptions(threshold=sys.maxsize)
