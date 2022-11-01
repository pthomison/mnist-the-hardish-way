import utils
import sys

import numpy as np

characters = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
char_image_count = 1000
input_shape = (128, 128, 1)
epochs = 10

characters_folders_base_path = "source-data/by_class"
characters_folders = list(map(lambda x: utils.hex_extractor(x), characters))

verbose = True
persistence = True

if verbose:
    np.set_printoptions(threshold=sys.maxsize)
