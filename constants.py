import utils
import sys

import numpy as np

characters = ["0", "1"]
char_image_count = 1000

characters_folders_base_path = "source-data/by_class"
characters_folders = list(map(lambda x: utils.hex_extractor(x), characters))

verbose = True

if verbose:
    np.set_printoptions(threshold=sys.maxsize)
