#!/usr/bin/env python3

import data
import model
import utils

import numpy as np


def main():
    training, testing = data.load_data()

    m = model.create(training[0], training[1])

    m.evaluate(training[0], training[1], verbose=2)
    m.evaluate(testing[0], testing[1], verbose=2)

    zero_data = utils.load_image_file("source-data/by_class/30/hsf_1/hsf_1_00000.png")
    one_data = utils.load_image_file("source-data/by_class/31/hsf_1/hsf_1_00000.png")
    two_data = utils.load_image_file("source-data/by_class/32/hsf_1/hsf_1_00000.png")
    three_data = utils.load_image_file("source-data/by_class/33/hsf_1/hsf_1_00000.png")
    four_data = utils.load_image_file("source-data/by_class/34/hsf_1/hsf_1_00000.png")
    five_data = utils.load_image_file("source-data/by_class/35/hsf_1/hsf_1_00000.png")
    six_data = utils.load_image_file("source-data/by_class/36/hsf_1/hsf_1_00000.png")
    seven_data = utils.load_image_file("source-data/by_class/37/hsf_1/hsf_1_00000.png")
    eight_data = utils.load_image_file("source-data/by_class/38/hsf_1/hsf_1_00000.png")
    nine_data = utils.load_image_file("source-data/by_class/39/hsf_1/hsf_1_00000.png")

    model.predict(m, zero_data)
    model.predict(m, one_data)
    model.predict(m, two_data)
    model.predict(m, three_data)
    model.predict(m, four_data)
    model.predict(m, five_data)
    model.predict(m, six_data)
    model.predict(m, seven_data)
    model.predict(m, eight_data)
    model.predict(m, nine_data)


if __name__ == '__main__':
    main()
