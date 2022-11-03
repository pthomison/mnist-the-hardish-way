#!/usr/bin/env python3
import constants
import data
import model
import utils


def main():
    training, testing = data.load_data()

    m = model.create(training[0], training[1])

    m.evaluate(training[0], training[1], verbose=2)
    m.evaluate(testing[0], testing[1], verbose=2)

    for char in constants.characters:
        utils.test_character(m, char)


if __name__ == '__main__':
    main()
