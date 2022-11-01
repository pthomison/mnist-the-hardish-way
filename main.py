#!/usr/bin/env python3

import data
import model


def main():
    training, testing = data.load_data()

    m = model.create(training[0], training[1])

    m.evaluate(training[0], training[1], verbose=2)
    m.evaluate(testing[0], testing[1], verbose=2)

    print(m.summary())


if __name__ == '__main__':
    main()
