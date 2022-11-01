#!/usr/bin/env python3

import tensorflow as tf
import constants


def create(training_data, training_classifications, epoch_count=10):
	# standardize to floating point between 0 and 1
	training_data = training_data / 255.0

	model = tf.keras.models.Sequential([
		tf.keras.layers.Flatten(input_shape=(128, 128)),
		tf.keras.layers.Dense(128, activation='relu'),
		tf.keras.layers.Dropout(0.2),
		tf.keras.layers.Dense(len(constants.characters))
	])

	loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

	model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

	model.fit(training_data, training_classifications, epochs=epoch_count)

	return model
