#!/usr/bin/env python3

import os

import tensorflow as tf
import numpy as np

import constants


def create(training_data, training_classifications):
	if os.path.exists("cache-data/model") and constants.persistence:
		return tf.keras.models.load_model("cache-data/model")

	# standardize to floating point between 0 and 1
	training_data = training_data / 255.0

	model = tf.keras.Sequential(
		[
			tf.keras.Input(shape=constants.input_shape),
			tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
			tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
			tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
			tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dropout(0.5),
			tf.keras.layers.Dense(len(constants.characters), activation="softmax"),
		]
	)

	model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

	model.fit(training_data, training_classifications, epochs=constants.epochs)

	if constants.persistence:
		model.save("cache-data/model")

	return model


def predict(model, input_data):
	prediction = model(np.expand_dims(input_data, 0))[0]

	max_index = 0
	max_index_value = 0

	for label in range(len(prediction)):
		if prediction[label] > max_index_value:
			max_index = label
			max_index_value = prediction[label]

	print(constants.characters[max_index])

