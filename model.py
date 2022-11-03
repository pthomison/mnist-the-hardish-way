#!/usr/bin/env python3

import os

import tensorflow as tf
import numpy as np

import constants


def create(training_data, training_classifications):
	# standardize to floating point between 0 and 1
	training_data = training_data / 255.0

	if os.path.exists(constants.model_cache_location) and constants.persistence:
		model = tf.keras.models.load_model(constants.model_cache_location)
	else:
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

	callbacks = [
		tf.keras.callbacks.EarlyStopping(
			# Stop training when `val_loss` is no longer improving
			monitor="accuracy",
			# "no longer improving" being defined as "no better than 1e-2 less"
			min_delta=1e-2,
			# "no longer improving" being further defined as "for at least 2 epochs"
			patience=2,
			verbose=1,
		)
	]

	model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

	model.fit(training_data, training_classifications, callbacks=callbacks, epochs=constants.epochs)

	if constants.persistence:
		model.save(constants.model_cache_location)

	return model


def predict(model, input_data):
	prediction = model(np.expand_dims(input_data, 0))[0]

	max_index = 0
	max_index_value = 0

	for label in range(len(prediction)):
		if prediction[label] > max_index_value:
			max_index = label
			max_index_value = prediction[label]

	return constants.characters[max_index]

