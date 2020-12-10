import os

import autokeras as ak
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_files
from tensorflow.keras.datasets import imdb


class AutoMLNeuralNetwork:

    def image_classification():
        clf = ak.ImageClassifier(
        overwrite=True,
        max_trials=1)

    def image_regression():
        reg = ak.ImageRegressor(
        overwrite=True,
        max_trials=1)

    
    def text_classification():
        clf = ak.TextClassifier(
        overwrite=True,
        max_trials=1)

    def text_regression():
        reg = ak.TextRegressor(
        overwrite=True,
        max_trials=1)

    def data_classification():
        clf = ak.StructuredDataClassifier(
        overwrite=True,
        max_trials=1)

    def data_regression():
        reg = ak.StructuredDataRegressor(
        overwrite=True,
        max_trials=1)