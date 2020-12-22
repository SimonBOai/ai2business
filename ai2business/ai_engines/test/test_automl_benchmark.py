import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from ai2business.ai_engines import automl_neural_network as an


def test_runtime_dataclassifier():

    train_file_path = tf.keras.utils.get_file(
        "train.csv", "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
    )
    test_file_path = tf.keras.utils.get_file(
        "test.csv", "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"
    )

    data_train = pd.read_csv(train_file_path)
    data_test = pd.read_csv(test_file_path)

    x_train = data_train.drop(columns="survived")
    y_train = data_train["survived"]
    x_test = data_test.drop(columns="survived")
    y_test = data_test["survived"]

    context = an.AutoMLPipeline(an.DataClassification(max_trials=4))
    context.run_automl()
    context.train = an.AutoMLFit(x_train, y_train, batch_size=32, epochs=100)
    context.run_automl()
    context.train = an.AutoMLEvaluate(x_test, y_test, batch_size=32)
    context.run_automl()
    context.train = an.AutoMLPredict(x_train, batch_size=32)
    context.run_automl()
    assert context.return_automl["model"] != None
    assert isinstance(context.return_automl["prediction"], np.ndarray)
    assert isinstance(context.return_automl["evaluation"], list)


def test_runtime_dataregression():

    data = fetch_california_housing()
    x_train, y_train, x_test, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.33,
        random_state=42,
    )
    context = an.AutoMLPipeline(an.DataRegression(max_trials=4))
    context.run_automl()
    context.train = an.AutoMLFit(x_train, y_train, batch_size=32, epochs=100)
    context.run_automl()
    context.train = an.AutoMLEvaluate(x_test, y_test, batch_size=32)
    context.run_automl()
    context.train = an.AutoMLPredict(x_train, batch_size=32)
    context.run_automl()
    assert context.return_automl["model"] != None
    assert isinstance(context.return_automl["prediction"], np.ndarray)
    assert isinstance(context.return_automl["evaluation"], list)


def test_return_train():

    context = an.AutoMLPipeline(an.DataRegression(max_trials=4))
    assert context.train == an.AutoMLPipeline(an.DataRegression(max_trials=4))
