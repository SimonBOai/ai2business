from abc import ABC, abstractmethod
from typing import Any

import autokeras as ak
from tensorflow.keras.models import load_model


class AutoMLModels:
    def __init__(
        self,
        directory: str = None,
        loss: str = None,
        objective: str = "val_loss",
        overwrite: bool = False,
        project_name: str = "AutoML_DeepLearning",
        max_model_size: int = None,
        max_trials: int = 100,
        metrics: str = None,
        seed: int = None,
        tuner: str = None,
    ) -> None:
        """[summary]

        Args:
            directory (str, optional): [description]. Defaults to None.
            loss (str, optional): [description]. Defaults to None.
            objective (str, optional): [description]. Defaults to "val_loss".
            overwrite (bool, optional): [description]. Defaults to False.
            project_name (str, optional): [description]. Defaults to "AutoML_DeepLearning".
            max_model_size (int, optional): [description]. Defaults to None.
            max_trials (int, optional): [description]. Defaults to 100.
            metrics (str, optional): [description]. Defaults to None.
            seed (int, optional): [description]. Defaults to None.
            tuner (str, optional): [description]. Defaults to None.
        """
        self.directory = directory
        self.loss = loss
        self.objective = objective
        self.overwrite = overwrite
        self.project_name = project_name
        self.max_model_size = max_model_size
        self.max_trials = max_trials
        self.metrics = metrics
        self.seed = seed
        self.tuner = tuner

    def image_classification(
        self, num_classes: int = None, multi_label: bool = False, **kwargs
    ):
        """[summary]

        Args:
            num_classes (int, optional): [description]. Defaults to None.
            multi_label (bool, optional): [description]. Defaults to False.
        """
        return ak.ImageClassifier(
            num_classes=num_classes,
            multi_label=multi_label,
            loss=self.loss,
            metrics=self.metrics,
            project_name=self.project_name,
            max_trials=self.max_trials,
            directory=self.directory,
            objective=self.objective,
            tuner=self.tuner,
            overwrite=self.overwrite,
            seed=self.seed,
            max_model_size=self.max_model_size,
            **kwargs,
        )

    def image_regression(self, output_dim: int = None, **kwargs):
        """[summary]

        Args:
            output_dim (int, optional): [description]. Defaults to None.
        """
        return ak.ImageRegressor(
            output_dim=output_dim,
            loss=self.loss,
            metrics=self.metrics,
            project_name=self.project_name,
            max_trials=self.max_trials,
            directory=self.directory,
            objective=self.objective,
            tuner=self.tuner,
            overwrite=self.overwrite,
            seed=self.seed,
            max_model_size=self.max_model_size,
            **kwargs,
        )

    def text_classification(
        self, num_classes: int = None, multi_label: bool = False, **kwargs
    ):
        """[summary]

        Args:
            num_classes (int, optional): [description]. Defaults to None.
            multi_label (bool, optional): [description]. Defaults to False.
        """
        return ak.TextClassifier(
            num_classes=num_classes,
            multi_label=multi_label,
            loss=self.loss,
            metrics=self.metrics,
            project_name=self.project_name,
            max_trials=self.max_trials,
            directory=self.directory,
            objective=self.objective,
            tuner=self.tuner,
            overwrite=self.overwrite,
            seed=self.seed,
            max_model_size=self.max_model_size,
            **kwargs,
        )

    def text_regression(self, output_dim: int = None, **kwargs):
        """[summary]

        Args:
            output_dim (int, optional): [description]. Defaults to None.
        """
        return ak.TextRegressor(
            output_dim=output_dim,
            loss=self.loss,
            metrics=self.metrics,
            project_name=self.project_name,
            max_trials=self.max_trials,
            directory=self.directory,
            objective=self.objective,
            tuner=self.tuner,
            overwrite=self.overwrite,
            seed=self.seed,
            max_model_size=self.max_model_size,
            **kwargs,
        )

    def data_classification(
        self,
        column_names: list = None,
        column_types: dict = None,
        num_classes: int = None,
        multi_label: bool = False,
        **kwargs,
    ):
        """[summary]

        Args:
            column_names (list, optional): [description]. Defaults to None.
            column_types (dict, optional): [description]. Defaults to None.
            num_classes (int, optional): [description]. Defaults to None.
            multi_label (bool, optional): [description]. Defaults to False.
        """
        return ak.StructuredDataClassifier(
            column_names=column_names,
            column_types=column_types,
            num_classes=num_classes,
            multi_label=multi_label,
            loss=self.loss,
            metrics=self.metrics,
            project_name=self.project_name,
            max_trials=self.max_trials,
            directory=self.directory,
            objective=self.objective,
            tuner=self.tuner,
            overwrite=self.overwrite,
            seed=self.seed,
            max_model_size=self.max_model_size,
            **kwargs,
        )

    def data_regression(
        self,
        column_names: list = None,
        column_types: dict = None,
        output_dim: int = None,
        **kwargs,
    ):
        """[summary]

        Args:
            column_names (list, optional): [description]. Defaults to None.
            column_types (dict, optional): [description]. Defaults to None.
            output_dim (int, optional): [description]. Defaults to None.
        """
        return ak.StructuredDataRegressor(
            column_names=column_names,
            column_types=column_types,
            output_dim=output_dim,
            loss=self.loss,
            metrics=self.metrics,
            project_name=self.project_name,
            max_trials=self.max_trials,
            directory=self.directory,
            objective=self.objective,
            tuner=self.tuner,
            overwrite=self.overwrite,
            seed=self.seed,
            max_model_size=self.max_model_size,
            **kwargs,
        )

    def timeseries_forecaster(
        self,
        column_names: list = None,
        column_types: dict = None,
        output_dim: int = None,
        lookback: int = None,
        predict_from: int = 1,
        predict_until: int = None,
        **kwargs,
    ):
        """[summary]

        Args:
            column_names (list, optional): [description]. Defaults to None.
            column_types (dict, optional): [description]. Defaults to None.
            output_dim (int, optional): [description]. Defaults to None.
            lookback (int, optional): [description]. Defaults to None.
            predict_from (int, optional): [description]. Defaults to 1.
            predict_until (int, optional): [description]. Defaults to None.
        """
        return ak.TimeseriesForecaster(
            olumn_names=column_names,
            column_types=column_types,
            output_dim=output_dim,
            lookback=lookback,
            predict_from=predict_from,
            predict_until=predict_until,
            loss=self.loss,
            metrics=self.metrics,
            project_name=self.project_name,
            max_trials=self.max_trials,
            directory=self.directory,
            objective=self.objective,
            tuner=self.tuner,
            overwrite=self.overwrite,
            seed=self.seed,
            max_model_size=self.max_model_size,
            **kwargs,
        )

    def multi_model(self, inputs: list, outputs: list, **kwargs) -> None:

        return ak.AutoModel(
            inputs=inputs,
            outputs=outputs,
            project_name=self.project_name,
            max_trials=self.max_trials,
            directory=self.directory,
            objective=self.objective,
            tuner=self.tuner,
            overwrite=self.overwrite,
            seed=self.seed,
            max_model_size=self.max_model_size,
        )

    @staticmethod
    def load_model(model_name: str = "model_autokeras") -> None:
        """[summary]

        Args:
            model_name (str, optional): [description]. Defaults to "model_autokeras".
        """
        return load_model(f"{model_name}", custom_objects=ak.CUSTOM_OBJECTS)


class AutoMLRoutines:
    def __init__(self, model: AutoMLModels) -> None:
        self.model = model

    def save_model(self, model_name: str = "model_autokeras") -> None:
        """[summary]

        Args:
            model_name (str, optional): [description]. Defaults to "model_autokeras".
        """
        _model = self.model.export_model()
        try:
            _model.save(f"{model_name}", save_format="tf")
        except:
            _model.save(f"{model_name}.h5")

    def fit_model(
        self,
        X: Any = None,
        y: Any = None,
        batch_size: int = 32,
        epochs: int = None,
        callbacks: list = None,
        validation_split: float = 0.2,
        validation_data: Any = None,
        **kwargs,
    ):
        """[summary]

        Args:
            X (Any, optional): [description]. Defaults to None.
            y (Any, optional): [description]. Defaults to None.
            batch_size (int, optional): [description]. Defaults to 32.
            epochs (int, optional): [description]. Defaults to None.
            callbacks (list, optional): [description]. Defaults to None.
            validation_split (float, optional): [description]. Defaults to 0.2.
            validation_data (Any, optional): [description]. Defaults to None.
        """
        self.model = self.model.fit(
            x=X,
            y=y,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            validation_split=validation_split,
            validation_data=validation_data,
            **kwargs,
        )

    def predict_model(
        self,
        X,
        batch_size: int = 32,
        **kwargs,
    ):
        """[summary]

        Args:
            X ([type]): [description]
            batch_size (int, optional): [description]. Defaults to 32.
        """
        self.model = self.model.predict(
            x=X,
            batch_size=batch_size,
            **kwargs,
        )

    def evaluate_model(self, X: Any, y: Any = None, batch_size: int = 32, **kwargs):
        """[summary]

        Args:
            X (Any): [description]
            y (Any, optional): [description]. Defaults to None.
            batch_size (int, optional): [description]. Defaults to 32.
        """
        self.model = self.model.evaluate(X=X, y=y, batch_size=batch_size, **kwargs)


class AutoMLPipeline:
    def __init__(self, train):

        self._train = train
        self.trainset = None

    @property
    def train(self):

        return self._train

    @train.setter
    def train(self, train):
        #self.trainset = self._train
        self._train = train

    def run_automl(self):

        self.trainset = self._train.do_algorithm(self.trainset)


class Procedure(ABC):
    @abstractmethod
    def do_algorithm(self):
        pass




class TextClassifiction(Procedure):
    def do_algorithm(self,trainset):
        _ = trainset 
        model = AutoMLModels().text_classification()
        return AutoMLRoutines(model)

    


class ConcreteStrategyB(Procedure):
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
    def do_algorithm(self, trainset):
        
        trainset.fit_model(self.x_train, self.y_train)
        predicted_y = self.trainset.predict_model(self.x_test)
    
        print(self.trainset.evaluate_model(self.x_test, self.y_test))


if __name__ == "__main__":
    import os
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.datasets import imdb
    from sklearn.datasets import load_files

    dataset = tf.keras.utils.get_file(
        fname="aclImdb.tar.gz",
        origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
        extract=True,
    )

    # set path to dataset
    IMDB_DATADIR = os.path.join(os.path.dirname(dataset), "aclImdb")

    classes = ["pos", "neg"]
    train_data = load_files(
        os.path.join(IMDB_DATADIR, "train"), shuffle=True, categories=classes
    )
    test_data = load_files(
        os.path.join(IMDB_DATADIR, "test"), shuffle=False, categories=classes
    )

    x_train = np.array(train_data.data)
    y_train = np.array(train_data.target)
    x_test = np.array(test_data.data)
    y_test = np.array(test_data.target)

    context = AutoMLPipeline(TextClassifiction())
    print(type(context))
    #context.run_automl()

    #context.train = ConcreteStrategyB(x_train, y_train, x_test, y_test)
    #context.run_automl()
