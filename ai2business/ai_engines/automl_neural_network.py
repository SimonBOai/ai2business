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
        max_trials: int = 1,
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
    ) -> ak.ImageClassifier:
        """image_classification [summary]

        [extended_summary]

        Args:
            num_classes (int, optional): [description]. Defaults to None.
            multi_label (bool, optional): [description]. Defaults to False.

        Returns:
            ak.ImageClassifier: [description]
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

    def image_regression(self, output_dim: int = None, **kwargs) -> ak.ImageRegressor:
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
    ) -> ak.TextClassifier:
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

    def text_regression(self, output_dim: int = None, **kwargs) -> ak.TextRegressor:
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
    ) -> ak.StructuredDataClassifier:
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
    ) -> ak.StructuredDataRegressor:
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
    ) -> ak.TimeseriesForecaster:
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

    def multi_model(self, inputs: list, outputs: list, **kwargs) -> ak.AutoModel:

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
    def load_model(model_name: str = "model_autokeras"):
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
        X_train: Any,
        y_train: Any,
        batch_size: int = 32,
        epochs: int = None,
        callbacks: list = None,
        validation_split: float = 0.2,
        validation_data: Any = None,
        **kwargs,
    ) -> None:
        """[summary]

        Args:
            X_train (Any): [description]. Defaults to None.
            y_train (Any): [description]. Defaults to None.
            batch_size (int, optional): [description]. Defaults to 32.
            epochs (int, optional): [description]. Defaults to None.
            callbacks (list, optional): [description]. Defaults to None.
            validation_split (float, optional): [description]. Defaults to 0.2.
            validation_data (Any, optional): [description]. Defaults to None.
        """
        self.model.fit(
            x=X_train,
            y=y_train,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            validation_split=validation_split,
            validation_data=validation_data,
            **kwargs,
        )
        return self.model

    def predict_model(
        self,
        X_train: Any,
        batch_size: int = 32,
        **kwargs,
    ) -> None:
        """predict_model [summary]

        [extended_summary]

        Args:
            X_train (Any): [description]
            batch_size (int, optional): [description]. Defaults to 32.
        """
        return self.model.predict(
            x=X_train,
            batch_size=batch_size,
            **kwargs,
        )

    def evaluate_model(
        self, X_test: Any, y_test: Any = None, batch_size: int = 32, **kwargs
    ) -> None:
        """[summary]

        Args:
            X_test (Any): [description]
            y_test (Any, optional): [description]. Defaults to None.
            batch_size (int, optional): [description]. Defaults to 32.
        """
        return self.model.evaluate(x=X_test, y=y_test, batch_size=batch_size, **kwargs)


class AutoMLPipeline:
    def __init__(self, train: Any) -> None:

        self._train = train
        self.automl_model = {"model": None, "prediction": None, "evaluation": None}

    @property
    def train(self) -> Any:

        return self._train

    @train.setter
    def train(self, train: Any) -> None:
        self._train = train

    def run_automl(self):

        self.automl_model = self._train.perform_job(self.automl_model)

    @property
    def return_automl(self) -> dict:
        return self.automl_model


class Procedure(ABC):
    @abstractmethod
    def perform_job(self):
        pass


class ImageClassification(Procedure):
    def perform_job(self, automl_model: dict):
        _ = automl_model
        model = AutoMLModels().image_classification()
        return {"model": AutoMLRoutines(model), "prediction": None, "evaluation": None}


class ImageRegression(Procedure):
    def perform_job(self, automl_model: dict):
        _ = automl_model
        model = AutoMLModels().image_regression()
        return {"model": AutoMLRoutines(model), "prediction": None, "evaluation": None}


class TextClassification(Procedure):
    def perform_job(self, automl_model: dict):
        _ = automl_model
        model = AutoMLModels().text_classification()
        return {"model": AutoMLRoutines(model), "prediction": None, "evaluation": None}


class TextRegression(Procedure):
    def perform_job(self, automl_model: dict):
        _ = automl_model
        model = AutoMLModels().text_regression()
        return {"model": AutoMLRoutines(model), "prediction": None, "evaluation": None}


class DataClassification(Procedure):
    def perform_job(self, automl_model: dict):
        _ = automl_model
        model = AutoMLModels().data_classification()
        return {"model": AutoMLRoutines(model), "prediction": None, "evaluation": None}


class DataRegression(Procedure):
    def perform_job(self, automl_model: dict):
        _ = automl_model
        model = AutoMLModels().data_regression()
        return {"model": AutoMLRoutines(model), "prediction": None, "evaluation": None}


class TimeseriesForecaster(Procedure):
    def perform_job(self, automl_model: dict):
        _ = automl_model
        model = AutoMLModels().timeseries_forecaster()
        return {"model": AutoMLRoutines(model), "prediction": None, "evaluation": None}


class MultiModel(Procedure):
    def perform_job(self, automl_model: dict):
        _ = automl_model
        model = AutoMLModels().multi_model()
        return {"model": AutoMLRoutines(model), "prediction": None, "evaluation": None}


class AutoMLFit(Procedure):
    def __init__(
        self,
        X_train: Any,
        y_train: Any,
        batch_size: int = 32,
        epochs: int = None,
        callbacks: list = None,
        validation_split: float = 0.2,
        validation_data: Any = None,
        **kwargs,
    ) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.epochs = epochs
        self.callbacks = callbacks
        self.validation_split = validation_split
        self.validation_data = validation_data
        self.kwargs = kwargs

    def perform_job(self, automl_model: dict):

        return {
            "model": automl_model["model"].fit_model(
                X_train=self.X_train,
                y_train=self.y_train,
                epochs=self.epochs,
                callbacks=self.callbacks,
                validation_split=self.validation_split,
                validation_data=self.validation_data,
                **self.kwargs,
            ),
            "prediction": None,
            "evaluation": None,
        }


class AutoMLPredict(Procedure):
    def __init__(
        self,
        X_train: Any,
        batch_size: int = 32,
        **kwargs,
    ) -> None:
        self.X_train = X_train
        self.batch_size = batch_size
        self.kwargs = kwargs

    def perform_job(self, automl_model: dict):
        return {
            "model": automl_model["model"],
            "prediction": automl_model["model"].predict(
                x=self.X_train, batch_size=self.batch_size, **self.kwargs
            ),
            "evaluation": automl_model["evaluation"],
        }


class AutoMLEvaluate(Procedure):
    def __init__(
        self, X_test: Any, y_test: Any = None, batch_size: int = 32, **kwargs
    ) -> None:
        self.X_test = X_test
        self.y_test = y_test
        self.batch_size = batch_size
        self.kwargs = kwargs

    def perform_job(self, automl_model: dict):
        return {
            "model": automl_model["model"],
            "prediction": automl_model["prediction"],
            "evaluation": automl_model["model"].evaluate(
                x=self.X_test,
                y=self.y_test,
                batch_size=self.batch_size,
                **self.kwargs,
            ),
        }


class AutoMLSave(Procedure):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def perform_job(self, automl_model: dict):
        _model = automl_model.export_model()
        try:
            _model.save(f"{self.model_name}", save_format="tf")
        except:
            _model.save(f"{self.model_name}.h5")
