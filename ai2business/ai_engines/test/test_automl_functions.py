from unittest import mock

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

from ai2business.ai_engines import automl_neural_network as an

data = load_boston()
x_train, x_test, y_train, y_test = train_test_split(
    data.data,
    data.target,
    test_size=0.33,
    random_state=42,
)


@mock.patch("ai2business.ai_engines.automl_neural_network.AutoMLFit")
def test_call_image_classification(AutoMLFit):

    context = an.AutoMLPipeline(an.ImageClassification())
    context.run_automl()
    context.train = an.AutoMLFit(x_train, y_train, batch_size=32, epochs=1)
    context.run_automl()
    assert an.AutoMLFit.is_called


@mock.patch("ai2business.ai_engines.automl_neural_network.AutoMLFit")
def test_call_image_regression(AutoMLFit):

    context = an.AutoMLPipeline(an.ImageRegression())
    context.run_automl()
    context.train = an.AutoMLFit(x_train, y_train, batch_size=32, epochs=1)
    context.run_automl()
    assert an.AutoMLFit.is_called


@mock.patch("ai2business.ai_engines.automl_neural_network.AutoMLFit")
def test_call_text_classification(AutoMLFit):

    context = an.AutoMLPipeline(an.TextClassification())
    context.run_automl()
    context.train = an.AutoMLFit(x_train, y_train, batch_size=32, epochs=1)
    context.run_automl()
    assert an.AutoMLFit.is_called


@mock.patch("ai2business.ai_engines.automl_neural_network.AutoMLFit")
def test_call_text_regression(AutoMLFit):

    context = an.AutoMLPipeline(an.TextRegression())
    context.run_automl()
    context.train = an.AutoMLFit(x_train, y_train, batch_size=32, epochs=1)
    context.run_automl()
    assert an.AutoMLFit.is_called


@mock.patch("ai2business.ai_engines.automl_neural_network.AutoMLFit")
def test_call_data_classification(AutoMLFit):

    context = an.AutoMLPipeline(an.DataClassification())
    context.run_automl()
    context.train = an.AutoMLFit(x_train, y_train, batch_size=32, epochs=1)
    context.run_automl()
    assert an.AutoMLFit.is_called


@mock.patch("ai2business.ai_engines.automl_neural_network.AutoMLFit")
def test_call_data_regression(AutoMLFit):

    context = an.AutoMLPipeline(an.DataRegression())
    context.run_automl()
    context.train = an.AutoMLFit(x_train, y_train, batch_size=32, epochs=1)
    context.run_automl()
    assert an.AutoMLFit.is_called


@mock.patch("ai2business.ai_engines.automl_neural_network.AutoMLFit")
def test_call_timeseries_forecast(AutoMLFit):

    context = an.AutoMLPipeline(an.TimeseriesForecaster())
    context.run_automl()
    context.train = an.AutoMLFit(x_train, y_train, batch_size=32, epochs=1)
    context.run_automl()
    assert an.AutoMLFit.is_called


@mock.patch("ai2business.ai_engines.automl_neural_network.AutoMLFit")
@mock.patch("ai2business.ai_engines.automl_neural_network.AutoMLPredict")
def test_call_prediction(AutoMLFit, AutoMLPredict):

    context = an.AutoMLPipeline(an.DataClassification())
    context.run_automl()
    context.train = an.AutoMLFit(x_train, y_train, batch_size=32, epochs=1)
    context.run_automl()
    context.train = an.AutoMLPredict(x_train, batch_size=32)
    context.run_automl()
    assert an.AutoMLPredict.is_called


@mock.patch("ai2business.ai_engines.automl_neural_network.AutoMLFit")
@mock.patch("ai2business.ai_engines.automl_neural_network.AutoMLEvaluate")
def test_call_evaluation(AutoMLFit, AutoMLEvaluate):

    context = an.AutoMLPipeline(an.DataClassification())
    context.run_automl()
    context.train = an.AutoMLFit(x_train, y_train, batch_size=32, epochs=1)
    context.run_automl()
    context.train = an.AutoMLEvaluate(x_test, y_test, batch_size=32)
    context.run_automl()
    assert an.AutoMLEvaluate.is_called


@mock.patch("ai2business.ai_engines.automl_neural_network.AutoMLFit")
@mock.patch("ai2business.ai_engines.automl_neural_network.AutoMLSave")
def test_call_save(AutoMLFit, AutoMLSave):

    context = an.AutoMLPipeline(an.DataClassification())
    context.run_automl()
    context.train = an.AutoMLFit(x_train, y_train, batch_size=32, epochs=1)
    context.run_automl()
    context.train = an.AutoMLSave("dummy")
    context.run_automl()
    assert an.AutoMLSave.is_called
