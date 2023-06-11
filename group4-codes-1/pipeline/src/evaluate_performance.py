import logging
from pathlib import Path
from typing import Any
from typing import Optional

import yaml
import numpy.typing
import numpy as np
import pandas as pd
from sklearn import metrics

logger = logging.getLogger(__name__)


def predict(test: pd.DataFrame, tmo, bin_predict_col: Optional[str] = None,) -> pd.DataFrame:
    """Predicts binary outcome.

    Args:
        test: Test data containing features
        tmo: Trained model object
        bin_predict_col: If provided, binary outcome will be predicted and populated to this column. Defaults to None.

    Returns:
        Data containing features and predicted binary outcome
    """
    x_test = test.iloc[:, :-1]
    
    if bin_predict_col is not None:
        test[bin_predict_col] = tmo.predict(x_test)
    
    return test


def calculate_accuracy(y_true: numpy.typing.ArrayLike, y_pred: numpy.typing.ArrayLike) -> float:
    """Calculates the accuracy based on true and predicted binary values.

    Args:
        y_true: True binary values
        y_pred_bin: Predicted binary values

    Returns:
        metric accuracy
    """
    accuracy = metrics.accuracy_score(y_true, y_pred)

    logger.info("The accuracy was %0.2f", accuracy)

    return accuracy


def calculate_f1(y_true: numpy.typing.ArrayLike, y_pred: numpy.typing.ArrayLike) -> float:
    """Calculates the accuracy based on true and predicted binary values.

    Args:
        y_true: True binary values
        y_pred_bin: Predicted binary values

    Returns:
        metric f1 score
    """
    f1 = metrics.f1_score(y_true, y_pred)

    logger.info("The f1-score was %0.2f", f1)

    return f1


def confusion_matrix(
    y_true: numpy.typing.ArrayLike, y_pred: numpy.typing.ArrayLike
) -> pd.DataFrame:
    """Calculates the confusion matrix based on true and predicted binary values.

    Args:
        y_true: True binary values
        y_pred_bin: Predicted binary values

    Returns:
        Confusion matrix
    """
    confusion = metrics.confusion_matrix(y_true, y_pred)
    confusion = pd.DataFrame(
        confusion,
        index=["Actual negative", "Actual positive"],
        columns=["Predicted negative", "Predicted positive"],
    )

    logger.info("\n%s", str(confusion))

    return confusion


def evaluate_performance(data: pd.DataFrame, config: dict) -> dict[str, Any]:
    """Evaluate performance of model

    Args:
        data: Scored model
        config: Configuration for the evaluate_performance function; see example config file for structure

    Returns:
        Dictionary of metrics and values
    """

    ypred = data[config["bin_predict_col"]].values
    ytrue = data[config["target_col"]].values

    metrics_to_calc = config["metrics"]

    metric_output: dict[str, Any] = {}

    if "accuracy" in metrics_to_calc:
        accuracy = calculate_accuracy(ytrue, ypred)
        metric_output["accuracy"] = accuracy
    if "f1" in metrics_to_calc:
        f1 = calculate_f1(ytrue, ypred)
        metric_output["f1"] = f1
    if "confusion_matrix" in metrics_to_calc:
        confusion = confusion_matrix(ytrue, ypred)
        metric_output["confusion_matrix"] = confusion
    for metric in metrics_to_calc:
        if metric not in ["f1", "accuracy", "confusion_matrix"]:
            logger.warning("No code exists to calculate %s", metric)

    return metric_output

def save_metrics(metrics: dict[str, Any], save_path: Path):
    """Save the metrics to disk as a yaml file

    Args:
        metrics_dict: Metrics as produced by evaluate_performance
        out_file: Path at which the output will be saved.
    """
    # Modify format of saved metrics
    for key, value in metrics.items():
        if isinstance(value, np.generic):
            metrics[key] = value.item()
        elif isinstance(value, np.ndarray):
            metrics[key] = value.tolist()
        elif isinstance(value, pd.DataFrame):
            metrics[key] = value.to_dict()

    try:
        with open(save_path, 'w') as f:
            yaml.dump(metrics, f, default_flow_style=False)
        logger.info('Metrics data is save to %s', save_path)
    except FileNotFoundError as e:
        logger.warning('FileNotFoundError occurred writing processed data to file %s: %s', save_path, e)
    