import logging
from pathlib import Path
import pickle
import typing
import json

import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = {"RandomForestClassifier": RandomForestClassifier}


def train_test_split(
    df: pd.DataFrame,
    target_col: str,
    random_state: int,
    test_size: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame()]:
    """Splits dataframe into train and test feature sets and targets.

    Args:
        df: Data containing features and target
        target_col: Column containing target values/labels
        random_state: Random state to make split reproducible
        test_size: Fraction of set to randomly sample to create test data. 

    Returns:
        `pd.DataFrame`: Features for training dataset
        `pd.DataFrame`: Features for testing dataset
        `pd.Series`: True target values for training dataset
        `pd.Series`: True target values for testing dataset
        'pd.DataFrame': Min, max, mean, and std for each feature that will be used for inference
    """
    features = df[[c for c in df.columns if c != target_col]]
    target = df[target_col]
    
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        features, target, test_size=test_size, random_state=random_state
    )
    return x_train, x_test, y_train, y_test


def save_data(train: pd.DataFrame, test: pd.DataFrame, meta: pd.DataFrame(), out_dir: Path):
    """_summary_

    Args:
        train: Training data frame
        test: Testing data frame
        out_dir: Save path
    """

    # Save Training data to disk
    train_file = out_dir / "train.csv"
    train.to_csv(train_file, index=False)
    logger.info("Training dataset saved to %s", train_file)

    # Save Test data to disk
    test_file = out_dir / "test.csv"
    test.to_csv(test_file, index=False)
    logger.info("Testing dataset saved to %s", test_file)

    # Save Test data to disk
    meta_file = out_dir / "meta.csv"
    meta.to_csv(meta_file, index=False)
    logger.info("Training metadata saved to %s", meta_file)


def standardize_data(
    x_train: pd.DataFrame, 
    x_test: pd.DataFrame, 
    col_names: list
)-> tuple[pd.DataFrame, pd.Series]:
    """Standardize x_train and x_test on col_names.

    Args:
        x_train: x_train data
        x_test: x_test data
        col_names: column names to standardize

    Returns:
        `pd.DataFrame`: Features for standardized x_train dataset
        `pd.DataFrame`: Features for standardized x_test dataset
    """
    scaler = StandardScaler()
    x_train[col_names] = scaler.fit_transform(x_train[col_names])
    x_test[col_names] = scaler.transform(x_test[col_names])
    return x_train, x_test


def fit_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: dict
) -> sklearn.base.BaseEstimator:
    """_summary_

    Args:
        x_train: Features for training dataset
        y_train: True target values for training dataset
        param_grid: Dictionary containing keyword arguments for model hyperparameter tuning.
            For RandomForestClassifier, this is all values that will be provided to
            `sklearn.ensemble.RandomForestClassifier()`

    Returns:
        Trained model object (model and best parameters)
    """
    
    rf = RandomForestClassifier()
    # Perform random search cross-validation
    grid_search = RandomizedSearchCV(rf, **param_grid, n_jobs=-1, refit=True)
    grid_search.fit(x_train, y_train)
    model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    logger.info("Trained model object created:\n%s", str(model))

    return model, best_params


def save_model(model_artifacts: typing.Any, out_file: Path):
    """_summary_

    Args:
        tmo: _description_
        out_file: _description_.
    """
    with out_file.open("wb") as file:
        try:
            pickle.dump(model_artifacts, file)
        except pickle.PickleError:
            logger.error("Error while writing trained model object to %s", out_file)
        else:
            logger.info("Trained model object saved to %s", out_file)

def save_param(best_params: dict, out_file: Path):
    """_summary_

    Args:
        tmo: _description_
        out_file: _description_.
    """
    with out_file.open("w") as file:
        try:
            json.dump(best_params, file)
        except pickle.PickleError:
            logger.error("Error while writing best parameters to %s", out_file)
        else:
            logger.info("Trained best parameters saved to %s", out_file)


def train_model(
    data: pd.DataFrame,
    config: dict,
) -> tuple[sklearn.base.ClassifierMixin, pd.DataFrame, pd.DataFrame]:
    """Orchestrate model training by creating train test split, training model, and returning artifacts

    Args:
        data: Full data set for model development; this will be split into train/test
        config: Configuration for the train_model step; see example config file for structure

    Returns:
        Tuple of artifacts from train_model step:
            Trained Model Object from sklearn
            Train data with features and target
            Test data with features and target
    """

    x_train, x_test, y_train, y_test = train_test_split(
        data, **config["train_test_split"]
    )

    meta = pd.DataFrame()
    meta["min"] = data.drop("fraud",axis=1).min()
    meta["max"] = data.drop("fraud",axis=1).max()
    meta["mean"] = x_train.mean()
    meta["std"] = x_train.std()

    x_train, x_test = standardize_data(x_train, x_test, **config["standardize_data"])

    model, best_params = fit_model(x_train, y_train, config["fit_model"])

    target_col = config["train_test_split"]["target_col"]
    x_train[target_col] = y_train
    x_test[target_col] = y_test

    return model, best_params, x_train, x_test, meta
