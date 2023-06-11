import logging
from pathlib import Path
import typing
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def preprocess(card: pd.DataFrame, config: typing.Dict) -> pd.DataFrame:
    """
    Preprocess the raw data: drop outliers
    """
    # deal with outliers
    for i in config["outliers"]:
        Q1, Q3 = np.percentile(card[[i]], [25, 75])
        IQR = Q3 - Q1
        card = card[~((card[i] < (Q1 - 1.5 * IQR)) | (card[i] > (Q3 + 1.5 * IQR)))]
    return card

def save_dataset(data: pd.DataFrame, artifacts: Path):
    """
    Save the dataset
    """
    # save the dataset to artifacts
    data.to_csv(artifacts / "card_preprocess.csv", index=False)
    logger.info("Saved dataset to %s", artifacts / "card_preprocess.csv")  