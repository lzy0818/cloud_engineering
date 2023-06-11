import json
import os
import re
from pathlib import Path
import logging
from time import sleep

import typer
import botocore
import joblib
import yaml
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import src.acquire_data as ad
import src.aws_utils as aws
import src.preprocess as p
import src.evaluate_performance as ep
import src.train_model as tm
import src.eda as eda

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# artifacts = Path() / "artifacts"

# ARTIFACTS_PREFIX = os.getenv("ARTIFACTS_PREFIX", "experiments/")
# BUCKET_NAME = os.getenv("BUCKET_NAME", "group4-project")

# MODELS = {
#     "DecisionTreeClassifier": DecisionTreeClassifier,
#     "RandomForestClassifier": RandomForestClassifier,
# }


def load_config(config_ref: str) -> dict:
    if config_ref.startswith("s3://"):
        # Get config file from S3
        config_file = Path("config/downloaded-config.yaml")
        try:
            bucket, key = re.match(r"s3://([^/]+)/(.+)", config_ref).groups()
            aws.download_s3(bucket, key, config_file)
        except AttributeError:  # If re.match() does not return groups
            logger.error("Could not parse S3 URI: %s", config_ref)
            config_file = Path("config/default.yaml")
        except botocore.exceptions.ClientError as e:  # If there is an error downloading
            logger.error("Unable to download config file from S3: %s", config_ref)
            logger.error(e)
    else:
        # Load config from local path
        config_file = Path(config_ref)
    if not config_file.exists():
        raise EnvironmentError(f"Config file at {config_file.absolute()} does not exist")

    with config_file.open() as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


def run_pipeline(config):
    """Run the pipeline to produce a classifier model for the Iris dataset"""
    run_config = config.get("run_config", {})

    # Set up output directory for saving artifacts
    artifacts = Path(run_config.get("output", "runs")) / run_config.get("version", "default")
    artifacts.mkdir(parents=True)

    # Save config file to artifacts directory for traceability
    with (artifacts / "config.yaml").open("w") as f:
        yaml.dump(config, f)

    # Download data from online source as df
    data = ad.acquire_data_kaggle()

    # # EDA: features here replace to raw data: class distribution, heatmap, pairplot, boxplot for outlier
    # figures = artifacts / "figures"
    # figures.mkdir()
    # eda.plot_figure(data, figures, config['eda'])

    # Preprocess data: drop outliers, generate new features(optional)
    data_preprocess = p.preprocess(data, config["preprocess"])
    p.save_dataset(data_preprocess, artifacts)

    # Split data into train/test set, standard transform (standardscaler), and train model based on config; save each to disk
    tmo, best_params, train, test, inference_info = tm.train_model(data_preprocess, config["train_model"])
    tm.save_data(train, test, inference_info, artifacts)
    tm.save_model(tmo, artifacts / "trained_model_object.pkl")
    tm.save_param(best_params, artifacts / "best_params.json")

    # Evaluate model performance metrics: y_pred + metrics (accuracy, f1, classification report)
    y_pred = ep.predict(test, tmo, config["evaluate_performance"]["bin_predict_col"])
    metrics = ep.evaluate_performance(y_pred, config["evaluate_performance"])
    ep.save_metrics(metrics, artifacts / "metrics.yaml")

    # Upload all artifacts to S3
    aws_config = config.get("aws", {})
    if aws_config.get("upload", False):
        uploads = aws.upload_artifacts(artifacts, aws_config)
        logger.info("Uploaded: \n%s\n%s", "=" * 80, "\n".join(uploads))


def process_message(msg: aws.Message):
    message_body = json.loads(msg.body)
    bucket_name = message_body["detail"]["bucket"]["name"]
    object_key = message_body["detail"]["object"]["key"]
    config_uri = f"s3://{bucket_name}/{object_key}"
    logger.info("Running pipeline with config from: %s", config_uri)
    # get config from S3
    config = load_config(config_uri)
    run_pipeline(config)


def main(
    sqs_queue_url: str,
    max_empty_receives: int = 1,
    delay_seconds: int = 5,
    wait_time_seconds: int = 10,
):
    # Keep track of the number of times we ask queue for messages but receive none
    empty_receives = 0
    # After so many empty receives, we will stop processing and await the next trigger
    while empty_receives < max_empty_receives:
        # check for new messages
        logger.info("Polling queue for messages...")
        messages = aws.get_messages(
            sqs_queue_url,
            # only ask for 2 messages at a time
            max_messages=2,
            # long pulling: if no messages are available, wait for up to 10 seconds for one to become available
            wait_time_seconds=wait_time_seconds,
        )
        logger.info("Received %d messages from queue", len(messages))

        if len(messages) == 0:
            # Increment our empty receive count by one if no messages come back
            empty_receives += 1
            sleep(delay_seconds)
            continue

        # Reset empty receive count if we get messages back
        empty_receives = 0
        # Process each message
        for m in messages:
            # Perform work based on message content
            try:
                process_message(m)
            # We want to suppress all errors so that we can continue processing next message
            except Exception as e:
                logger.error("Unable to process message, continuing...")
                logger.error(e)
                continue
            # We must explicitly delete the message after processing it
            aws.delete_message(sqs_queue_url, m.handle)
        # Pause before asking the queue for more messages
        sleep(delay_seconds)


if __name__ == "__main__":
    typer.run(main)
