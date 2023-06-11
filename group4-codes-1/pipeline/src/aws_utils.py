from pathlib import Path
import logging
from dataclasses import dataclass

import boto3
import botocore.exceptions


logger = logging.getLogger(__name__)


def upload_artifacts(artifacts: Path, config: dict) -> list[str]:
    """Upload all the artifacts in the specified directory to S3


    Args:
        artifacts: Directory containing all the artifacts from a given experiment
        config: Config required to upload artifacts to S3; see example config file for structure

    Returns:
        List of S3 uri's for each file that was uploaded
    """
    bucket_name = config["bucket_name"]
    PREFIX = Path(config["prefix"])
    # Create an S3 client using the default credentials chain
    try:
        session = boto3.Session()
        s3_client = session.client("s3")
    except botocore.exceptions.ClientError as e:
        logger.error("could not create boto3 client; error: %s", e)

    s3_uris = []  # Keep track of uploaded URIs
    for file_name in artifacts.glob("**/*"):
        # Skip directories
        if file_name.is_dir():
            continue

        s3_key = PREFIX / file_name.relative_to(artifacts.parent)
        try:
            # Upload file to specified S3 bucket
            s3_client.upload_file(
                Filename=str(file_name), Bucket=bucket_name, Key=str(s3_key)
            )
        except botocore.exceptions.BotoCoreError as err:
            logger.warning("could not upload %s; skipping... Error: %s", file_name, err)
        else:
            # Record successful upload
            s3_uris.append(f"s3://{bucket_name}/{s3_key}")
            logger.debug("file %s uploaded to S3 %s", file_name, bucket_name)

    return s3_uris

def download_s3(bucket_name: str, object_key: str, local_file_path: Path):
    s3 = boto3.client("s3")
    print(f"Fetching Key: {object_key} from S3 Bucket: {bucket_name}")
    s3.download_file(bucket_name, object_key, str(local_file_path))
    logger.info("File downloaded successfully to %s", local_file_path)


def upload_files_to_s3(bucket_name: str, prefix: str, directory: Path) -> bool:
    # Check for AWS credentials
    try:
        session = boto3.Session()
        s3 = session.client("s3")
    except Exception as e:
        logger.error("Failed to create boto3 session: %s", e)
        return False

    # Check if bucket exists
    try:
        s3.head_bucket(Bucket=bucket_name)
    except Exception as e:
        logger.error(
            "The bucket '%s' does not exist or you do not have permission to access it", bucket_name
        )
        logger.error(e)
        return False

    # Iterate through files in directory and upload to S3
    for file_path in directory.glob("*"):
        if file_path.is_file():
            try:
                key = str(Path(prefix) / Path(file_path.name))  # Use prefix instead of file parent
                with file_path.open("rb") as data:
                    s3.upload_fileobj(data, bucket_name, key)
                logger.debug("File '%s' uploaded to S3 bucket '%s'", file_path.name, bucket_name)
            except Exception as e:
                logger.error("Failed to upload file '%s': %s", file_path.name, e)
                return False

    return True


@dataclass
class Message:
    handle: str
    body: str


def get_messages(
    queue_url: str,
    max_messages: int = 1,
    wait_time_seconds: int = 1,
) -> list[Message]:
    sqs = boto3.client("sqs")
    try:
        response = sqs.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=max_messages,
            WaitTimeSeconds=wait_time_seconds,
        )
    except botocore.exceptions.ClientError as e:
        logger.error(e)
        return []
    if "Messages" not in response:
        return []
    return [Message(m["ReceiptHandle"], m["Body"]) for m in response["Messages"]]


def delete_message(queue_url: str, receipt_handle: str):
    sqs = boto3.client("sqs")
    sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)
