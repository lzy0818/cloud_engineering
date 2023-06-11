from pathlib import Path
from typing import List

import boto3


def download_s3(bucket_name: str, object_key: str, local_file_path: Path):
    s3 = boto3.client("s3")
    print(f"Fetching Key: {object_key} from S3 Bucket: {bucket_name}")
    local_file_path = Path(local_file_path)
    local_file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        s3.download_file(bucket_name, object_key, str(local_file_path))
        print(f"File downloaded successfully to {local_file_path}")
    except Exception as e:
        print(f"Error downloading file: {local_file_path}")

def list_directories_s3(bucket_name, prefix):
    s3 = boto3.client('s3')
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter='/')

    directories = [prefix]

    if 'CommonPrefixes' in response:
        for obj in response['CommonPrefixes']:
            directory = obj['Prefix']
            directories.append(directory)

    return directories

def get_s3_directory_names(bucket_name: str, directory_prefix: str) -> List[str]:
    s3 = boto3.client("s3")
    print(f"Fetching directory names from S3 Bucket: {bucket_name}, Directory Prefix: {directory_prefix}")
    try:
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=directory_prefix, Delimiter="/")
        directories = [prefix["Prefix"] for prefix in response.get("CommonPrefixes", [])]
        print(f"Directory names fetched successfully: {directories}")
        return directories
    except Exception as e:
        print(f"Error fetching directory names: {e}")
        return []

