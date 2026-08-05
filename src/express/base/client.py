import os
from typing import List


class Remote:
    """Manages S3 connection credentials and clients."""
    def __init__(self):
        import boto3
        from express.base.config import ensure_remote_config

        ensure_remote_config()
        for var in ["S3_AK", "S3_SK", "S3_ENDPOINT", "S3_BUCKET"]:
            if not os.getenv(var):
                raise EnvironmentError(f"Missing required environment variable: {var}")
        self.s3_ak = os.getenv("S3_AK")
        self.s3_sk = os.getenv("S3_SK")
        self.s3_endpoint = os.getenv("S3_ENDPOINT")
        self.s3_bucket = os.getenv("S3_BUCKET")
        self.s3 = boto3.resource(
            's3',
            aws_access_key_id=self.s3_ak,
            aws_secret_access_key=self.s3_sk,
            endpoint_url=self.s3_endpoint
        )
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=self.s3_ak,
            aws_secret_access_key=self.s3_sk,
            endpoint_url=self.s3_endpoint
        )


def is_remote_file_exists(s3_client, bucket: str, key: str) -> bool:
    """Checks if a file exists in the specified S3 bucket."""
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except s3_client.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        raise

def search_extension(remote: Remote, extension: str) -> List[str]:
    """Searches for objects in the S3 bucket that match the given file extension."""
    paginator = remote.s3_client.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=remote.s3_bucket)
    objects = page_iterator.search(f"Contents[?ends_with(Key, `{extension}`)][].Key")

    found_files = []
    for item in objects:
        if item:
            found_files.append(item)

    return found_files


def list_objects(remote: Remote) -> List[dict]:
    """Lists all objects in the bucket as dicts with Key and Size."""
    paginator = remote.s3_client.get_paginator('list_objects_v2')
    objects: List[dict] = []
    for page in paginator.paginate(Bucket=remote.s3_bucket):
        for obj in page.get('Contents') or []:
            objects.append({"Key": obj["Key"], "Size": obj["Size"]})
    return objects


def delete_objects(remote: Remote, keys: List[str]) -> None:
    """Deletes objects by key in batches of up to 1000."""
    if not keys:
        return
    for i in range(0, len(keys), 1000):
        batch = keys[i:i + 1000]
        remote.s3_client.delete_objects(
            Bucket=remote.s3_bucket,
            Delete={"Objects": [{"Key": key} for key in batch], "Quiet": True},
        )