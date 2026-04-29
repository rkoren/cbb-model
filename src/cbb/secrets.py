"""Secret resolution: AWS Secrets Manager with local .env fallback.

Priority: environment variable → AWS Secrets Manager → error.
Set AWS_PROFILE or AWS_DEFAULT_REGION in environment as needed.
"""

import logging
import os

log = logging.getLogger(__name__)

_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
_RAW_CACHE: dict[str, str] = {}


def _sm_client():
    import boto3
    return boto3.client("secretsmanager", region_name=_REGION)


def get_raw_secret(secret_id: str) -> str:
    """Fetch a plain-string secret from AWS Secrets Manager by secret ID.

    Args:
        secret_id: The secret name or ARN (e.g. "kenpom_key").

    Returns:
        The raw SecretString value.

    Raises:
        RuntimeError: If the secret cannot be fetched.
    """
    if secret_id in _RAW_CACHE:
        return _RAW_CACHE[secret_id]

    try:
        response = _sm_client().get_secret_value(SecretId=secret_id)
        value = response["SecretString"]
        _RAW_CACHE[secret_id] = value
        return value
    except Exception as e:
        raise RuntimeError(
            f"Could not fetch secret '{secret_id}' from AWS Secrets Manager: {e}"
        ) from e


def kenpom_api_key() -> str:
    """Return the KenPom API key.

    Checks KENPOM_API_KEY env var first (local .env / CI), then fetches
    the 'kenpom_key' secret from AWS Secrets Manager and parses the
    JSON bundle {"KENPOM_API_KEY": "..."}.
    """
    if val := os.environ.get("KENPOM_API_KEY"):
        return val
    import json
    raw = get_raw_secret("kenpom_key")
    return json.loads(raw)["KENPOM_API_KEY"]
