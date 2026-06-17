"""KenPom secret resolution — a thin shim over the platform resolver.

The provider chain (env / ``.env`` → AWS Secrets Manager / SSM), in-process TTL cache, and
never-log discipline all live in ``kitchen.secrets``. Where the secret comes from is declared
in ``params.yaml`` under ``secrets:`` (``KENPOM_API_KEY`` → SM bundle ``kenpom_key``), so there
is no hardcoded ARN or bespoke fetch logic here anymore.
"""

from kitchen.secrets import SecretNotFound, get

__all__ = ["SecretNotFound", "kenpom_api_key"]


def kenpom_api_key() -> str:
    """Return the KenPom API key, resolved through the kitchen secrets manifest."""
    return get("KENPOM_API_KEY")
