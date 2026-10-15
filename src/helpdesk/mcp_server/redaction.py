"""Keep secrets out of span attributes and tool results, and bound their size."""

from collections.abc import Iterable

REDACTED = "[REDACTED]"


class SecretRedactor:
    """Replaces every occurrence of a known secret value.

    Tool arguments and results should never contain the mock token in the first place;
    this is the backstop that makes that a guarantee rather than a hope.
    """

    def __init__(self, secrets: Iterable[str]) -> None:
        # Longest first, so a secret that contains another is replaced whole.
        self._secrets = sorted({s for s in secrets if s}, key=len, reverse=True)

    def __call__(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, REDACTED)
        return text


def truncate_utf8(text: str, max_bytes: int) -> str:
    """Cut to at most `max_bytes` of UTF-8 without splitting a character."""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")
