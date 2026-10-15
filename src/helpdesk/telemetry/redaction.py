"""Redaction applied to every span before any exporter sees it (CONTRACTS §6).

Instrumentation defaults are not trusted: header capture can be switched on by env vars,
and exception messages or tool arguments can carry anything. So the exporter path drops
secret-bearing header attributes, scrubs known secret values everywhere, and truncates
tool results, for the JSONL and the OTLP exporter alike.
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Self

from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import Link, Status
from opentelemetry.util.types import AnyValue
from pydantic import SecretStr

from helpdesk.shared.settings import Settings
from helpdesk.shared.trace_attributes import TOOL_RESULT

REDACTED = "[REDACTED]"

# Last segment of an attribute key after normalising like OTel's header capture does
# (`http.request.header.x_admin_token`). x-api-key is how the Anthropic SDK sends its key.
_SECRET_HEADER_NAMES = frozenset(
    {"authorization", "proxy_authorization", "x_admin_token", "x_api_key"}
)


def is_secret_header_key(key: str) -> bool:
    normalised = key.strip().lower().replace("-", "_")
    return normalised.rsplit(".", 1)[-1] in _SECRET_HEADER_NAMES


def truncate_utf8(text: str, max_bytes: int) -> str:
    """Cut `text` to at most `max_bytes` UTF-8 bytes without splitting a character."""
    encoded = text.encode("utf-8", errors="surrogatepass")
    if len(encoded) <= max_bytes:
        return text
    # A cut inside a multi-byte sequence leaves an incomplete tail; "ignore" drops just that.
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


class Redactor:
    def __init__(self, secrets: Iterable[str], result_max_bytes: int) -> None:
        # Longest first, so a secret that contains another is replaced whole.
        self._secrets = tuple(sorted({s for s in secrets if s}, key=len, reverse=True))
        self._secret_bytes = tuple(s.encode("utf-8") for s in self._secrets)
        self._result_max_bytes = result_max_bytes

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        candidates: tuple[SecretStr | None, ...] = (
            settings.mock_api_token,
            settings.mock_admin_token,
            settings.anthropic_api_key,
            settings.openai_api_key,
        )
        return cls(
            (secret.get_secret_value() for secret in candidates if secret is not None),
            settings.trace_result_max_bytes,
        )

    def scrub(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, REDACTED)
        return text

    def _scrub_bytes(self, data: bytes) -> bytes:
        for secret in self._secret_bytes:
            data = data.replace(secret, REDACTED.encode("ascii"))
        return data

    def value(self, value: AnyValue) -> AnyValue:
        match value:
            case str():
                return self.scrub(value)
            case bytes():
                return self._scrub_bytes(value)
            case None | bool() | int() | float():
                return value
            case Mapping():
                return self.attributes(value)
            case Sequence():
                return tuple(self.value(item) for item in value)

    def attributes(self, attributes: Mapping[str, AnyValue] | None) -> dict[str, AnyValue]:
        redacted: dict[str, AnyValue] = {}
        for key, raw in (attributes or {}).items():
            if is_secret_header_key(key):
                continue
            value = self.value(raw)
            # Scrub before truncating: a cut through a secret would leak its prefix unmatched.
            if key == TOOL_RESULT and isinstance(value, str):
                value = truncate_utf8(value, self._result_max_bytes)
            redacted[key] = value
        return redacted

    def span(self, span: ReadableSpan) -> ReadableSpan:
        """A redacted copy; finished SDK spans are immutable, so they cannot be edited in place."""
        status = span.status
        return ReadableSpan(
            name=self.scrub(span.name),
            context=span.context,
            parent=span.parent,
            resource=span.resource,
            attributes=self.attributes(span.attributes),
            events=[
                Event(
                    name=self.scrub(event.name),
                    attributes=self.attributes(event.attributes),
                    timestamp=event.timestamp,
                )
                for event in span.events
            ],
            links=[Link(link.context, self.attributes(link.attributes)) for link in span.links],
            kind=span.kind,
            status=Status(
                status.status_code,
                None if status.description is None else self.scrub(status.description),
            ),
            start_time=span.start_time,
            end_time=span.end_time,
            instrumentation_scope=span.instrumentation_scope,
        )


class RedactingSpanExporter(SpanExporter):
    """Wraps a real exporter so nothing reaches it unredacted."""

    def __init__(self, inner: SpanExporter, redactor: Redactor) -> None:
        self.inner = inner
        self._redactor = redactor

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return self.inner.export([self._redactor.span(span) for span in spans])

    def shutdown(self) -> None:
        self.inner.shutdown()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return self.inner.force_flush(timeout_millis)
