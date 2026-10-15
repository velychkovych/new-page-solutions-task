import json
from pathlib import Path
from typing import Any

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Status, StatusCode

from helpdesk.shared import trace_attributes as ta
from helpdesk.shared.settings import Settings
from helpdesk.telemetry import setup_telemetry
from helpdesk.telemetry.redaction import (
    REDACTED,
    RedactingSpanExporter,
    Redactor,
    is_secret_header_key,
    truncate_utf8,
)


def _jsonl_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize(
    "key",
    [
        "http.request.header.authorization",
        "http.request.header.Authorization",
        "http.request.header.x_admin_token",
        "http.request.header.x-admin-token",
        "HTTP.REQUEST.HEADER.X-ADMIN-TOKEN",
        "http.response.header.authorization",
        "http.request.header.proxy_authorization",
        "http.request.header.x_api_key",
        "Authorization",
        "X-Admin-Token",
    ],
)
def test_secret_header_keys_are_recognised(key: str) -> None:
    assert is_secret_header_key(key)


@pytest.mark.parametrize(
    "key", ["http.request.header.content_type", ta.TOOL_ARGS, "authorization_url.count"]
)
def test_other_keys_are_kept(key: str) -> None:
    assert not is_secret_header_key(key)


def test_truncate_utf8_never_splits_a_character() -> None:
    # 1 + 2*2000 bytes: the 2048-byte cut lands in the middle of a two-byte "é".
    text = "a" + "é" * 2000
    cut = truncate_utf8(text, 2048)
    assert len(cut.encode("utf-8")) == 2047
    assert text.startswith(cut)

    emoji = "🙂" * 1000  # four bytes each; 2048 is a multiple of 4, 2047 is not
    assert truncate_utf8(emoji, 2047) == "🙂" * 511
    assert truncate_utf8("short", 2048) == "short"


def test_tool_result_is_truncated_to_the_configured_bytes(make_settings: Any) -> None:
    settings: Settings = make_settings()
    assert settings.trace_result_max_bytes == 2048
    redactor = Redactor.from_settings(settings)

    attrs = redactor.attributes({ta.TOOL_RESULT: "ж" * 3000, ta.TOOL_ARGS: "x" * 3000})

    result = attrs[ta.TOOL_RESULT]
    assert isinstance(result, str)
    assert len(result.encode("utf-8")) == 2048
    assert attrs[ta.TOOL_ARGS] == "x" * 3000  # only the result is truncated


def test_secret_is_scrubbed_before_truncation(make_settings: Any, secrets: tuple[str, ...]) -> None:
    redactor = Redactor.from_settings(make_settings())
    token = secrets[0]

    # The 2048-byte cut falls inside the token. Truncating first would leak its prefix.
    attrs = redactor.attributes({ta.TOOL_RESULT: "x" * 2040 + token})

    result = attrs[ta.TOOL_RESULT]
    assert isinstance(result, str)
    assert token[:8] not in result
    assert result.startswith("x" * 2040 + REDACTED[:8])


def test_scrubs_strings_sequences_and_nested_values(
    make_settings: Any, secrets: tuple[str, ...]
) -> None:
    redactor = Redactor.from_settings(make_settings())
    api, admin, anthropic, openai = secrets

    attrs = redactor.attributes(
        {
            "plain": f"Bearer {api}",
            "list": (f"k={anthropic}", "fine"),
            "nested": {"headers": {"Authorization": f"Bearer {api}"}, "key": openai},
            "bytes": admin.encode(),
            "count": 3,
        }
    )

    dumped = repr(attrs)
    for secret in secrets:
        assert secret not in dumped
    assert attrs["plain"] == f"Bearer {REDACTED}"
    assert attrs["list"] == (f"k={REDACTED}", "fine")
    assert attrs["nested"] == {"headers": {}, "key": REDACTED}
    assert attrs["bytes"] == REDACTED.encode()
    assert attrs["count"] == 3


def test_redactor_ignores_unset_secrets() -> None:
    redactor = Redactor(["", "tok"], result_max_bytes=10)
    assert redactor.scrub("a tok b") == f"a {REDACTED} b"
    assert redactor.scrub("nothing") == "nothing"


def test_redacting_exporter_wraps_any_exporter(
    make_settings: Any, secrets: tuple[str, ...]
) -> None:
    """The same wrapper sits in front of OTLP, so this covers the OTLP path too."""
    inner = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(
        SimpleSpanProcessor(RedactingSpanExporter(inner, Redactor.from_settings(make_settings())))
    )
    tracer = provider.get_tracer("test")

    with tracer.start_as_current_span(f"call {secrets[2]}") as span:
        span.set_attribute("http.request.header.authorization", (f"Bearer {secrets[0]}",))
        span.add_event("send", {"http.request.header.x_admin_token": secrets[1], "k": secrets[3]})
        span.set_status(Status(StatusCode.ERROR, f"failed with {secrets[0]}"))

    (exported,) = inner.get_finished_spans()
    assert exported.name == f"call {REDACTED}"
    assert "http.request.header.authorization" not in (exported.attributes or {})
    assert dict(exported.events[0].attributes or {}) == {"k": REDACTED}
    assert exported.status.description == f"failed with {REDACTED}"
    assert exported.status.status_code is StatusCode.ERROR


def test_nothing_secret_reaches_the_jsonl_file(
    make_settings: Any, secrets: tuple[str, ...]
) -> None:
    settings: Settings = make_settings()
    telemetry = setup_telemetry("agent", settings)
    api, admin, anthropic, openai = secrets
    tracer = trace.get_tracer("test")

    with tracer.start_as_current_span(ta.SPAN_TOOL) as span:
        span.set_attributes(
            {
                "http.request.header.authorization": (f"Bearer {api}",),
                "http.request.header.x_admin_token": (admin,),
                "http.request.header.content_type": ("application/json",),
                ta.TOOL_ARGS: json.dumps({"query": f"vpn {anthropic}"}),
                ta.TOOL_RESULT: f"echo {openai} " + "r" * 5000,
            }
        )
        span.add_event(
            ta.EVENT_RETRY,
            {
                ta.RETRY_ATTEMPT: 1,
                "http.request.header.Authorization": f"Bearer {api}",
                "detail": f"admin={admin}",
            },
        )
        try:
            raise RuntimeError(f"upstream said {anthropic}")
        except RuntimeError as exc:
            span.record_exception(exc)
    assert telemetry.flush()

    path = settings.traces_dir / "agent.jsonl"
    raw = path.read_text(encoding="utf-8")
    for secret in secrets:
        assert secret not in raw
    assert "authorization" not in raw.lower()
    assert "x_admin_token" not in raw

    (record,) = _jsonl_lines(path)
    attrs = record["attributes"]
    assert attrs["http.request.header.content_type"] == ["application/json"]
    assert json.loads(attrs[ta.TOOL_ARGS]) == {"query": f"vpn {REDACTED}"}
    assert len(attrs[ta.TOOL_RESULT].encode("utf-8")) == 2048
    assert attrs[ta.TOOL_RESULT].startswith(f"echo {REDACTED} r")
    retry, exception = record["events"]
    assert retry["attributes"] == {ta.RETRY_ATTEMPT: 1, "detail": f"admin={REDACTED}"}
    assert exception["attributes"]["exception.message"] == f"upstream said {REDACTED}"
