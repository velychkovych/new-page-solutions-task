"""Static checks on web/index.html: self-contained, no framework, required controls,
the contract endpoints, and no HTML sinks for untrusted text."""

import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

from helpdesk.shared.ids import PRINCIPAL_ID_PATTERN

WEB = Path(__file__).resolve().parents[3] / "web"
INDEX = WEB / "index.html"


class _Collector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: list[tuple[str, dict[str, str]]] = []
        self.scripts: list[str] = []
        self.styles: list[str] = []
        self._in: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append((tag, {k: v or "" for k, v in attrs}))
        if tag in {"script", "style"}:
            self._in = tag

    def handle_endtag(self, tag: str) -> None:
        if tag == self._in:
            self._in = None

    def handle_data(self, data: str) -> None:
        if self._in == "script":
            self.scripts.append(data)
        elif self._in == "style":
            self.styles.append(data)


@pytest.fixture(scope="module")
def html() -> str:
    return INDEX.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def page(html: str) -> _Collector:
    collector = _Collector()
    collector.feed(html)
    collector.close()
    return collector


@pytest.fixture(scope="module")
def script(page: _Collector) -> str:
    return "\n".join(page.scripts)


def by_id(page: _Collector, element_id: str) -> tuple[str, dict[str, str]]:
    matches = [(tag, attrs) for tag, attrs in page.tags if attrs.get("id") == element_id]
    assert len(matches) == 1, f"expected exactly one #{element_id}"
    return matches[0]


def test_web_dir_has_no_build_artefacts() -> None:
    files = {p.relative_to(WEB).as_posix() for p in WEB.rglob("*") if p.is_file()}
    assert "index.html" in files
    others = files - {"index.html"}
    assert all(re.fullmatch(r"fixtures/[a-z0-9-]+\.(sse|json)", f) for f in others), others


def test_loads_no_external_resources(page: _Collector, html: str) -> None:
    for tag, attrs in page.tags:
        if tag == "script":
            assert "src" not in attrs, "scripts must be inline"
        for key in ("src", "href", "srcset", "data", "action", "poster"):
            value = attrs.get(key, "").strip().lower()
            assert not value.startswith(("http:", "https:", "//")), (tag, key, value)
        assert tag not in {"iframe", "object", "embed", "base"}
    css = "\n".join(page.styles).lower()
    assert "@import" not in css
    assert not re.search(r"url\(\s*['\"]?(https?:)?//", css)
    # No absolute URL anywhere: everything the page fetches is same-origin and relative.
    assert not re.search(r"https?://", html)


def test_has_a_strict_content_security_policy(page: _Collector) -> None:
    [policy] = [
        attrs["content"]
        for tag, attrs in page.tags
        if tag == "meta" and attrs.get("http-equiv", "").lower() == "content-security-policy"
    ]
    assert "default-src 'none'" in policy
    assert "connect-src 'self'" in policy


def test_uses_no_framework_or_module_imports(script: str, page: _Collector) -> None:
    assert not any(attrs.get("type") == "module" for tag, attrs in page.tags if tag == "script")
    assert not re.search(r"^\s*import\s|\bimport\s*\(|\brequire\s*\(", script, re.MULTILINE)
    for name in ("react", "vue", "angular", "jquery", "svelte", "htmx", "alpine", "preact"):
        assert not re.search(rf"\b{name}\b", script, re.IGNORECASE), name


@pytest.mark.parametrize(
    ("element_id", "tag"),
    [
        ("user-id", "input"),
        ("session-id", "input"),
        ("new-session", "button"),
        ("message", "textarea"),
        ("send", "button"),
        ("approve", "button"),
        ("decline", "button"),
        ("timeline", "ol"),
        ("status-line", "p"),
        ("trace-id", "span"),
        ("pending", "div"),
    ],
)
def test_has_required_controls(page: _Collector, element_id: str, tag: str) -> None:
    assert by_id(page, element_id)[0] == tag


def test_decision_buttons_start_disabled(page: _Collector) -> None:
    for element_id in ("approve", "decline"):
        assert "disabled" in by_id(page, element_id)[1]


def test_status_line_is_announced(page: _Collector) -> None:
    assert by_id(page, "status-line")[1].get("role") == "status"


def test_id_inputs_validate_with_the_shared_pattern(page: _Collector, script: str) -> None:
    # The JS check is the authoritative one and must be the server's exact pattern.
    assert f"/{PRINCIPAL_ID_PATTERN}/" in script
    for element_id in ("user-id", "session-id"):
        attrs = by_id(page, element_id)[1]
        assert attrs.get("maxlength") == "64"
        # HTML patterns compile with the `v` flag, where a literal "-" in a class must be
        # escaped; an invalid pattern would be silently ignored by the browser.
        assert attrs.get("pattern") == r"[a-zA-Z0-9_\-]{1,64}"


def test_message_length_matches_the_contract(page: _Collector) -> None:
    assert by_id(page, "message")[1].get("maxlength") == "4000"


def test_uses_the_contract_endpoints(script: str) -> None:
    assert '"/chat/stream"' in script
    assert '"/chat"' in script
    assert "`/sessions/${" in script
    assert "encodeURIComponent" in script


def test_reads_the_stream_with_fetch_not_event_source(script: str) -> None:
    assert "EventSource" not in script
    assert ".getReader()" in script
    assert "new TextDecoder()" in script
    assert "stream: true" in script
    assert "text/event-stream" in script


def test_sends_structured_decisions(script: str) -> None:
    assert "decision: { action_id: action.action_id, approve }" in script


@pytest.mark.parametrize(
    "sink",
    [
        r"\.innerHTML\b",
        r"\.outerHTML\b",
        r"insertAdjacentHTML",
        r"document\.write",
        r"\beval\s*\(",
        r"new\s+Function\s*\(",
        r"createContextualFragment",
        r"DOMParser",
        r"srcdoc",
    ],
)
def test_never_uses_html_sinks(html: str, sink: str) -> None:
    assert not re.search(sink, html), sink


def test_has_no_inline_event_handler_attributes(page: _Collector) -> None:
    for tag, attrs in page.tags:
        assert not [k for k in attrs if k.startswith("on")], tag


def test_jaeger_url_comes_only_from_the_meta_tag(page: _Collector, script: str) -> None:
    metas = [
        attrs
        for tag, attrs in page.tags
        if tag == "meta" and attrs.get("name") == "helpdesk-jaeger-ui-url"
    ]
    assert metas == [{"name": "helpdesk-jaeger-ui-url", "content": ""}]
    # No query-string override (test_page_behaviour checks the link itself).
    assert 'params.get("jaeger")' not in script
    assert "`trace/${traceId}`" in script
    # Only http(s) bases become links.
    assert 'url.protocol !== "http:" && url.protocol !== "https:"' in script


def test_fixture_mode_replays_files_from_the_fixtures_dir(script: str) -> None:
    assert 'params.get("fixture")' in script
    assert "fixtures/${file}" in script
    assert "createSseParser(" in script
    assert '"approve" : "decline"' in script
