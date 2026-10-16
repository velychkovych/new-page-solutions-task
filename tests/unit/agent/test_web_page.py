"""GET / and /fixtures/ (CONTRACTS §3.1): the page with its Jaeger meta, nothing else."""

import shutil
from html.parser import HTMLParser
from pathlib import Path

import pytest

from helpdesk.agent.web_page import JAEGER_META

WEB_DIR = Path(__file__).resolve().parents[3] / "web"


class _Metas(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.jaeger: list[str | None] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "meta" and values.get("name") == "helpdesk-jaeger-ui-url":
            self.jaeger.append(values.get("content"))


def _jaeger_metas(page: str) -> list[str | None]:
    parser = _Metas()
    parser.feed(page)
    return parser.jaeger


async def test_index_is_the_page_with_an_empty_jaeger_meta_when_unset(
    make_client, provider, fake_model
):
    client = await make_client(fake_model(responses=[]), provider, web_dir=WEB_DIR)

    response = await client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.text == (WEB_DIR / "index.html").read_text(encoding="utf-8")
    assert _jaeger_metas(response.text) == [""]


async def test_index_fills_the_jaeger_meta_from_settings(make_client, provider, fake_model):
    client = await make_client(
        fake_model(responses=[]),
        provider,
        web_dir=WEB_DIR,
        jaeger_ui_url="http://localhost:16686/",
    )

    page = (await client.get("/")).text

    assert _jaeger_metas(page) == ["http://localhost:16686/"]
    assert JAEGER_META not in page


async def test_jaeger_url_is_attribute_escaped(make_client, provider, fake_model):
    hostile = 'http://x/"><script>alert(1)</script><meta content="&'
    client = await make_client(
        fake_model(responses=[]), provider, web_dir=WEB_DIR, jaeger_ui_url=hostile
    )

    page = (await client.get("/")).text

    assert _jaeger_metas(page) == [hostile]
    assert "<script>alert(1)</script>" not in page
    assert "&quot;&gt;&lt;script&gt;" in page


@pytest.mark.parametrize("name", ["propose.sse", "approve.sse", "fallback.json"])
async def test_fixtures_are_served(make_client, provider, fake_model, name):
    client = await make_client(fake_model(responses=[]), provider, web_dir=WEB_DIR)

    response = await client.get(f"/fixtures/{name}")

    assert response.status_code == 200
    assert response.content == (WEB_DIR / "fixtures" / name).read_bytes()
    expected = "text/event-stream" if name.endswith(".sse") else "application/json"
    assert response.headers["content-type"].startswith(expected)


@pytest.mark.parametrize(
    "path",
    [
        "/fixtures/",
        "/fixtures",
        "/fixtures/missing.sse",
        "/fixtures/..%2Findex.html",
        "/fixtures/%2e%2e%2f%2e%2e%2fpyproject.toml",
        "/fixtures/..%5C..%5Cpyproject.toml",
        "/fixtures/Propose.sse",
        "/fixtures/propose.sse.bak",
        "/fixtures/.hidden.json",
        "/index.html",
        "/web/index.html",
        "/pyproject.toml",
    ],
)
async def test_nothing_else_is_served(make_client, provider, fake_model, path):
    client = await make_client(fake_model(responses=[]), provider, web_dir=WEB_DIR)

    response = await client.get(path)

    assert response.status_code in {404, 405}
    assert b"[project]" not in response.content
    assert b"<!doctype html>" not in response.content


async def test_a_symlink_out_of_fixtures_is_not_followed(
    make_client, provider, fake_model, tmp_path
):
    web = tmp_path / "web"
    (web / "fixtures").mkdir(parents=True)
    shutil.copy(WEB_DIR / "index.html", web / "index.html")
    (tmp_path / "secret.json").write_text('{"secret": true}')
    (web / "fixtures" / "leak.json").symlink_to(tmp_path / "secret.json")
    client = await make_client(fake_model(responses=[]), provider, web_dir=web)

    response = await client.get("/fixtures/leak.json")

    assert response.status_code == 404
    assert b"secret" not in response.content


async def test_a_missing_page_is_404(make_client, provider, fake_model, tmp_path):
    client = await make_client(fake_model(responses=[]), provider, web_dir=tmp_path / "nowhere")

    assert (await client.get("/")).status_code == 404
