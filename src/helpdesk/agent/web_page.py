"""The web page the agent serves (CONTRACTS §3.1 `GET /`) and its replay fixtures.

Only two things are served from `web_dir`: `index.html` and the files directly under
`fixtures/` whose names match the page's own fixture pattern. There is no directory
listing and no general static mount, so nothing else in the image or repo is reachable.
"""

import html
import re
from pathlib import Path
from typing import Final

INDEX_FILE: Final = "index.html"
FIXTURES_DIR: Final = "fixtures"
# web/index.html ships this tag empty; the agent fills it from Settings.jaeger_ui_url.
JAEGER_META: Final = '<meta name="helpdesk-jaeger-ui-url" content="">'
# The page's FIXTURE_RE plus the two extensions it fetches; no "/", "\\" or "..".
_FIXTURE_NAME: Final = re.compile(r"[a-z0-9][a-z0-9-]{0,63}\.(sse|json)")
FIXTURE_MEDIA_TYPES: Final[dict[str, str]] = {
    ".sse": "text/event-stream",
    ".json": "application/json",
}


class WebPageError(RuntimeError):
    """index.html is missing or no longer has the Jaeger meta tag the agent fills."""


def render_index(web_dir: Path, jaeger_ui_url: str | None) -> str:
    try:
        page = (web_dir / INDEX_FILE).read_text(encoding="utf-8")
    except OSError as error:
        raise WebPageError(f"cannot read {INDEX_FILE} from WEB_DIR") from error
    if page.count(JAEGER_META) != 1:
        raise WebPageError(f"{INDEX_FILE} must contain exactly one {JAEGER_META}")
    # quote=True: the value lands inside a double-quoted attribute.
    url = html.escape(jaeger_ui_url or "", quote=True)
    return page.replace(JAEGER_META, f'<meta name="helpdesk-jaeger-ui-url" content="{url}">')


def fixture_path(web_dir: Path, name: str) -> Path | None:
    """The fixture file for `name`, or None when the name or the file is not allowed."""
    if _FIXTURE_NAME.fullmatch(name) is None:
        return None
    directory = (web_dir / FIXTURES_DIR).resolve()
    path = (directory / name).resolve()
    # A symlink inside fixtures/ must not lead out of it.
    if path.parent != directory or not path.is_file():
        return None
    return path
