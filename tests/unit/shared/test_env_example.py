import re
from pathlib import Path

from helpdesk.shared.settings import Settings

ROOT = Path(__file__).resolve().parents[3]


def _keys(path: Path) -> list[str]:
    return re.findall(r"^([A-Z][A-Z0-9_]*)=", path.read_text(), flags=re.MULTILINE)


def test_env_example_lists_every_setting_exactly_once() -> None:
    keys = _keys(ROOT / ".env.example")
    assert len(keys) == len(set(keys))
    assert set(keys) == {name.upper() for name in Settings.model_fields}


def test_env_example_has_no_secret_values() -> None:
    text = (ROOT / ".env.example").read_text()
    for secret in ("MOCK_API_TOKEN", "MOCK_ADMIN_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        assert re.search(rf"^{secret}=$", text, flags=re.MULTILINE), secret
