import importlib.util
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[3]


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("init_env", ROOT / "scripts" / "init_env.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_render_fills_only_the_generated_tokens() -> None:
    init_env = _load()
    out = init_env.render("MOCK_API_TOKEN=\nMOCK_ADMIN_TOKEN=\nANTHROPIC_API_KEY=\n")
    values = dict(line.split("=", 1) for line in out.splitlines())
    assert len(values["MOCK_API_TOKEN"]) >= 32
    assert len(values["MOCK_ADMIN_TOKEN"]) >= 32
    assert values["MOCK_API_TOKEN"] != values["MOCK_ADMIN_TOKEN"]
    assert values["ANTHROPIC_API_KEY"] == ""
