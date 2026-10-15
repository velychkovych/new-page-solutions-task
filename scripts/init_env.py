"""Create .env from .env.example with freshly generated mock tokens.

Never overwrites an existing .env, and never prints a token.
"""

import re
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GENERATED = ("MOCK_API_TOKEN", "MOCK_ADMIN_TOKEN")


def render(template: str) -> str:
    for name in GENERATED:
        template = re.sub(
            rf"^{name}=$", f"{name}={secrets.token_urlsafe(32)}", template, flags=re.MULTILINE
        )
    return template


def main() -> int:
    target = ROOT / ".env"
    if target.exists():
        print(".env already exists; left unchanged.")
        return 0
    target.write_text(render((ROOT / ".env.example").read_text()))
    target.chmod(0o600)
    print(f"Created .env with generated {', '.join(GENERATED)}. Add ANTHROPIC_API_KEY to it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
