"""`python -m helpdesk.mock`: serve the mock on MOCK_HOST:MOCK_PORT."""

import uvicorn

from helpdesk.mock.app import create_app
from helpdesk.shared.settings import MissingSecretError, Settings


def main() -> None:
    settings = Settings()
    try:
        app = create_app(settings)
    except MissingSecretError as exc:
        # The message names the env var only, never a value.
        raise SystemExit(f"helpdesk.mock refuses to start: {exc}") from None
    uvicorn.run(
        app,
        host=settings.mock_host,
        port=settings.mock_port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
