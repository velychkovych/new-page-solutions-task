"""`python -m helpdesk.mock.seed [--reset]` (CONTRACTS §1.8).

Without `--reset` it only fills an empty database, so the mock can run it on every start
without touching existing data. `--reset` is what `make seed` and the eval runner use to
put the mock back into its known state between cases.
"""

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from helpdesk.mock.db import ALL_TABLES, init_schema, transaction
from helpdesk.mock.repository import insert_ticket
from helpdesk.mock.seed_data import SEED_TICKETS
from helpdesk.shared.settings import Settings


async def seed_database(db_path: Path, *, reset: bool = False) -> int:
    """Create the schema and insert the seed tickets if there are none. Returns how many
    tickets were inserted (0 when the database already had tickets)."""
    await init_schema(db_path)
    async with transaction(db_path, immediate=True) as conn:
        if reset:
            for table in ALL_TABLES:
                await conn.execute(f"DELETE FROM {table}")  # noqa: S608 - fixed table names
        async with conn.execute("SELECT COUNT(*) FROM tickets") as cursor:
            row = await cursor.fetchone()
        if row is not None and row[0] > 0:
            return 0
        for ticket in SEED_TICKETS:
            await insert_ticket(conn, ticket)
    return len(SEED_TICKETS)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m helpdesk.mock.seed")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="delete all tickets, comments, idempotency records and write counts first",
    )
    args = parser.parse_args(argv)
    settings = Settings()
    inserted = asyncio.run(seed_database(settings.mock_db_path, reset=args.reset))
    action = "reset and seeded" if args.reset else "seeded"
    if inserted:
        sys.stdout.write(f"{action} {inserted} tickets in {settings.mock_db_path}\n")
    else:
        sys.stdout.write(f"{settings.mock_db_path} already has tickets; nothing inserted\n")


if __name__ == "__main__":
    main()
