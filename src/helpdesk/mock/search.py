"""Search semantics (CONTRACTS §1.4): AND of whitespace-separated, case-insensitive terms."""

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict

from helpdesk.shared.tickets import TicketStatus


class SearchFilter(BaseModel):
    """The normalised `(q, status)` pair. Cursors are bound to exactly this value."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    terms: tuple[str, ...]
    statuses: tuple[TicketStatus, ...]

    @classmethod
    def build(cls, q: str | None, statuses: Iterable[TicketStatus]) -> SearchFilter:
        # casefold, not lower: it also folds e.g. German "ß" so "STRASSE" finds "straße".
        terms = tuple(term.casefold() for term in (q or "").split())
        # Order and duplicates of `status` don't change the result, so they don't change
        # the filter (and a cursor stays valid if the client reorders the params).
        return cls(terms=terms, statuses=tuple(sorted(set(statuses))))

    def matches(self, title: str, description: str) -> bool:
        haystack = f"{title} {description}".casefold()
        return all(term in haystack for term in self.terms)
