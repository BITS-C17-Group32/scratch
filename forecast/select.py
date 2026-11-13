"""Interactive store/family selection for app.py."""

from __future__ import annotations

import sys
from pathlib import Path

from forecast.cleaning import available_series


def match_family(text: str, families: list[str]) -> str | None:
    """Exact name (case-insensitive), unique prefix, or 1-based list index."""
    text = text.strip()
    if text.isdigit():
        idx = int(text) - 1
        return families[idx] if 0 <= idx < len(families) else None
    hits = [f for f in families if f.lower() == text.lower()]
    if not hits:
        hits = [f for f in families if f.lower().startswith(text.lower())]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        print(f"  ambiguous, matches: {', '.join(hits)}")
    return None


def prompt(raw_text: str, eof_hint: str = "") -> str:
    try:
        return input(raw_text).strip()
    except EOFError:
        sys.exit(f"no interactive input available{eof_hint}")


def print_families(families: list[str]) -> None:
    for i, family in enumerate(families, 1):
        print(f"  {i:>2}. {family}")


def choose_store(stores: list[int]) -> int:
    """Prompt until the user enters a valid store number (or 'q')."""
    print(f"\nAvailable Stores ({len(stores)}): {', '.join(map(str, stores))}")
    raw = prompt(f"Store number ({stores[0]}-{stores[-1]}), or 'q' to quit: ")
    while not (raw.isdigit() and int(raw) in stores):
        if raw.lower() == "q":
            sys.exit(0)
        print(f"  not a valid store; pick from {stores[0]}..{stores[-1]}")
        raw = prompt("Store number: ")
    return int(raw)


def choose_family(families: list[str]) -> str:
    """Prompt until the user enters a resolvable family (or 'q')."""
    print(f"\nAvailable Families ({len(families)}):")
    print_families(families)
    raw = prompt("Family (name, unique prefix, or number), or 'q' to quit: ")
    while True:
        if raw.lower() == "q":
            sys.exit(0)
        matched = match_family(raw, families)
        if matched is not None:
            return matched
        raw = prompt("Family: ")


def choose_series(data_dir: Path) -> tuple[int, str]:
    """Interactive selection of one (store, family) pair."""
    stores, families = available_series(data_dir)
    store = choose_store(stores)
    family = choose_family(families)
    return store, family
