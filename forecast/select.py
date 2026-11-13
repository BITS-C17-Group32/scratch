"""Interactive store, family, and forecast-window selection for app.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from forecast.cleaning import available_series
from forecast.windows import MAX_HORIZON


def match_family(user_input: str, families: list[str]) -> str | None:
    """Resolve a family by 1-based index, case-insensitive exact match, or
    unique prefix.
    """
    # A number is always a list index, so "12" picks the 12th family and never
    # a family whose name starts with a digit.
    query = user_input.strip()
    if query.isdigit():
        index = int(query) - 1
        return families[index] if 0 <= index < len(families) else None

    exact_matches = [f for f in families if f.lower() == query.lower()]
    if len(exact_matches) > 1:
        print(f"  ambiguous, matches: {', '.join(exact_matches)}")
        return None
    if exact_matches:
        return exact_matches[0]

    prefix_matches = [f for f in families if f.lower().startswith(query.lower())]
    if len(prefix_matches) == 1:
        return prefix_matches[0]
    if len(prefix_matches) > 1:
        print(f"  ambiguous, matches: {', '.join(prefix_matches)}")
    return None


def prompt(prompt_text: str, eof_hint: str = "") -> str:
    try:
        return input(prompt_text).strip()
    except EOFError:
        sys.exit(f"no interactive input available{eof_hint}")


def print_families(families: list[str]) -> None:
    for i, family in enumerate(families, 1):
        print(f"  {i:>2}. {family}")


def choose_store(stores: list[int]) -> int:
    """Prompt until the user enters a valid store number, or 'q' to quit."""
    print(f"\nAvailable Stores ({len(stores)}): {', '.join(map(str, stores))}")
    user_input = prompt(f"Store number ({stores[0]}-{stores[-1]}), or 'q' to quit: ")
    while True:
        if user_input.lower() == "q":
            sys.exit(0)
        if user_input.isdigit() and int(user_input) in stores:
            return int(user_input)
        print(f"  not a valid store; pick from {stores[0]}..{stores[-1]}")
        user_input = prompt("Store number: ")


def choose_family(families: list[str]) -> str:
    """Prompt until the user enters a resolvable family name, or 'q' to quit."""
    print(f"\nAvailable Families ({len(families)}):")
    print_families(families)
    user_input = prompt("Family (name, unique prefix, or number), or 'q' to quit: ")
    while True:
        if user_input.lower() == "q":
            sys.exit(0)
        matched = match_family(user_input, families)
        if matched is not None:
            return matched
        user_input = prompt("Family: ")


def choose_series(data_dir: Path) -> tuple[int, str]:
    """Prompt for and return a store number and product family."""
    stores, families = available_series(data_dir)
    store = choose_store(stores)
    family = choose_family(families)
    return store, family


def _prompt_optional(text: str) -> str:
    """Read one line, returning empty on end-of-input so piped runs can skip."""
    try:
        return input(text).strip()
    except EOFError:
        return ""


def _parse_date(text: str) -> pd.Timestamp:
    """Parse a strict YYYY-MM-DD date, rejecting times and time zones."""
    try:
        parsed = pd.to_datetime(text, format="%Y-%m-%d", errors="raise")
    except ValueError:
        raise ValueError(f"'{text}' is not a valid date (use YYYY-MM-DD)") from None
    if pd.isna(parsed):
        raise ValueError(f"'{text}' is not a valid date (use YYYY-MM-DD)")
    return pd.Timestamp(parsed)


def choose_future_dates(
    last_date: pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Prompt for an optional future forecast window, or None to skip.

    A non-empty start requires an end date on the next prompt. The start
    must sit after the last recorded date, and the rollout from the day
    after the last recorded date through the end date may not exceed
    MAX_HORIZON days, because the models still forecast the gap days.
    """
    start_text = _prompt_optional(
        f"\nFuture forecast start date (YYYY-MM-DD, after {last_date:%Y-%m-%d}), "
        "or Enter to skip: "
    )
    if not start_text:
        return None
    end_text = _prompt_optional("Future forecast end date (YYYY-MM-DD): ")
    if not end_text:
        raise ValueError("a start date needs an end date on the next prompt")
    start = _parse_date(start_text)
    end = _parse_date(end_text)
    if start <= last_date:
        raise ValueError(
            f"start date {start:%Y-%m-%d} must be after the last recorded day "
            f"{last_date:%Y-%m-%d}"
        )
    if end < start:
        raise ValueError(
            f"end date {end:%Y-%m-%d} is before the start date {start:%Y-%m-%d}"
        )
    span = (end - last_date).days
    if span > MAX_HORIZON:
        raise ValueError(
            f"the forecast ends {span} days after the last recorded day, "
            f"and the maximum is {MAX_HORIZON}"
        )
    return start, end
