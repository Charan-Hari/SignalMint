"""Unified record loading: real CWRU with a graceful synthetic fallback."""

from __future__ import annotations

from .cwru import SignalRecord, get_cwru_records, get_synthetic_records

__all__ = ["load_records"]


def load_records(
    data_dir: str = "data",
    source: str = "auto",
    download: bool = True,
    numbers: list[int] | None = None,
) -> list[SignalRecord]:
    """Load labeled signal records.

    Args:
        data_dir: Root data directory.
        source: ``"cwru"``, ``"synthetic"``, or ``"auto"`` (try CWRU, fall back
            to synthetic on any failure -- keeps offline/CI runs unblocked).
        download: Whether to download missing CWRU files.
        numbers: Optional explicit CWRU file numbers.

    Returns:
        A list of :class:`SignalRecord`.
    """
    if source == "synthetic":
        return get_synthetic_records()
    if source == "cwru":
        return get_cwru_records(data_dir, numbers=numbers, download=download)
    # auto
    try:
        records = get_cwru_records(data_dir, numbers=numbers, download=download)
        if records:
            return records
    except Exception:  # noqa: BLE001 - fall back to synthetic on any I/O failure
        pass
    return get_synthetic_records()
