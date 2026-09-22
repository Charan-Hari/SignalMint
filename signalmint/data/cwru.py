"""CWRU bearing-vibration dataset adapter, with an offline synthetic fallback.

The Case Western Reserve University (CWRU) Bearing Data Center distributes
individually numbered MATLAB ``.mat`` files. This module can download a curated
subset (normal baseline + a few drive-end fault files), load the drive-end
vibration channel, and expose everything as plain :class:`SignalRecord` objects.

For CI and offline development a :func:`synthetic_bearing` generator produces
physically-motivated normal/fault vibration so the *entire* pipeline (train,
anomaly, compression, quantization, C-runtime parity) runs with zero downloads.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "SignalRecord",
    "CWRU_FILES",
    "CWRU_BASE_URL",
    "download_cwru_file",
    "load_mat_drive_end",
    "get_cwru_records",
    "synthetic_bearing",
    "get_synthetic_records",
]

CWRU_BASE_URL = "https://engineering.case.edu/sites/default/files/{number}.mat"

# Curated subset: file number -> (record name, label, fault_type).
# label: 0 = normal/healthy, 1 = fault. Kept small to bound download size.
CWRU_FILES: dict[int, tuple[str, int, str]] = {
    97: ("normal_0hp", 0, "normal"),
    98: ("normal_1hp", 0, "normal"),
    105: ("inner_race_007_0hp", 1, "inner_race"),
    118: ("ball_007_0hp", 1, "ball"),
    130: ("outer_race_007_0hp", 1, "outer_race"),
}


@dataclass
class SignalRecord:
    """A single 1-D vibration signal with its health label.

    Attributes:
        name: Human-readable identifier.
        signal: 1-D float32 vibration samples.
        label: 0 for normal/healthy, 1 for fault.
        fault_type: Descriptive fault category (``"normal"`` when healthy).
        source: ``"cwru"`` or ``"synthetic"``.
    """

    name: str
    signal: np.ndarray
    label: int
    fault_type: str
    source: str


def download_cwru_file(
    number: int, data_dir: str | Path, force: bool = False, retries: int = 3
) -> Path:
    """Download one numbered CWRU ``.mat`` file into ``data_dir/cwru``.

    Returns the local path. Skips the download if the file already exists. The
    download is retried a few times and written atomically (to a temp file that
    is renamed on success) so an interrupted transfer never leaves a corrupt
    ``.mat`` behind.
    """
    dest_dir = Path(data_dir) / "cwru"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{number}.mat"
    if dest.exists() and not force:
        return dest

    url = CWRU_BASE_URL.format(number=number)
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 (trusted host)
                data = resp.read()
            if len(data) < 1024:
                raise OSError(f"CWRU file {number} looks truncated ({len(data)} bytes)")
            tmp = dest.with_suffix(".mat.part")
            tmp.write_bytes(data)
            tmp.replace(dest)  # atomic on the same filesystem
            return dest
        except Exception as err:  # noqa: BLE001 - retry on any transient failure
            last_err = err
            if attempt == retries:
                break
    raise RuntimeError(
        f"failed to download CWRU file {number} from {url} after {retries} attempts: {last_err}"
    )


def load_mat_drive_end(path: str | Path) -> np.ndarray:
    """Load the drive-end (``*_DE_time``) vibration channel from a CWRU ``.mat``.

    Falls back to the first ``*_time`` channel if no drive-end channel exists.
    """
    from scipy.io import loadmat

    mat = loadmat(str(path))
    de_keys = [k for k in mat if k.endswith("_DE_time")]
    if not de_keys:
        de_keys = [k for k in mat if k.endswith("_time")]
    if not de_keys:
        raise ValueError(f"no vibration channel found in {path}")
    signal = np.asarray(mat[de_keys[0]], dtype=np.float32).reshape(-1)
    return signal


def get_cwru_records(
    data_dir: str | Path,
    numbers: list[int] | None = None,
    download: bool = True,
) -> list[SignalRecord]:
    """Load CWRU records, downloading the curated subset on demand.

    Args:
        data_dir: Root data directory (files land under ``data_dir/cwru``).
        numbers: File numbers to load; defaults to :data:`CWRU_FILES` keys.
        download: If True, fetch missing files; otherwise only load local ones.

    Returns:
        A list of :class:`SignalRecord`. Files that are missing (when
        ``download`` is False) or fail to load are skipped.
    """
    numbers = numbers if numbers is not None else list(CWRU_FILES)
    records: list[SignalRecord] = []
    for number in numbers:
        name, label, fault_type = CWRU_FILES.get(
            number, (f"file_{number}", 0, "unknown")
        )
        path = Path(data_dir) / "cwru" / f"{number}.mat"
        if not path.exists():
            if not download:
                continue
            path = download_cwru_file(number, data_dir)
        signal = load_mat_drive_end(path)
        records.append(
            SignalRecord(
                name=name,
                signal=signal,
                label=label,
                fault_type=fault_type,
                source="cwru",
            )
        )
    return records


def synthetic_bearing(
    n_samples: int,
    sample_rate: int = 12_000,
    fault: bool = False,
    rpm: float = 1797.0,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a physically-motivated synthetic bearing-vibration signal.

    Healthy signals are shaft harmonics plus broadband noise. Faulty signals add
    periodic impact bursts (the ballpass frequency signature of a defect),
    exponentially damped and amplitude-modulated -- the classic vibration cue a
    likelihood model learns to flag.

    Args:
        n_samples: Length of the signal.
        sample_rate: Sampling rate in Hz.
        fault: Whether to inject a bearing-fault impulse train.
        rpm: Shaft speed in revolutions per minute.
        seed: RNG seed for reproducibility.

    Returns:
        A 1-D float32 signal.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / sample_rate
    shaft_hz = rpm / 60.0

    # Shaft harmonics (healthy structure).
    signal = (
        1.0 * np.sin(2 * np.pi * shaft_hz * t)
        + 0.4 * np.sin(2 * np.pi * 2 * shaft_hz * t)
        + 0.2 * np.sin(2 * np.pi * 3 * shaft_hz * t)
    )
    signal += 0.15 * rng.standard_normal(n_samples)

    if fault:
        # Ballpass frequency (defect impacts), ~5.4x shaft for a typical bearing.
        bpf = 5.4152 * shaft_hz
        period = max(int(round(sample_rate / bpf)), 1)
        resonance_hz = 3000.0
        impulse = np.zeros(n_samples)
        for start in range(0, n_samples, period):
            length = min(period, n_samples - start)
            local_t = np.arange(length) / sample_rate
            decay = np.exp(-local_t * 800.0)
            ring = np.sin(2 * np.pi * resonance_hz * local_t)
            impulse[start : start + length] += decay * ring
        # Amplitude-modulate impacts by shaft rotation and add jitter.
        modulation = 0.8 + 0.2 * np.sin(2 * np.pi * shaft_hz * t)
        signal += 1.2 * impulse * modulation
        signal += 0.05 * rng.standard_normal(n_samples)

    return signal.astype(np.float32)


def get_synthetic_records(
    n_normal: int = 4,
    n_fault: int = 4,
    n_samples: int = 24_000,
    sample_rate: int = 12_000,
    seed: int = 0,
) -> list[SignalRecord]:
    """Build a balanced synthetic dataset mirroring the CWRU record interface."""
    rng = np.random.default_rng(seed)
    records: list[SignalRecord] = []
    for i in range(n_normal):
        rpm = float(rng.uniform(1730, 1797))
        sig = synthetic_bearing(
            n_samples, sample_rate, fault=False, rpm=rpm, seed=int(rng.integers(1e9))
        )
        records.append(SignalRecord(f"synthetic_normal_{i}", sig, 0, "normal", "synthetic"))
    fault_types = ["inner_race", "ball", "outer_race"]
    for i in range(n_fault):
        rpm = float(rng.uniform(1730, 1797))
        sig = synthetic_bearing(
            n_samples, sample_rate, fault=True, rpm=rpm, seed=int(rng.integers(1e9))
        )
        ft = fault_types[i % len(fault_types)]
        records.append(SignalRecord(f"synthetic_fault_{i}", sig, 1, ft, "synthetic"))
    return records
