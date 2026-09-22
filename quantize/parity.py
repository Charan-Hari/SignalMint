"""Bit-exact parity harness between the Python integer reference and the C runtime.

Builds the C runtime (via MSVC or gcc/clang, whichever is available), runs it on a
symbol frame, and compares its int32 logits to :class:`IntegerRuntime` exactly.
Returns a structured result so both a script and a test can use it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..checkpoint import load_checkpoint
from ..model.wavenet import WaveNetLite
from .export import export_c_header
from .int_infer import IntegerRuntime
from .quantizer import quantize_model

__all__ = ["ParityResult", "find_c_compiler", "build_runtime", "check_parity"]

RUNTIME_DIR = Path(__file__).resolve().parents[2] / "runtime"


@dataclass
class ParityResult:
    exact: bool
    max_abs_diff: int
    num_mismatch: int
    num_values: int
    compiler: str


def _find_vcvars() -> str | None:
    roots = [
        Path("C:/Program Files (x86)/Microsoft Visual Studio"),
        Path("C:/Program Files/Microsoft Visual Studio"),
    ]
    for root in roots:
        if not root.exists():
            continue
        hits = list(root.glob("*/*/VC/Auxiliary/Build/vcvars64.bat"))
        if hits:
            return str(hits[0])
    return None


def find_c_compiler() -> tuple[str, str] | None:
    """Return ``(kind, path)`` for an available compiler, or None.

    ``kind`` is ``"gcc"``, ``"clang"`` or ``"msvc"``.
    """
    for kind in ("gcc", "clang", "cc"):
        path = shutil.which(kind)
        if path:
            return ("gcc" if kind != "clang" else "clang", path)
    vcvars = _find_vcvars()
    if vcvars:
        return ("msvc", vcvars)
    return None


def build_runtime(runtime_dir: Path = RUNTIME_DIR, exe_name: str = "sm_infer") -> tuple[str, Path]:
    """Compile the C runtime. Returns ``(compiler_kind, exe_path)``.

    Requires ``model_data.h`` to already exist in ``runtime_dir``.
    """
    comp = find_c_compiler()
    if comp is None:
        raise RuntimeError("no C compiler found (need gcc/clang or MSVC build tools)")
    kind, path = comp
    sources = ["sm_infer.c", "signalmint_rt.c"]

    if kind == "msvc":
        exe = runtime_dir / f"{exe_name}.exe"
        cmd = (
            f'call "{path}" >nul && cd /d "{runtime_dir}" && '
            f'cl /nologo /O2 /D_CRT_SECURE_NO_WARNINGS {" ".join(sources)} /Fe:{exe.name}'
        )
        # shell=True keeps the embedded quotes intact (a list would mangle them).
        subprocess.run(cmd, check=True, capture_output=True, text=True, shell=True)
    else:
        exe = runtime_dir / (f"{exe_name}.exe" if os.name == "nt" else exe_name)
        cmd = [path, "-O2", "-std=c11", *sources, "-o", exe.name]
        subprocess.run(cmd, cwd=runtime_dir, check=True, capture_output=True, text=True)
    return kind, exe


def check_parity(
    model: WaveNetLite, symbols: np.ndarray, runtime_dir: Path = RUNTIME_DIR
) -> ParityResult:
    """Export, build, run the C runtime and compare to the integer reference."""
    qm = quantize_model(model)
    export_c_header(qm, runtime_dir / "model_data.h")
    kind, exe = build_runtime(runtime_dir)

    rt = IntegerRuntime(qm)
    symbols = np.asarray(symbols, dtype=np.int64).reshape(-1)
    py_logits = rt.logits(symbols)
    T, B = py_logits.shape

    with tempfile.TemporaryDirectory() as d:
        sp = Path(d) / "syms.txt"
        np.savetxt(sp, symbols, fmt="%d")
        out = subprocess.run([str(exe), str(sp)], capture_output=True, text=True, check=True)
    c_logits = np.array(out.stdout.split(), dtype=np.int64).reshape(T, B)

    diff = np.abs(py_logits - c_logits)
    return ParityResult(
        exact=bool(np.array_equal(py_logits, c_logits)),
        max_abs_diff=int(diff.max()) if diff.size else 0,
        num_mismatch=int((diff > 0).sum()),
        num_values=int(py_logits.size),
        compiler=kind,
    )


def check_parity_from_checkpoint(
    checkpoint: str, symbols: np.ndarray, runtime_dir: Path = RUNTIME_DIR
) -> ParityResult:
    model, _cfg = load_checkpoint(checkpoint)
    return check_parity(model, symbols, runtime_dir)
