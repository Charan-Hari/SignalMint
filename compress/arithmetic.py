"""A classic CACM-style integer arithmetic coder.

This is a general, model-agnostic range coder: at each step the caller supplies a
cumulative-frequency interval ``[cum_low, cum_high)`` out of ``total`` for the
symbol being coded. Driving it with a neural model's per-symbol probabilities
turns "predict the next sample well" into "spend fewer bits" -- the compression
head of SignalMint.

The implementation follows the well-known Witten/Neal/Cleary (1987) scheme with
underflow handling, operating on a 32-bit state. It is exact and reversible; the
round-trip is covered by tests.
"""

from __future__ import annotations

__all__ = [
    "BitWriter",
    "BitReader",
    "ArithmeticEncoder",
    "ArithmeticDecoder",
]


class BitWriter:
    """Accumulates individual bits MSB-first into a byte buffer."""

    def __init__(self) -> None:
        self._bytes = bytearray()
        self._cur = 0
        self._nbits = 0

    def write_bit(self, bit: int) -> None:
        self._cur = (self._cur << 1) | (bit & 1)
        self._nbits += 1
        if self._nbits == 8:
            self._bytes.append(self._cur)
            self._cur = 0
            self._nbits = 0

    def getvalue(self) -> bytes:
        """Flush any partial byte (zero-padded) and return the buffer."""
        if self._nbits == 0:
            return bytes(self._bytes)
        out = bytearray(self._bytes)
        out.append(self._cur << (8 - self._nbits))
        return bytes(out)

    @property
    def num_bits(self) -> int:
        return len(self._bytes) * 8 + self._nbits


class BitReader:
    """Reads bits MSB-first from a byte buffer, yielding 0 past the end."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    def read_bit(self) -> int:
        byte_index = self._pos >> 3
        if byte_index >= len(self._data):
            self._pos += 1
            return 0
        bit = (self._data[byte_index] >> (7 - (self._pos & 7))) & 1
        self._pos += 1
        return bit


class _Coder:
    """Shared 32-bit range constants."""

    PRECISION = 32
    FULL = 1 << PRECISION
    MASK = FULL - 1
    HALF = FULL >> 1
    QUARTER = FULL >> 2
    THREE_QUARTER = 3 * (FULL >> 2)
    # Cumulative-frequency totals must stay below this to avoid range collapse.
    MAX_TOTAL = QUARTER


class ArithmeticEncoder(_Coder):
    def __init__(self) -> None:
        self.low = 0
        self.high = self.MASK
        self.pending = 0
        self.out = BitWriter()

    def _emit(self, bit: int) -> None:
        self.out.write_bit(bit)
        inverse = 1 - bit
        for _ in range(self.pending):
            self.out.write_bit(inverse)
        self.pending = 0

    def encode(self, cum_low: int, cum_high: int, total: int) -> None:
        if not (0 <= cum_low < cum_high <= total):
            raise ValueError("invalid cumulative interval")
        if total > self.MAX_TOTAL:
            raise ValueError("total frequency too large for 32-bit coder")
        span = self.high - self.low + 1
        self.high = self.low + (span * cum_high) // total - 1
        self.low = self.low + (span * cum_low) // total
        while True:
            if self.high < self.HALF:
                self._emit(0)
            elif self.low >= self.HALF:
                self._emit(1)
                self.low -= self.HALF
                self.high -= self.HALF
            elif self.low >= self.QUARTER and self.high < self.THREE_QUARTER:
                self.pending += 1
                self.low -= self.QUARTER
                self.high -= self.QUARTER
            else:
                break
            self.low = (self.low << 1) & self.MASK
            self.high = ((self.high << 1) | 1) & self.MASK

    def finish(self) -> bytes:
        self.pending += 1
        self._emit(0 if self.low < self.QUARTER else 1)
        return self.out.getvalue()

    @property
    def num_bits(self) -> int:
        return self.out.num_bits


class ArithmeticDecoder(_Coder):
    def __init__(self, data: bytes) -> None:
        self.low = 0
        self.high = self.MASK
        self.reader = BitReader(data)
        self.code = 0
        for _ in range(self.PRECISION):
            self.code = (self.code << 1) | self.reader.read_bit()

    def decode_target(self, total: int) -> int:
        """Return a value in ``[0, total)`` locating the next symbol's interval."""
        if total > self.MAX_TOTAL:
            raise ValueError("total frequency too large for 32-bit coder")
        span = self.high - self.low + 1
        return ((self.code - self.low + 1) * total - 1) // span

    def update(self, cum_low: int, cum_high: int, total: int) -> None:
        span = self.high - self.low + 1
        self.high = self.low + (span * cum_high) // total - 1
        self.low = self.low + (span * cum_low) // total
        while True:
            if self.high < self.HALF:
                pass
            elif self.low >= self.HALF:
                self.code -= self.HALF
                self.low -= self.HALF
                self.high -= self.HALF
            elif self.low >= self.QUARTER and self.high < self.THREE_QUARTER:
                self.code -= self.QUARTER
                self.low -= self.QUARTER
                self.high -= self.QUARTER
            else:
                break
            self.low = (self.low << 1) & self.MASK
            self.high = ((self.high << 1) | 1) & self.MASK
            self.code = ((self.code << 1) | self.reader.read_bit()) & self.MASK
