"""Per-field resampling of non-uniform, possibly gapped sensor data.

Each field configures its own method and maximum gap. Queries landing in
a gap larger than the limit, outside the recorded range, or inside a
clock-rollback interval report "missing"/"ambiguous" instead of
fabricating a straight line through unobserved time.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

# value, status; status in {"ok", "missing", "ambiguous"}
Reading = tuple[float | None, str]

_UNIT_FACTORS = {
    ("kPa", "hPa"): ("mul", 10.0),
    ("hPa", "kPa"): ("mul", 0.1),
    ("Pa", "kPa"): ("mul", 0.001),
    ("kPa", "Pa"): ("mul", 1000.0),
    ("mV", "V"): ("mul", 0.001),
    ("V", "mV"): ("mul", 1000.0),
    ("degC", "K"): ("add", 273.15),
    ("K", "degC"): ("add", -273.15),
}


def convert_unit(value: float, from_unit: str, to_unit: str) -> float:
    if from_unit == to_unit:
        return value
    key = (from_unit, to_unit)
    if key not in _UNIT_FACTORS:
        raise ValueError(f"no unit conversion {from_unit} -> {to_unit}")
    op, k = _UNIT_FACTORS[key]
    return value * k if op == "mul" else value + k


@dataclass
class FieldSampling:
    method: str = "linear"   # "linear" | "hold"
    max_gap: float = 1.0     # seconds of unobserved time tolerated
    display_unit: str | None = None  # defaults to the field's own unit
    decimals: int = 1


def _sorted_unique(times: list[float], values: list[float | None]):
    pairs = sorted(zip(times, values), key=lambda p: p[0])
    st = [p[0] for p in pairs]
    sv = [p[1] for p in pairs]
    return st, sv


class SeriesSampler:
    """Timestamp-keyed lookup over one field's samples."""

    def __init__(self, times: list[float], values: list[float | None],
                 ambiguous: list[tuple[float, float]] | None = None):
        self.times, self.values = _sorted_unique(times, values)
        self.ambiguous = ambiguous or []

    def _is_ambiguous(self, t: float) -> bool:
        return any(lo <= t <= hi for lo, hi in self.ambiguous)

    def at(self, t: float, cfg: FieldSampling) -> Reading:
        ts, vs = self.times, self.values
        if not ts:
            return None, "missing"
        if self._is_ambiguous(t):
            return None, "ambiguous"
        if t < ts[0] or t > ts[-1]:
            return None, "missing"
        i = bisect_right(ts, t) - 1
        i = max(0, min(i, len(ts) - 1))
        if cfg.method == "hold":
            if t - ts[i] > cfg.max_gap:
                return None, "missing"
            v = vs[i]
            return (v, "ok") if v is not None else (None, "missing")
        # linear
        if i + 1 >= len(ts):
            v = vs[i]
            return (v, "ok") if v is not None else (None, "missing")
        t0, t1 = ts[i], ts[i + 1]
        v0, v1 = vs[i], vs[i + 1]
        if t1 - t0 > cfg.max_gap:
            return None, "missing"
        if self._is_ambiguous(t0) or self._is_ambiguous(t1):
            return None, "ambiguous"
        if v0 is None or v1 is None:
            return None, "missing"
        if t1 == t0:
            return v0, "ok"
        w = (t - t0) / (t1 - t0)
        return v0 + (v1 - v0) * w, "ok"


def event_active(event_times: list[float], t: float, half_window: float) -> bool:
    """True when a logged event falls within half a frame of t."""
    return any(abs(te - t) <= half_window for te in event_times)


def events_in_window(event_times: list[float], lo: float, hi: float) -> list[float]:
    return [te for te in event_times if lo <= te <= hi]
