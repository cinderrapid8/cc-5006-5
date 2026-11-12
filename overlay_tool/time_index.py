"""Invalidatable time index for scrubbing long experiment videos.

Two costs used to be paid on every scrub event: the observation table was
parsed from disk again, and the bracketing samples were found by scanning
the series. This module parses each data version once into a TimeIndex and
then answers queries with a binary (or cursor) lookup.

The index separates *display* from *exact* data, on purpose:

- ``exact_reading`` mirrors SeriesSampler.at plus unit conversion one to
  one. It is the only source for numbers that end up overlaid on frames
  or written into manifests, so optimization never changes those values.
- ``field_curve`` returns a decimated polyline plus explicit
  missing/ambiguous spans for drawing. Decimation only removes interior
  observed points of valid segments; every gap/rollback boundary is kept,
  so non-monotonic and missing sections keep their existing meaning and
  the overlaid values are never derived from the sparse polyline.

An index belongs to one material version. The TimeIndexRegistry keys it
by data-file fingerprint, so rewriting a data version (same name, new
bytes) invalidates the old entry before it can be served, and background
work can check its material token before publishing anything.
"""
from __future__ import annotations

import hashlib
import json
import threading
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from pathlib import Path

from .model import SensorData
from .resample import FieldSampling, SeriesSampler, convert_unit
from .timeline import ambiguous_intervals


@dataclass(frozen=True)
class MaterialToken:
    """Identity of one observed material state: file path plus content
    hash. mtime_ns is a cheap pre-check; the sha256 decides validity."""
    version: str
    path: str
    sha256: str
    mtime_ns: int


@dataclass
class Reading:
    value: float | None
    status: str  # "ok" | "missing" | "ambiguous"


@dataclass
class CurveSegment:
    """One drawable piece of a field curve.

    kind "ok" carries decimated observed points (already converted to the
    display unit); "missing"/"ambiguous" carry the time span to shade.
    """
    kind: str
    t0: float
    t1: float
    points: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class FieldCurve:
    name: str
    display_unit: str
    segments: list[CurveSegment]


def fingerprint_data_file(path) -> MaterialToken:
    p = path if isinstance(path, Path) else Path(path)
    raw = p.read_bytes()
    return MaterialToken(
        version=json.loads(raw.decode("utf-8"))["version"],
        path=str(p),
        sha256=hashlib.sha256(raw).hexdigest(),
        mtime_ns=p.stat().st_mtime_ns,
    )


class FieldIndex:
    """Sorted, timestamp-keyed index for one continuous field."""

    def __init__(self, name: str, raw_times: list[float],
                 raw_values: list[float | None], field_unit: str):
        pairs = sorted(zip(raw_times, raw_values), key=lambda p: p[0])
        self.name = name
        self.field_unit = field_unit
        self.times = [p[0] for p in pairs]
        self.values = [p[1] for p in pairs]
        # rollback needs the original, possibly non-monotonic order
        self.ambiguous = ambiguous_intervals(list(raw_times))
        self._sampler = SeriesSampler(raw_times, raw_values, self.ambiguous)
        self.cursor = 0

    def invalidate_cursor(self) -> None:
        self.cursor = 0

    def _ambiguous_at(self, t: float) -> bool:
        ivs = self.ambiguous
        i = bisect_left(ivs, (t, float("inf")))
        for j in (i - 1, i):
            if 0 <= j < len(ivs) and ivs[j][0] <= t <= ivs[j][1]:
                return True
        return False

    def exact(self, t: float, cfg: FieldSampling) -> Reading:
        """Same semantics (and code path) as SeriesSampler.at."""
        v, status = self._sampler.at(t, cfg)
        return Reading(v, status)

    def locate(self, t: float) -> int:
        """Index i of the last sample with times[i] <= t, clamped."""
        return max(0, min(bisect_right(self.times, t) - 1,
                          len(self.times) - 1))

    def locate_sequential(self, t: float) -> int:
        """Cursor-accelerated locate for monotonically non-decreasing
        queries (frame order). Random access falls back to bisect."""
        ts = self.times
        if not ts:
            return 0
        i = self.cursor
        if i and t < ts[i]:
            i = self.locate(t)  # query moved backwards: cursor unusable
        else:
            while i + 1 < len(ts) and ts[i + 1] <= t:
                i += 1
        self.cursor = i
        return i

    def field_curve(self, lo: float, hi: float, cfg: FieldSampling,
                    display_unit: str, max_points: int) -> FieldCurve:
        """Decimated display geometry for [lo, hi].

        Built only from observed samples: sample pairs whose interpolation
        would report missing/ambiguous are never connected, so the shaded
        spans reproduce the exact path's status. Values are converted to
        the display unit; the exact reading path is untouched.
        """
        ts, vs = self.times, self.values
        if not ts:
            return FieldCurve(self.name, display_unit,
                              [CurveSegment("missing", lo, hi)])
        start = max(0, bisect_left(ts, lo) - 1)
        end = min(len(ts), bisect_right(ts, hi) + 1)
        sample_idx = list(range(start, end))

        def pair_status(i: int, j: int) -> tuple[str, float, float] | None:
            """Status the exact path assigns to the span between two
            adjacent observed samples; None means an interpolable pair.
            The returned span is the exact region to shade."""
            t0, t1 = ts[i], ts[j]
            v0, v1 = vs[i], vs[j]
            if self._ambiguous_at(t0) or self._ambiguous_at(t1):
                return "ambiguous", t0, t1
            if v0 is None or v1 is None:
                return "missing", t0, t1
            if t1 - t0 > cfg.max_gap:
                if cfg.method == "hold":
                    # the held value survives max_gap seconds, then missing
                    return "missing", t0 + cfg.max_gap, t1
                return "missing", t0, t1
            return None

        segments: list[CurveSegment] = []
        cur: list[int] = []

        def flush(indices: list[int]):
            if not indices:
                return
            # keep every boundary index; thin interior samples into
            # budget blocks, retaining each block's min and max so a
            # decimated polyline keeps the observed envelope instead of
            # aliasing oscillations
            keep = {indices[0], indices[-1]}
            interior = indices[1:-1]
            n_blocks = max(1, max_points - 2)
            block = max(1, len(interior) // n_blocks)
            for b in range(0, len(interior), block):
                chunk = [k for k in interior[b:b + block] if vs[k] is not None]
                if not chunk:
                    continue
                keep.add(min(chunk, key=lambda k: vs[k]))
                keep.add(max(chunk, key=lambda k: vs[k]))
            pts = [(ts[k], convert_unit(vs[k], self.field_unit, display_unit))
                   for k in indices if k in keep and vs[k] is not None]
            segments.append(CurveSegment(
                "ok", ts[indices[0]], ts[indices[-1]], pts))

        prev_i: int | None = None
        for k in sample_idx:
            if prev_i is None:
                cur = [k]
                prev_i = k
                continue
            bad = pair_status(prev_i, k)
            if bad is not None:
                kind, b0, b1 = bad
                flush(cur)
                cur = []
                joined = segments and segments[-1].kind == kind \
                    and segments[-1].t1 >= b0
                if joined:
                    segments[-1].t1 = max(segments[-1].t1, b1)
                else:
                    segments.append(CurveSegment(kind, b0, b1))
            else:
                cur.append(k)
            prev_i = k
        flush(cur)

        # spans before the first / after the last observed sample
        head = CurveSegment("missing", lo, min(hi, ts[0])) \
            if lo < ts[0] else None
        tail = CurveSegment("missing", max(lo, ts[-1]), hi) \
            if hi > ts[-1] else None
        out = []
        if head and head.t1 > head.t0:
            out.append(head)
        out.extend(segments)
        if tail and tail.t1 > tail.t0:
            out.append(tail)
        return FieldCurve(self.name, display_unit, out)


class TimeIndex:
    """Parsed, queryable snapshot of one SensorData version."""

    def __init__(self, data: SensorData, token: MaterialToken | None):
        self.data = data
        self.token = token
        self.fields: dict[str, FieldIndex] = {}
        for name, (times, values) in data.series.items():
            self.fields[name] = FieldIndex(
                name, list(times), list(values), data.fields[name].unit)
        self.events = {name: sorted(ts)
                       for name, ts in data.events.items()}

    def invalidate_cursors(self) -> None:
        for fi in self.fields.values():
            fi.invalidate_cursor()

    def exact_reading(self, name: str, t: float,
                      cfg: FieldSampling, display_unit: str) -> Reading:
        r = self.fields[name].exact(t, cfg)
        if r.value is None:
            return r
        return Reading(
            convert_unit(r.value, self.data.fields[name].unit, display_unit),
            r.status)

    def field_curve(self, name: str, lo: float, hi: float,
                    cfg: FieldSampling, display_unit: str,
                    max_points: int) -> FieldCurve:
        return self.fields[name].field_curve(
            lo, hi, cfg, display_unit, max_points)

    def events_in_window(self, name: str, lo: float, hi: float) -> list[float]:
        ts = self.events.get(name, [])
        a = bisect_left(ts, lo)
        b = bisect_right(ts, hi)
        return ts[a:b]

    def event_active(self, name: str, t: float, half_window: float) -> bool:
        ts = self.events.get(name, [])
        i = bisect_left(ts, t - half_window)
        return i < len(ts) and abs(ts[i] - t) <= half_window


class TimeIndexRegistry:
    """Process-level cache of TimeIndex objects, keyed by material
    fingerprint. Thread-safe; stale entries are replaced, never served."""

    def __init__(self):
        self._lock = threading.RLock()
        self._entries: dict[str, tuple[MaterialToken, TimeIndex]] = {}

    def _key(self, root, token: MaterialToken) -> str:
        return f"{Path(root).resolve()}::{token.version}::{token.sha256}"

    def get_or_build(self, root, data_path,
                     data: SensorData | None = None) -> TimeIndex:
        token = fingerprint_data_file(data_path)
        key = self._key(root, token)
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                return entry[1]
            if data is None or data.sha256() != token.sha256:
                data = SensorData.load(data_path)
            index = TimeIndex(data, token)
            self._entries[key] = (token, index)
            self._drop_other_versions(root, token)
            return index

    def _drop_other_versions(self, root, token: MaterialToken) -> None:
        prefix = f"{Path(root).resolve()}::{token.version}::"
        for k in [k for k in self._entries if k.startswith(prefix)
                  and k != self._key(root, token)]:
            del self._entries[k]

    def get(self, root, version: str) -> TimeIndex | None:
        """Return the cached index only if its file is unchanged."""
        path = Path(root) / "data" / f"{version}.json"
        if not path.exists():
            return None
        token = fingerprint_data_file(path)
        with self._lock:
            entry = self._entries.get(self._key(root, token))
            return entry[1] if entry is not None else None

    def invalidate(self, root, version: str | None = None) -> None:
        prefix = f"{Path(root).resolve()}::"
        if version is not None:
            prefix += f"{version}::"
        with self._lock:
            for k in [k for k in self._entries if k.startswith(prefix)]:
                del self._entries[k]

    def token_for(self, root, version: str) -> MaterialToken | None:
        path = Path(root) / "data" / f"{version}.json"
        if not path.exists():
            return None
        return fingerprint_data_file(path)
