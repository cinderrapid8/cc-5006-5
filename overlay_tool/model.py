"""Data model: sensor data versions, anchors, alignment.

Every sample carries its own logger timestamp. Array row numbers are
never used as video frame numbers; the only bridge between data and
video is the anchored time mapping in timeline.py.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class Field:
    name: str
    label: str
    unit: str
    kind: str  # "continuous" | "event"
    color: str = "#ffb000"


@dataclass
class SensorData:
    """One immutable version of the recorded sensor data."""
    version: str
    fields: dict[str, Field]
    # continuous fields: name -> (times, values); event fields: name -> times
    series: dict[str, tuple[list[float], list[float | None]]] = field(default_factory=dict)
    events: dict[str, list[float]] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def canonical(self) -> str:
        obj = {
            "version": self.version,
            "fields": {k: asdict(v) for k, v in sorted(self.fields.items())},
            "series": {k: v for k, v in sorted(self.series.items())},
            "events": {k: v for k, v in sorted(self.events.items())},
            "meta": self.meta,
        }
        return json.dumps(obj, sort_keys=True, separators=(",", ":"))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical().encode()).hexdigest()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.canonical(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SensorData":
        obj = json.loads(path.read_text(encoding="utf-8"))
        fields = {k: Field(**v) for k, v in obj["fields"].items()}
        series = {k: (list(v[0]), list(v[1])) for k, v in obj.get("series", {}).items()}
        events = {k: list(v) for k, v in obj.get("events", {}).items()}
        return cls(version=obj["version"], fields=fields, series=series,
                   events=events, meta=obj.get("meta", {}))

    def time_range(self) -> tuple[float, float] | None:
        lo, hi = None, None
        for times, _ in self.series.values():
            if times:
                lo = min(lo, min(times)) if lo is not None else min(times)
                hi = max(hi, max(times)) if hi is not None else max(times)
        for times in self.events.values():
            if times:
                lo = min(lo, min(times)) if lo is not None else min(times)
                hi = max(hi, max(times)) if hi is not None else max(times)
        return (lo, hi) if lo is not None else None


@dataclass
class Anchor:
    """A human-placed tie point: something seen in the video paired with
    the moment the data logger recorded it. Times are in each clock's own
    seconds."""
    label: str
    video_time: float
    data_time: float
    note: str = ""


@dataclass
class Alignment:
    """data_time = rate * video_time + offset.

    offset_lo/offset_hi bracket every anchor residual. This is an
    empirical range over the placed anchors, not a precision guarantee.
    """
    offset: float
    rate: float
    offset_lo: float
    offset_hi: float
    residuals: list[float]

    def to_dict(self) -> dict:
        return asdict(self)


def load_anchors(path: Path) -> list[Anchor]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    return [Anchor(**a) for a in obj["anchors"]]
