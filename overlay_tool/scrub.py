"""Scrub/preview service: answers "what is on screen at video time t"
without re-parsing the observation table on every drag event.

Two outputs are produced separately and must not be conflated:

- ``curves``: decimated display geometry and shaded missing/ambiguous
  spans. Thinning is purely visual; it never feeds overlaid numbers.
- ``readings``: exact per-field values from TimeIndex.exact_reading,
  i.e. the same computation the frame renderer writes into manifests.

The registry only serves an index whose fingerprint still matches the
file on disk, so callers always see the current material version.
"""
from __future__ import annotations

from .resample import FieldSampling
from .store import ProjectStore
from .time_index import TimeIndex, TimeIndexRegistry
from .timeline import fit_alignment, video_to_data_time
from .model import load_anchors

# max observed points kept per field polyline in a preview response
PREVIEW_MAX_POINTS = 120

_registry = TimeIndexRegistry()


def registry() -> TimeIndexRegistry:
    return _registry


def _field_cfg(fcfg: dict) -> FieldSampling:
    return FieldSampling(
        method=fcfg.get("method", "linear"),
        max_gap=fcfg.get("max_gap", 1.0),
        display_unit=fcfg.get("display_unit"),
        decimals=fcfg.get("decimals", 1),
    )


def open_index(store: ProjectStore, version: str) -> TimeIndex:
    """Cached, fingerprint-checked index for one data version."""
    path = store.root / "data" / f"{version}.json"
    return _registry.get_or_build(store.root, path)


def preview(store: ProjectStore, params: dict, video_time: float) -> dict:
    version = params["data_version"]
    index = open_index(store, version)
    offset = float(params.get("offset", 0.0))
    rate = float(params.get("rate", 1.0))
    window_s = float(params.get("window_s", 6.0))
    t_data = video_to_data_time(video_time, offset, rate)
    lo, hi = t_data - window_s, t_data

    curves, readings = [], []
    for fcfg in params.get("fields", []):
        if not fcfg.get("visible", True):
            continue
        name = fcfg["name"]
        field = index.data.fields[name]
        cfg = _field_cfg(fcfg)
        display_unit = fcfg.get("display_unit") or field.unit
        if field.kind == "continuous":
            r = index.exact_reading(name, t_data, cfg, display_unit)
            dec = cfg.decimals
            readings.append({
                "name": name, "label": field.label,
                "value": (round(r.value, dec)
                          if r.value is not None else None),
                "raw_value": r.value,
                "status": r.status,
                "unit": display_unit,
                "decimals": dec,
            })
            fc = index.field_curve(name, lo, hi, cfg, display_unit,
                                   PREVIEW_MAX_POINTS)
            curves.append({
                "name": name,
                "display_unit": fc.display_unit,
                "segments": [
                    {"kind": s.kind, "t0": s.t0, "t1": s.t1,
                     "points": s.points}
                    for s in fc.segments
                ],
            })
        else:
            active = index.event_active(name, t_data,
                                        half_window=0.5 / store.source_meta()["fps"])
            readings.append({
                "name": name, "label": field.label,
                "value": None, "status": "ok" if active else "inactive",
                "unit": field.unit, "active": bool(active),
            })
            curves.append({
                "name": name, "display_unit": field.unit,
                "events": index.events_in_window(name, lo, hi),
            })

    return {
        "material_token": {
            "version": index.token.version,
            "sha256": index.token.sha256,
        },
        "video_time_s": video_time,
        "data_time_s": t_data,
        "window_s": window_s,
        "readings": readings,
        "curves": curves,
    }


def alignment_snapshot(store: ProjectStore) -> dict:
    return fit_alignment(load_anchors(store.anchors_path())).to_dict()
