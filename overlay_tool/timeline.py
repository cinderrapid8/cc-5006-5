"""Time alignment between video clock and data-logger clock.

The mapping is data_time = rate * video_time + offset, fitted from
manual anchors. Clock error is reported as the range of anchor
residuals; we never claim a tighter sync than the anchors support.
"""
from __future__ import annotations

from .model import Anchor, Alignment


def fit_alignment(anchors: list[Anchor]) -> Alignment:
    if not anchors:
        raise ValueError("need at least one anchor")
    if len(anchors) >= 2:
        # least squares rate over anchor pairs
        n = len(anchors)
        sv = sum(a.video_time for a in anchors)
        sd = sum(a.data_time for a in anchors)
        svv = sum(a.video_time ** 2 for a in anchors)
        svd = sum(a.video_time * a.data_time for a in anchors)
        denom = n * svv - sv * sv
        rate = (n * svd - sv * sd) / denom if abs(denom) > 1e-12 else 1.0
    else:
        rate = 1.0
    offsets = [a.data_time - rate * a.video_time for a in anchors]
    offset = sum(offsets) / len(offsets)
    residuals = [o - offset for o in offsets]
    return Alignment(offset=offset, rate=rate,
                     offset_lo=min(offsets), offset_hi=max(offsets),
                     residuals=residuals)


def video_to_data_time(video_time: float, offset: float, rate: float) -> float:
    return rate * video_time + offset


def frame_video_time(frame_index: int, fps: float) -> float:
    """Frame index -> video clock seconds. This is the only index->time
    conversion allowed; sensor rows have no fixed cadence and must be
    looked up by their own timestamps."""
    return frame_index / fps


def detect_rollback(times: list[float]) -> list[tuple[int, float, float]]:
    """Return (index, previous_time, time) where the logger clock moved
    backwards."""
    out = []
    for i in range(1, len(times)):
        if times[i] < times[i - 1]:
            out.append((i, times[i - 1], times[i]))
    return out


def ambiguous_intervals(times: list[float]) -> list[tuple[float, float]]:
    """Time spans where non-monotonic logger timestamps make the physical
    moment of a sample unknowable from the file alone. Samples whose
    stamps fall inside such an interval must not be interpolated as if
    they were ordinary observations."""
    rollbacks = detect_rollback(times)
    if not rollbacks:
        return []
    intervals = []
    for idx, prev_t, t in rollbacks:
        lo = t
        hi = prev_t
        # the ambiguity lasts until stamps climb back above prev_t
        j = idx
        while j < len(times) and times[j] <= prev_t:
            j += 1
        if j < len(times):
            hi = max(hi, times[j])
        intervals.append((lo, hi))
    # merge overlaps
    intervals.sort()
    merged: list[list[float]] = []
    for lo, hi in intervals:
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [(lo, hi) for lo, hi in merged]


def in_any_interval(t: float, intervals: list[tuple[float, float]]) -> bool:
    return any(lo <= t <= hi for lo, hi in intervals)
