"""Time-base query benchmark: cost of locating samples and reading values.

Two implementations of the same time-base queries are compared:

- baseline: every query re-parses the observation table from JSON, sorts
  it, recomputes rollback intervals, and scans linearly for the bracket
  (the pre-optimization behavior on each scrub event);
- optimized: the table is parsed once into a TimeIndex; location is a
  bisect (or a monotonic cursor) and readings come from the same
  SeriesSampler semantics as the renderer.

Cost decomposition (all times in milliseconds):

  parse        json.load + building the sorted table / TimeIndex
  locate       finding the bracketing samples per query
  read         resampling (gap / rollback / interpolation) per query
  curve        display-only decimated polyline per window
  total        wall time of the full path

Workloads step up load in both directions requested for the data-sampling
study: denser observations and larger missing sections. Non-monotonic
rollback segments are retained at every level.

Value deviation is measured between the sparse display polyline and the
exact readings (the only numbers allowed into overlays/manifests), both
in display units and normalized to the field range. Status mismatches
are fatal: display sparsification must never change missing/ambiguous
interpretation. Run:  python -m overlay_tool.benchmark [--out results/bench.json]
"""
from __future__ import annotations

import argparse
import bisect
import json
import random
import statistics
import time
from pathlib import Path

from .model import Field, SensorData
from .resample import FieldSampling
from .time_index import TimeIndex
from .timeline import ambiguous_intervals

# (label, observation spacing s, gap count, gap length s, rollback count)
LEVELS = [
    ("L1_sparse", 1.00, 1, 2.0, 1),
    ("L2_nominal", 0.20, 2, 3.0, 1),
    ("L3_dense", 0.05, 3, 4.0, 2),
    ("L4_very_dense", 0.02, 4, 5.0, 2),
    ("L5_dense_long_gaps", 0.02, 6, 8.0, 3),
]
SPAN_S = 3600.0          # one hour of logged experiment time
WINDOW_S = 6.0
N_QUERIES = 2000
SEED = 20261112


def synthesize_workload(level: tuple, seed: int) -> SensorData:
    label, dt, n_gaps, gap_len, n_rollback = level
    rng = random.Random(seed)
    times: list[float] = []
    t = 0.0
    gap_centers = sorted(rng.uniform(100, SPAN_S - 100) for _ in range(n_gaps))
    gaps = [(c - gap_len / 2, c + gap_len / 2) for c in gap_centers]
    while t <= SPAN_S:
        if not any(a <= t <= b for a, b in gaps):
            times.append(round(t, 4))
        t += dt * rng.uniform(0.8, 1.2)
    values = [round(20 + 5 * (i % 97) / 97.0 + rng.uniform(-0.05, 0.05), 4)
              for i in range(len(times))]

    # clock rollbacks: splice short runs whose stamps move backwards,
    # in original order, keeping the same non-monotonic interpretation
    # as the demo data (the index sorts its own working copy)
    for k in range(n_rollback):
        anchor = rng.uniform(SPAN_S * 0.2, SPAN_S * 0.8)
        j = bisect.bisect_left(times, anchor)
        for m in range(1, 25):
            if j + m < len(times):
                times[j + m] = round(times[j] - m * dt, 4)

    events = sorted(rng.uniform(0, SPAN_S) for _ in range(60))
    fields = {
        "probe": Field("probe", "Probe", "degC", "continuous"),
        "stim": Field("stim", "Stim", "", "event"),
    }
    return SensorData(
        version=f"bench_{label}", fields=fields,
        series={"probe": (times, values)},
        events={"stim": events},
        meta={"level": label, "n_samples": len(times)},
    )


def _ms(seconds: float) -> float:
    return seconds * 1000.0


class NaiveView:
    """Re-parses and rescans on every query, as the old scrub path did."""

    def __init__(self, data: SensorData):
        self.raw = json.loads(data.canonical())

    def parse(self):
        obj = self.raw
        times, values = obj["series"]["probe"]
        pairs = sorted(zip(times, values), key=lambda p: p[0])
        ts = [p[0] for p in pairs]
        vs = [p[1] for p in pairs]
        amb = ambiguous_intervals(list(times))
        return ts, vs, amb

    def read_at(self, t: float, cfg: FieldSampling, ts, vs, amb):
        # linear scan for the bracket
        i = 0
        while i + 1 < len(ts) and ts[i + 1] <= t:
            i += 1
        if not ts or t < ts[0] or t > ts[-1]:
            return None, "missing"
        if any(a <= t <= b for a, b in amb):
            return None, "ambiguous"
        if i + 1 >= len(ts):
            return vs[i], "ok"
        t0, t1 = ts[i], ts[i + 1]
        if any(a <= t0 <= b or a <= t1 <= b for a, b in amb):
            return None, "ambiguous"
        if t1 - t0 > cfg.max_gap or vs[i] is None or vs[i + 1] is None:
            return None, "missing"
        w = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        return vs[i] + (vs[i + 1] - vs[i]) * w, "ok"


class IndexedView:
    """One parse, fingerprinted index; bisect/cursor location after."""

    def __init__(self, data: SensorData):
        self.index = TimeIndex(data, token=None)  # type: ignore[arg-type]
        self.fi = self.index.fields["probe"]

    def query_bisect(self, t: float, cfg: FieldSampling):
        self.fi.locate(t)
        r = self.fi.exact(t, cfg)
        return r.value, r.status

    def query_cursor(self, t: float, cfg: FieldSampling):
        self.fi.locate_sequential(t)
        r = self.fi.exact(t, cfg)
        return r.value, r.status


def _display_polyline_value(index: TimeIndex, t: float, cfg: FieldSampling,
                            max_points: int):
    """Value the drawn (decimated) polyline would give at t by linear
    interpolation between its retained points."""
    fc = index.field_curve("probe", t - WINDOW_S, t, cfg, "degC",
                           max_points=max_points)
    pts: list[tuple[float, float]] = []
    for seg in fc.segments:
        if seg.kind == "ok":
            pts.extend(seg.points)
    pts.sort()
    if not pts:
        return None
    i = bisect.bisect_right(pts, t, key=lambda p: p[0]) - 1
    if i < 0 or i + 1 >= len(pts):
        return None
    (t0, v0), (t1, v1) = pts[i], pts[i + 1]
    if t1 == t0:
        return v0
    return v0 + (v1 - v0) * (t - t0) / (t1 - t0)


def run_level(level: tuple, repeats: int) -> dict:
    data = synthesize_workload(level, SEED)
    cfg = FieldSampling(method="linear", max_gap=1.0)
    rng = random.Random(SEED + 1)
    queries = sorted(rng.uniform(0, SPAN_S) for _ in range(N_QUERIES))
    n_samples = len(data.series["probe"][0])

    # ---- baseline: parse+scan per query ----
    naive = NaiveView(data)
    base_parse, base_locate, base_read, base_total = [], [], [], []
    base_results: list[tuple] = []
    t_all = time.perf_counter()
    for q in queries:
        t0 = time.perf_counter()
        ts, vs, amb = naive.parse()
        t1 = time.perf_counter()
        i = 0
        while i + 1 < len(ts) and ts[i + 1] <= q:
            i += 1
        t2 = time.perf_counter()
        v, st = naive.read_at(q, cfg, ts, vs, amb)
        t3 = time.perf_counter()
        base_parse.append(_ms(t1 - t0))
        base_locate.append(_ms(t2 - t1))
        base_read.append(_ms(t3 - t2))
        base_total.append(_ms(t3 - t0))
        base_results.append((v, st))
    baseline_wall = _ms(time.perf_counter() - t_all)

    # ---- optimized: one build, indexed queries ----
    t0 = time.perf_counter()
    indexed = IndexedView(data)
    build_ms = _ms(time.perf_counter() - t0)

    opt_locate, opt_read, opt_total = [], [], []
    opt_results: list[tuple] = []
    t_all = time.perf_counter()
    for q in queries:
        t0 = time.perf_counter()
        indexed.fi.locate(q)
        t1 = time.perf_counter()
        r = indexed.fi.exact(q, cfg)
        t2 = time.perf_counter()
        opt_locate.append(_ms(t1 - t0))
        opt_read.append(_ms(t2 - t1))
        opt_total.append(_ms(t2 - t0))
        opt_results.append((r.value, r.status))
    optimized_wall = _ms(time.perf_counter() - t_all)

    # monotonic cursor path (frame-order scrubbing)
    indexed.fi.invalidate_cursor()
    cur_locate = []
    t_all = time.perf_counter()
    for q in queries:
        t0 = time.perf_counter()
        indexed.fi.locate_sequential(q)
        cur_locate.append(_ms(time.perf_counter() - t0))
    cursor_wall = _ms(time.perf_counter() - t_all)

    # display curve cost + deviation from exact values, at the production
    # polyline budget (120 pts) and an aggressive budget (24 pts)
    curve_ms, deviations, coarse_deviations, status_mismatch = [], [], [], 0
    rng2 = random.Random(SEED + 2)
    vrange = max(v for v in data.series["probe"][1] if v is not None) - \
        min(v for v in data.series["probe"][1] if v is not None)
    probe_queries = [rng2.uniform(WINDOW_S, SPAN_S) for _ in range(200)]
    for q in probe_queries:
        t0 = time.perf_counter()
        poly_v = _display_polyline_value(indexed.index, q, cfg, 120)
        curve_ms.append(_ms(time.perf_counter() - t0))
        coarse_v = _display_polyline_value(indexed.index, q, cfg, 24)
        exact = indexed.fi.exact(q, cfg)
        if exact.status != "ok":
            continue  # shaded spans, not polyline-interpolated
        if poly_v is None:
            status_mismatch += 1
            continue
        deviations.append(abs(poly_v - exact.value))
        if coarse_v is not None:
            coarse_deviations.append(abs(coarse_v - exact.value))

    # the optimized path must reproduce baseline statuses and values
    value_mismatch = 0
    for (bv, bs), (ov, os_) in zip(base_results, opt_results):
        if bs != os_:
            status_mismatch += 1
        elif bs == "ok" and abs(bv - ov) > 1e-9:
            value_mismatch += 1

    return {
        "level": level[0],
        "n_samples": n_samples,
        "n_queries": N_QUERIES,
        "baseline": {
            "parse_ms_total": round(sum(base_parse), 3),
            "locate_ms_total": round(sum(base_locate), 3),
            "read_ms_total": round(sum(base_read), 3),
            "wall_ms": round(baseline_wall, 3),
            "per_query_ms": {
                "parse": round(statistics.median(base_parse), 5),
                "locate": round(statistics.median(base_locate), 5),
                "read": round(statistics.median(base_read), 5),
                "total": round(statistics.median(base_total), 5),
            },
        },
        "optimized": {
            "index_build_ms": round(build_ms, 3),
            "locate_ms_total": round(sum(opt_locate), 3),
            "read_ms_total": round(sum(opt_read), 3),
            "wall_ms": round(optimized_wall, 3),
            "per_query_ms": {
                "locate": round(statistics.median(opt_locate), 5),
                "read": round(statistics.median(opt_read), 5),
                "total": round(statistics.median(opt_total), 5),
            },
            "cursor_locate_ms_total": round(sum(cur_locate), 3),
            "cursor_wall_ms": round(cursor_wall, 3),
            "cursor_per_query_ms": round(statistics.median(cur_locate), 5),
        },
        "display_curve": {
            "per_window_ms_median": round(statistics.median(curve_ms), 5),
            "max_points": 120,
            "max_deviation_display_unit": round(max(deviations, default=0.0), 6),
            "median_deviation_display_unit": round(
                statistics.median(deviations) if deviations else 0.0, 6),
            "max_normalized_deviation_pct": round(
                100 * max(deviations, default=0.0) / vrange, 5),
            "coarse_24pts_max_normalized_deviation_pct": round(
                100 * max(coarse_deviations, default=0.0) / vrange, 5),
            "status_mismatches": status_mismatch,
        },
        "correctness": {
            "status_mismatches": status_mismatch,
            "value_mismatches_vs_baseline": value_mismatch,
        },
        "speedup_wall_x": round(baseline_wall / optimized_wall, 2),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/time_base_benchmark.json")
    ap.add_argument("--repeats", type=int, default=1)
    args = ap.parse_args()

    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    levels = [run_level(lvl, args.repeats) for lvl in LEVELS]
    fatal = [r for r in levels
             if r["correctness"]["status_mismatches"]
             or r["correctness"]["value_mismatches_vs_baseline"]
             or r["display_curve"]["status_mismatches"]]
    report = {
        "started_at": started,
        "n_queries_per_level": N_QUERIES,
        "span_s": SPAN_S,
        "cost_order_note": (
            "baseline per query: parse -> linear locate -> read; "
            "optimized: index build once -> bisect/cursor locate -> read. "
            "Display curve building is a separate, additional path whose "
            "decimated values never replace exact readings."
        ),
        "levels": levels,
        "all_correct": not fatal,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print(f"{'level':<22}{'samples':>9}{'base ms':>11}{'opt ms':>10}"
          f"{'speedup':>9}{'max dev %':>11}{'status mm':>10}")
    for r in levels:
        d = r["display_curve"]
        print(f"{r['level']:<22}{r['n_samples']:>9}"
              f"{r['baseline']['wall_ms']:>11.1f}"
              f"{r['optimized']['wall_ms']:>10.1f}"
              f"{r['speedup_wall_x']:>8.1f}x"
              f"{d['max_normalized_deviation_pct']:>10.4f}%"
              f"{r['correctness']['status_mismatches']:>10}")
    print(f"\nreport: {out}")
    if fatal:
        print("FAIL: behavior diverged from baseline; roll back the "
              "offending sampling optimization.")
        raise SystemExit(1)
    print("OK: exact values and missing/ambiguous statuses match baseline "
          "at every level.")


if __name__ == "__main__":
    main()
