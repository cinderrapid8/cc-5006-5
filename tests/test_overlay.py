"""Tests for the overlay tool. Uses a tiny synthetic project so renders
stay fast."""
import csv
import json
import shutil
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

from overlay_tool.model import SensorData, Field, Anchor
from overlay_tool.timeline import (fit_alignment, ambiguous_intervals,
                                   frame_video_time, video_to_data_time)
from overlay_tool.resample import (FieldSampling, SeriesSampler, convert_unit,
                                   event_active)
from overlay_tool.store import ProjectStore
from overlay_tool.render import render

FPS = 10
OFFSET = 0.5


@pytest.fixture(scope="module")
def tiny_project(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny")
    # 30 frames, flash at frame 10 (video t=1.0)
    writer = imageio.get_writer(str(root / "source.mp4"), fps=FPS,
                                codec="libx264", pixelformat="yuv420p",
                                macro_block_size=16)
    for i in range(30):
        frame = np.full((240, 320, 3), 30, dtype=np.uint8)
        if i == 10:
            frame[:] = 240
        writer.append_data(frame)
    writer.close()
    (root / "source_meta.json").write_text(json.dumps(
        {"fps": FPS, "width": 320, "height": 240, "frames": 30,
         "duration_s": 3.0}))
    fields = {
        "temperature": Field("temperature", "Temp", "degC", "continuous"),
        "pressure": Field("pressure", "Pressure", "kPa", "continuous"),
        "stim_pulse": Field("stim_pulse", "STIM", "", "event"),
    }
    # non-uniform temperature with a gap at data t in [1.0, 2.0]
    tt = [0.2, 0.55, 0.9, 2.1, 2.4, 2.55]
    tv = [20.0, 20.5, 21.0, 23.0, 23.5, 23.6]
    # pressure with a clock rollback: 0.1..0.8 then 0.3..0.5 then 0.9..1.4
    pt = [0.1, 0.4, 0.8, 0.3, 0.5, 0.9, 1.4]
    pv = [100, 101, 102, 101.5, 102.5, 103, 104]
    v1 = SensorData("v1", fields,
                    series={"temperature": (tt, tv), "pressure": (pt, pv)},
                    events={"stim_pulse": [1.0 + OFFSET]})
    v1.save(root / "data" / "v1.json")
    # v2: pressure in hPa
    fields2 = dict(fields)
    fields2["pressure"] = Field("pressure", "Pressure", "hPa", "continuous")
    v2 = SensorData("v2", fields2,
                    series={"temperature": (tt, tv),
                            "pressure": (pt, [v * 10 for v in pv])},
                    events={"stim_pulse": [1.0 + OFFSET]})
    v2.save(root / "data" / "v2.json")
    (root / "anchors.json").write_text(json.dumps({"anchors": [
        {"label": "flash<->pulse", "video_time": 1.0, "data_time": 1.5,
         "note": "t"}]}))
    return root


def base_params(**kw):
    p = {"data_version": "v1", "offset": OFFSET, "rate": 1.0,
         "window_s": 2.0, "layout": "bottom",
         "fields": [
             {"name": "temperature", "visible": True, "method": "linear",
              "max_gap": 0.5, "display_unit": None, "decimals": 1},
             {"name": "pressure", "visible": True, "method": "linear",
              "max_gap": 0.5, "display_unit": None, "decimals": 1},
             {"name": "stim_pulse", "visible": True},
         ]}
    p.update(kw)
    return p


# ---------- timeline ----------

def test_fit_alignment_two_anchors():
    a = [Anchor("a", 1.0, 2.5), Anchor("b", 3.0, 4.6)]
    al = fit_alignment(a)
    assert al.rate == pytest.approx(1.05)
    assert al.offset == pytest.approx(1.45)
    assert al.offset_lo <= al.offset <= al.offset_hi


def test_offset_range_brackets_anchor_residuals():
    a = [Anchor("a", 1.0, 2.4), Anchor("b", 2.0, 3.5), Anchor("c", 3.0, 4.45)]
    al = fit_alignment(a)
    for x in a:
        assert al.offset_lo - 1e-9 <= x.data_time - al.rate * x.video_time \
            <= al.offset_hi + 1e-9


def test_frame_video_time():
    assert frame_video_time(30, 30.0) == 1.0
    assert video_to_data_time(1.0, 0.5, 1.0) == 1.5


def test_ambiguous_intervals_detects_rollback():
    assert ambiguous_intervals([0.1, 0.4, 0.8, 0.3, 0.5, 0.9]) == [(0.3, 0.9)]
    assert ambiguous_intervals([0.1, 0.2, 0.3]) == []


# ---------- resample ----------

def test_lookup_uses_timestamps_not_row_index():
    # 3 samples spanning 10 s: row index has nothing to do with seconds
    s = SeriesSampler([0.0, 5.0, 10.0], [0.0, 50.0, 100.0])
    cfg = FieldSampling(method="linear", max_gap=20.0)
    v, st = s.at(2.5, cfg)
    assert st == "ok" and v == pytest.approx(25.0)
    v, st = s.at(7.5, cfg)
    assert st == "ok" and v == pytest.approx(75.0)


def test_gap_beyond_limit_is_missing_not_bridged():
    s = SeriesSampler([0.0, 0.5, 5.0, 5.5], [1.0, 2.0, 3.0, 4.0])
    cfg = FieldSampling(method="linear", max_gap=1.0)
    assert s.at(2.5, cfg)[1] == "missing"      # inside the 4.5 s gap
    assert s.at(0.25, cfg)[0] == pytest.approx(1.5)  # inside a 0.5 s gap


def test_hold_respects_max_age():
    s = SeriesSampler([0.0, 10.0], [7.0, 8.0])
    cfg = FieldSampling(method="hold", max_gap=2.0)
    assert s.at(1.9, cfg)[0] == 7.0
    assert s.at(2.1, cfg)[1] == "missing"


def test_rollback_region_is_ambiguous():
    times = [0.1, 0.4, 0.8, 0.3, 0.5, 0.9]
    amb = ambiguous_intervals(times)
    s = SeriesSampler(times, [1, 2, 3, 4, 5, 6], ambiguous=amb)
    cfg = FieldSampling(method="linear", max_gap=10.0)
    assert s.at(0.6, cfg)[1] == "ambiguous"
    assert s.at(0.05, cfg)[1] == "missing"   # before first sample
    assert s.at(1.0, cfg)[1] == "missing"    # after last sample


def test_unit_conversion():
    assert convert_unit(101.0, "kPa", "hPa") == pytest.approx(1010.0)
    assert convert_unit(20.0, "degC", "K") == pytest.approx(293.15)
    assert convert_unit(5.0, "kPa", "kPa") == 5.0
    with pytest.raises(ValueError):
        convert_unit(1.0, "kPa", "degC")


def test_event_active_window():
    assert event_active([1.5], 1.52, 0.05)
    assert not event_active([1.5], 1.6, 0.05)


# ---------- rendering / versioning ----------

def test_render_produces_manifest_and_video(tiny_project):
    rid = render(tiny_project, base_params())
    d = ProjectStore(tiny_project).render_dir(rid)
    assert (d / "overlay.mp4").exists()
    rows = list(csv.DictReader(open(d / "manifest.csv")))
    assert len(rows) == 30
    # text and curve share one mapping: data_time == video_time*rate+offset
    for r in rows:
        vt, dt = float(r["video_time_s"]), float(r["data_time_s"])
        assert dt == pytest.approx(vt * 1.0 + OFFSET, abs=1e-3)


def test_render_manifest_physics(tiny_project):
    rid = render(tiny_project, base_params())
    d = ProjectStore(tiny_project).render_dir(rid)
    rows = list(csv.DictReader(open(d / "manifest.csv")))
    # flash frame 10 -> video 1.0 -> data 1.5 -> pulse active
    assert rows[10]["stim_pulse_active"] == "1"
    assert rows[9]["stim_pulse_active"] == "0"
    # temperature gap: data t in [1.0, 2.0] -> video t in [0.5, 1.5]
    assert rows[10]["temperature_status"] == "missing"
    assert rows[2]["temperature_status"] == "ok"
    # rollback: data t 0.3..0.9 ambiguous -> video t -0.2..0.4
    assert rows[2]["pressure_status"] == "ambiguous"
    # tail: data ends at 2.55 -> video 2.05; frame 25 (t=2.5) all missing
    assert rows[25]["temperature_status"] == "missing"
    assert rows[25]["banner"] == "no_data"


def test_render_reproducible(tiny_project):
    rid1 = render(tiny_project, base_params())
    rid2 = render(tiny_project, base_params())
    assert rid1 == rid2  # identical inputs -> same candidate, no re-render
    d = ProjectStore(tiny_project).render_dir(rid1)
    first = (d / "manifest.csv").read_bytes()
    render(tiny_project, base_params())
    assert (d / "manifest.csv").read_bytes() == first


def test_revision_makes_new_candidate_and_unit_change(tiny_project):
    rid1 = render(tiny_project, base_params(data_version="v1"))
    rid2 = render(tiny_project, base_params(data_version="v2"))
    assert rid1 != rid2
    store = ProjectStore(tiny_project)
    r1 = list(csv.DictReader(open(store.render_dir(rid1) / "manifest.csv")))
    r2 = list(csv.DictReader(open(store.render_dir(rid2) / "manifest.csv")))
    row1 = next(r for r in r1 if r["pressure_status"] == "ok")
    row2 = r2[r1.index(row1)]
    assert row1["pressure_unit"] == "kPa"
    assert row2["pressure_unit"] == "hPa"
    assert float(row2["pressure"]) == pytest.approx(
        float(row1["pressure"]) * 10, rel=0.01)


def test_export_immutable(tiny_project):
    store = ProjectStore(tiny_project)
    rid = render(tiny_project, base_params())
    eid = store.export(rid, note="release")
    exp_dir = store.root / "exports" / eid
    snap = {p.name: p.read_bytes() for p in exp_dir.iterdir()}
    # a later revision render must not touch the export
    render(tiny_project, base_params(data_version="v2"))
    render(tiny_project, base_params(offset=0.7))
    after = {p.name: p.read_bytes() for p in exp_dir.iterdir()}
    assert snap == after
    # exporting again creates a new export, never overwrites
    eid2 = store.export(rid, note="duplicate release")
    assert eid2 != eid


def test_offset_override_shifts_mapping(tiny_project):
    store = ProjectStore(tiny_project)
    rid = render(tiny_project, base_params(offset=1.0))
    rows = list(csv.DictReader(open(store.render_dir(rid) / "manifest.csv")))
    assert float(rows[10]["data_time_s"]) == pytest.approx(2.0, abs=1e-3)
    # pulse at data 1.5 now appears at video t=0.5 -> frame 5
    assert rows[5]["stim_pulse_active"] == "1"
