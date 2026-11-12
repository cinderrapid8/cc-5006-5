"""Tests for the invalidatable time index, the scrub preview service and
the benchmark's correctness contract."""
import json
import threading
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

from overlay_tool.model import SensorData, Field
from overlay_tool.resample import FieldSampling
from overlay_tool.scrub import preview, open_index, registry, PREVIEW_MAX_POINTS
from overlay_tool.store import ProjectStore
from overlay_tool.time_index import fingerprint_data_file
from overlay_tool.benchmark import LEVELS, run_level, synthesize_workload

FPS = 10
OFFSET = 0.5
CFG = FieldSampling(method="linear", max_gap=0.5)


@pytest.fixture()
def tiny_project(tmp_path):
    root = tmp_path / "tiny"
    root.mkdir(parents=True)
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
    tt = [0.2, 0.55, 0.9, 2.1, 2.4, 2.55]
    tv = [20.0, 20.5, 21.0, 23.0, 23.5, 23.6]
    pt = [0.1, 0.2, 0.4, 0.8, 0.3, 0.5, 0.9, 1.4]
    pv = [100, 100.5, 101, 102, 101.5, 102.5, 103, 104]
    SensorData("v1", fields,
               series={"temperature": (tt, tv), "pressure": (pt, pv)},
               events={"stim_pulse": [1.0 + OFFSET]}).save(root / "data" / "v1.json")
    (root / "anchors.json").write_text(json.dumps({"anchors": [
        {"label": "flash<->pulse", "video_time": 1.0, "data_time": 1.5,
         "note": "t"}]}))
    registry().invalidate(root)
    return root


def params():
    return {
        "data_version": "v1", "offset": OFFSET, "rate": 1.0,
        "window_s": 2.0, "layout": "bottom",
        "fields": [
            {"name": "temperature", "visible": True, "method": "linear",
             "max_gap": 0.5, "display_unit": None, "decimals": 1},
            {"name": "pressure", "visible": True, "method": "linear",
             "max_gap": 0.5, "display_unit": None, "decimals": 1},
            {"name": "stim_pulse", "visible": True},
        ],
    }


# ---------- index exactness ----------

def test_exact_reading_matches_series_sampler(tiny_project):
    store = ProjectStore(tiny_project)
    idx = open_index(store, "v1")
    from overlay_tool.resample import SeriesSampler
    from overlay_tool.timeline import ambiguous_intervals
    for name in ("temperature", "pressure"):
        times, values = idx.data.series[name]
        sampler = SeriesSampler(times, values, ambiguous_intervals(times))
        for t in [0.1, 0.3, 0.7, 1.2, 1.7, 2.2, 2.55, 3.0]:
            got = idx.exact_reading(name, t, CFG, idx.data.fields[name].unit)
            want_v, want_st = sampler.at(t, CFG)
            assert got.status == want_st
            if want_v is not None:
                assert got.value == pytest.approx(want_v)
            else:
                assert got.value is None


def test_exact_reading_applies_unit_conversion(tiny_project):
    store = ProjectStore(tiny_project)
    idx = open_index(store, "v1")
    # t=0.15 brackets samples 0.1/0.2, both before the 0.3..0.9 rollback
    r = idx.exact_reading("pressure", 0.15, CFG, "hPa")
    assert r.status == "ok"
    assert r.value == pytest.approx(convert_v1_pressure(idx, 0.15))


def convert_v1_pressure(idx, t):
    from overlay_tool.resample import convert_unit
    r = idx.fields["pressure"].exact(t, CFG)
    return convert_unit(r.value, "kPa", "hPa")


def test_cursor_and_bisect_locate_agree(tiny_project):
    store = ProjectStore(tiny_project)
    idx = open_index(store, "v1")
    fi = idx.fields["temperature"]
    queries = [0.1, 0.2, 0.55, 0.9, 1.5, 2.1, 2.6]
    assert [fi.locate(q) for q in queries] == \
           [fi.locate_sequential(q) for q in queries]
    # backwards query falls back correctly
    assert fi.locate_sequential(0.3) == fi.locate(0.3)


# ---------- display/exact separation ----------

def test_curve_keeps_gap_and_rollback_spans(tiny_project):
    store = ProjectStore(tiny_project)
    idx = open_index(store, "v1")
    # temperature gap 0.9..2.1 must appear as a missing span
    fc = idx.field_curve("temperature", 0.0, 2.6, CFG, "degC", 120)
    kinds = [s.kind for s in fc.segments]
    assert "missing" in kinds
    # pressure rollback 0.3..0.9 must appear as an ambiguous span
    fcp = idx.field_curve("pressure", 0.0, 1.5, CFG, "kPa", 120)
    assert any(s.kind == "ambiguous" for s in fcp.segments)
    # decimation never exceeds the point budget per ok segment
    for s in fc.segments:
        if s.kind == "ok":
            assert len(s.points) <= PREVIEW_MAX_POINTS


def test_curve_points_are_observed_samples_only(tiny_project):
    store = ProjectStore(tiny_project)
    idx = open_index(store, "v1")
    fc = idx.field_curve("temperature", 0.2, 2.55, CFG, "degC", 4)
    observed = set(idx.data.series["temperature"][0])
    for s in fc.segments:
        if s.kind == "ok":
            for t, _v in s.points:
                assert t in observed


def test_hold_curve_shades_only_part_past_max_gap(tiny_project):
    from overlay_tool.resample import FieldSampling
    store = ProjectStore(tiny_project)
    idx = open_index(store, "v1")
    # temperature samples 0.9 -> 2.1: held value ok for max_gap 0.5 s,
    # missing only on (1.4, 2.1]
    hold_cfg = FieldSampling(method="hold", max_gap=0.5)
    fc = idx.field_curve("temperature", 0.5, 2.4, hold_cfg, "degC", 120)
    missing = [s for s in fc.segments if s.kind == "missing"]
    assert missing, "expected a missing span in the long gap"
    assert all(s.t0 >= 1.4 - 1e-9 for s in missing)
    # the held region before the missing span still draws
    ok_pts = [p for s in fc.segments if s.kind == "ok" for p in s.points]
    assert any(abs(t - 0.9) < 1e-9 for t, _ in ok_pts)


# ---------- preview service ----------

def test_preview_exact_readings_align_with_manifest_semantics(tiny_project):
    store = ProjectStore(tiny_project)
    pv = preview(store, params(), video_time=1.0)
    # video 1.0 -> data 1.5: temperature inside its gap -> missing
    temp = next(r for r in pv["readings"] if r["name"] == "temperature")
    assert temp["status"] == "missing" and temp["value"] is None
    # data 1.5 sits inside pressure rollback 0.3..0.9? no -> check stamp
    pv0 = preview(store, params(), video_time=0.2)  # data 0.7 -> ambiguous
    pres = next(r for r in pv0["readings"] if r["name"] == "pressure")
    assert pres["status"] == "ambiguous"
    # pulse at data 1.5 active at video 1.0
    evt = next(r for r in pv["readings"] if r["name"] == "stim_pulse")
    assert evt["active"] is True
    assert pv["material_token"]["version"] == "v1"


def test_preview_includes_sparse_curve_and_exact_value_separately(tiny_project):
    store = ProjectStore(tiny_project)
    pv = preview(store, params(), video_time=2.0)  # data 2.5 -> ok temp
    temp_curve = next(c for c in pv["curves"] if c["name"] == "temperature")
    temp_read = next(r for r in pv["readings"] if r["name"] == "temperature")
    assert temp_read["status"] == "ok"
    assert any(s["kind"] == "ok" for s in temp_curve["segments"])
    # exact value is delivered outside the curve geometry
    assert isinstance(temp_read["raw_value"], float)


# ---------- invalidation / material versioning ----------

def test_registry_rebuilds_after_file_revision(tiny_project):
    store = ProjectStore(tiny_project)
    idx1 = open_index(store, "v1")
    token1 = fingerprint_data_file(tiny_project / "data" / "v1.json")
    # revise the file in place (same version name, new bytes)
    data = SensorData.load(tiny_project / "data" / "v1.json")
    times, values = data.series["temperature"]
    data.series["temperature"] = (times, [v + 1.0 for v in values])
    data.save(tiny_project / "data" / "v1.json")
    token2 = fingerprint_data_file(tiny_project / "data" / "v1.json")
    assert token1.sha256 != token2.sha256
    idx2 = open_index(store, "v1")
    assert idx2 is not idx1
    assert idx2.token.sha256 == token2.sha256
    # old fingerprint entry is gone: only current material is served
    assert registry().get(tiny_project, "v1").token.sha256 == token2.sha256


def test_explicit_invalidation_drops_entries(tiny_project):
    store = ProjectStore(tiny_project)
    open_index(store, "v1")
    registry().invalidate(tiny_project, "v1")
    assert registry().get(tiny_project, "v1") is None


def test_token_pins_material_even_under_concurrent_access(tiny_project):
    store = ProjectStore(tiny_project)
    idx = open_index(store, "v1")
    seen = []

    def worker():
        seen.append(open_index(store, "v1").token.sha256)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert set(seen) == {idx.token.sha256}


# ---------- benchmark contract ----------

@pytest.mark.benchmark
@pytest.mark.parametrize("level", LEVELS)
def test_benchmark_levels_match_baseline_with_zero_status_drift(level):
    report = run_level(level, repeats=1)
    assert report["correctness"]["status_mismatches"] == 0
    assert report["correctness"]["value_mismatches_vs_baseline"] == 0
    assert report["display_curve"]["status_mismatches"] == 0
    assert report["speedup_wall_x"] >= 1.0


# ---------- HTTP API and background-task version guard ----------

def test_preview_endpoint_serves_current_material(tiny_project):
    from overlay_tool.app import create_app
    app = create_app(tiny_project)
    client = app.test_client()
    resp = client.post("/api/preview", json={
        "video_time_s": 1.0, "params": params()})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["material_token"]["version"] == "v1"
    assert "readings" in body and "curves" in body
    page = client.get("/").get_data(as_text=True)
    assert 'id="scrub"' in page and 'id="scrub_canvas"' in page
    js = client.get("/static/app.js").get_data(as_text=True)
    assert "/api/preview" in js and "paintPreview" in js


def test_background_render_marked_stale_when_material_changes(tiny_project,
                                                              monkeypatch):
    from overlay_tool import app as appmod
    from overlay_tool.app import create_app

    # make the render slow enough that the revision lands while it runs
    import overlay_tool.render as render_mod
    real_render = render_mod.render

    def slow_render(project, p):
        import time as _t
        data = SensorData.load(Path(project) / "data" / "v1.json")
        times, values = data.series["temperature"]
        data.series["temperature"] = (times, [v + 0.5 for v in values])
        data.save(Path(project) / "data" / "v1.json")
        _t.sleep(0.2)
        return real_render(project, p)

    monkeypatch.setattr(appmod, "render", slow_render)
    app = create_app(tiny_project)
    client = app.test_client()
    resp = client.post("/api/render", json=params())
    rid = resp.get_json()["render_id"]
    import time
    for _ in range(50):
        st = client.get(f"/api/render/{rid}").get_json()
        if st["status"] != "running":
            break
        time.sleep(0.1)
    assert st["status"] == "stale"
    # a stale render must not publish a manifest
    assert not (tiny_project / "renders" / rid / "manifest.csv").exists()


@pytest.mark.benchmark
def test_benchmark_workloads_grow_density_and_gaps():
    sizes = [synthesize_workload(lvl, 20261112).meta["n_samples"]
             for lvl in LEVELS]
    # L1..L4 strictly increase observation density; L5 stays dense while
    # adding the longest missing sections
    assert sizes[:4] == sorted(sizes[:4]) and len(set(sizes[:4])) == 4
    assert sizes[4] > sizes[1]
    # every workload retains non-monotonic rollback structure
    for lvl in LEVELS:
        data = synthesize_workload(lvl, 20261112)
        times = data.series["probe"][0]
        assert any(times[i] < times[i - 1] for i in range(1, len(times)))
