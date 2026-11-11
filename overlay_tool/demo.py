"""End-to-end demo: synthesize the experiment, verify the alignment and
sampling choices against the known flash/pulse pairs, render candidates
for both data revisions, and export reproducible releases.

Run:  python -m overlay_tool.demo projects/demo
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from .model import load_anchors
from .render import render
from .store import ProjectStore
from .synth import synthesize
from .timeline import fit_alignment, video_to_data_time

DEFAULT_PARAMS = {
    "layout": "bottom",
    "window_s": 6.0,
    "fields": [
        {"name": "temperature", "visible": True, "method": "linear",
         "max_gap": 1.5, "display_unit": None, "decimals": 1, "color": "#ffb000"},
        {"name": "pressure", "visible": True, "method": "linear",
         "max_gap": 0.4, "display_unit": None, "decimals": 1, "color": "#33ccff"},
        {"name": "stim_pulse", "visible": True, "color": "#ff5555"},
    ],
}


def _rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _row_at_video_time(rows, fps, t_video):
    return rows[round(t_video * fps)]


def run_qc(project: Path, rid_v1: str, rid_v2: str) -> dict:
    store = ProjectStore(project)
    truth = json.loads((project / "ground_truth.json").read_text())
    meta = store.source_meta()
    fps = meta["fps"]
    alignment = fit_alignment(load_anchors(store.anchors_path()))
    rows1 = _rows(store.render_dir(rid_v1) / "manifest.csv")
    rows2 = _rows(store.render_dir(rid_v2) / "manifest.csv")
    checks = []

    def check(name, ok, detail):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})
        return ok

    # 1. anchors recover the known logger offset, range brackets it
    off_err = abs(alignment.offset - truth["true_offset_s"])
    check("offset_fit", off_err <= 0.05
          and alignment.offset_lo <= truth["true_offset_s"] <= alignment.offset_hi,
          f"fitted {alignment.offset:+.4f}s vs true {truth['true_offset_s']:+.3f}s; "
          f"range [{alignment.offset_lo:+.4f}, {alignment.offset_hi:+.4f}]")

    # 2. every flash frame shows an active pulse in the manifest
    for k, tv in enumerate(truth["flash_video_times_s"]):
        r = _row_at_video_time(rows1, fps, tv)
        check(f"flash_pulse_{k+1}", r["stim_pulse_active"] == "1",
              f"frame {r['frame']} video_t={r['video_time_s']} "
              f"data_t={r['data_time_s']} active={r['stim_pulse_active']}")

    # 3. temperature dropout is reported as missing, not bridged
    gap_mid_data = sum(truth["temp_gap_data_s"]) / 2
    tv_gap = (gap_mid_data - alignment.offset) / alignment.rate
    r = _row_at_video_time(rows1, fps, tv_gap)
    check("dropout_is_missing", r["temperature_status"] == "missing",
          f"video_t={r['video_time_s']} temp_status={r['temperature_status']}")

    # 4. v1 rollback region is flagged ambiguous; v2 repair reads normally
    rb = truth["rollback_data_s"]
    rb_mid = sum(rb["misstamped_as"]) / 2
    tv_rb = (rb_mid - alignment.offset) / alignment.rate
    r1 = _row_at_video_time(rows1, fps, tv_rb)
    r2 = _row_at_video_time(rows2, fps, tv_rb)
    check("rollback_ambiguous_v1", r1["pressure_status"] == "ambiguous",
          f"v1 pressure_status={r1['pressure_status']} at data_t={r1['data_time_s']}")
    check("rollback_repaired_v2", r2["pressure_status"] == "ok",
          f"v2 pressure_status={r2['pressure_status']} value={r2['pressure']}")

    # 5. tail beyond logger range shows no data
    r = rows1[-1]
    check("tail_no_data", r["banner"] == "no_data"
          and r["temperature_status"] == "missing"
          and r["pressure_status"] == "missing",
          f"last frame video_t={r['video_time_s']} banner={r['banner']!r}")

    # 6. unit revision: v2 pressure is hPa and matches v1 kPa x10 x1.02
    common = None
    for a, b in zip(rows1, rows2):
        if a["pressure_status"] == "ok" and b["pressure_status"] == "ok":
            common = (a, b)
            break
    if common:
        a, b = common
        ratio = float(b["pressure"]) / float(a["pressure"])
        check("unit_revision_hpa",
              b["pressure_unit"] == "hPa" and abs(ratio - 10.2) < 0.15,
              f"v1 {a['pressure']} {a['pressure_unit']} -> "
              f"v2 {b['pressure']} {b['pressure_unit']} (ratio {ratio:.2f})")
    else:
        check("unit_revision_hpa", False, "no common ok frame found")

    # 7. head of video precedes first sample -> missing, not row zero
    r = rows1[0]
    check("head_before_first_sample", r["temperature_status"] == "missing",
          f"frame0 data_t={r['data_time_s']} status={r['temperature_status']}")

    qc = {"all_pass": all(c["pass"] for c in checks), "checks": checks}
    (project / "qc_report.json").write_text(json.dumps(qc, indent=2))
    return qc


def run(project: Path) -> dict:
    project = Path(project)
    print(f"[demo] synthesizing experiment in {project}")
    synthesize(project)
    store = ProjectStore(project)
    alignment = fit_alignment(load_anchors(store.anchors_path()))
    print(f"[demo] anchor fit: offset {alignment.offset:+.4f}s "
          f"range [{alignment.offset_lo:+.4f}, {alignment.offset_hi:+.4f}] "
          f"rate {alignment.rate:.6f}")

    params = dict(DEFAULT_PARAMS, offset=round(alignment.offset, 4),
                  rate=round(alignment.rate, 6))
    (project / "render_params.json").write_text(json.dumps(params, indent=2))

    print("[demo] rendering candidate on data v1 ...")
    rid1 = render(project, dict(params, data_version="v1"))
    print(f"[demo] v1 candidate {rid1}")
    exp1 = store.export(rid1, note="initial release on raw export v1")

    print("[demo] data revised to v2 -> new candidate render")
    rid2 = render(project, dict(params, data_version="v2"))
    print(f"[demo] v2 candidate {rid2}")
    exp2 = store.export(rid2, note="revised units hPa, bias fix, rollback repaired")

    print("[demo] running QC checks ...")
    qc = run_qc(project, rid1, rid2)
    for c in qc["checks"]:
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['check']}: {c['detail']}")
    print(f"[demo] exports: {exp1}, {exp2}")
    print(f"[demo] QC {'ALL PASS' if qc['all_pass'] else 'HAS FAILURES'}")
    return {"rid_v1": rid1, "rid_v2": rid2, "exports": [exp1, exp2], "qc": qc}


if __name__ == "__main__":
    proj = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("projects/demo")
    result = run(proj)
    sys.exit(0 if result["qc"]["all_pass"] else 1)
