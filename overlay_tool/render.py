"""Frame renderer: burns curves, current readings and event markers into
the video, and writes a per-frame manifest so every number on screen can
be checked against the recorded data time.

Within one frame, the curve window and the numeric readouts are computed
from the same data_time value, so text and curve always refer to the
same instant.
"""
from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .model import SensorData, load_anchors
from .resample import (FieldSampling, SeriesSampler, convert_unit,
                       event_active, events_in_window)
from .timeline import (ambiguous_intervals, fit_alignment, frame_video_time,
                       video_to_data_time)
from .store import ProjectStore, render_id_for, canonical_params

CURVE_SAMPLES = 160  # resampling points across the curve window
RENDERER_VERSION = "overlay-1.1"  # bump when drawing code changes pixels


def complete_params(params: dict, store: ProjectStore) -> dict:
    """Fill in everything the render depends on, so the render id hashes
    all inputs."""
    p = json.loads(json.dumps(params))  # deep copy: never mutate caller's dicts
    p.setdefault("renderer", RENDERER_VERSION)
    meta = store.source_meta()
    p.setdefault("fps", meta["fps"])
    p.setdefault("width", meta["width"])
    p.setdefault("height", meta["height"])
    p.setdefault("rate", 1.0)
    p.setdefault("layout", "bottom")
    p.setdefault("window_s", 6.0)
    p.setdefault("frame_limit", None)
    data = store.load_data(p["data_version"])
    p["data_sha256"] = data.sha256()
    anchors_file = store.anchors_path()
    p["anchors_sha256"] = hashlib.sha256(
        anchors_file.read_bytes()).hexdigest() if anchors_file.exists() else None
    # fixed y-ranges (display units) so re-renders are identical
    for f in p["fields"]:
        if f["name"] in data.series and ("y_min" not in f or "y_max" not in f):
            unit = f.get("display_unit") or data.fields[f["name"]].unit
            vals = [convert_unit(v, data.fields[f["name"]].unit, unit)
                    for v in data.series[f["name"]][1] if v is not None]
            lo, hi = min(vals), max(vals)
            pad = (hi - lo) * 0.08 or 1.0
            f.setdefault("y_min", round(lo - pad, 3))
            f.setdefault("y_max", round(hi + pad, 3))
    return p


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=size)


def _hex(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


class _FieldState:
    def __init__(self, fcfg: dict, data: SensorData):
        self.cfg = fcfg
        self.name = fcfg["name"]
        self.field = data.fields[self.name]
        self.kind = self.field.kind
        self.display_unit = fcfg.get("display_unit") or self.field.unit
        self.sampling = FieldSampling(
            method=fcfg.get("method", "linear"),
            max_gap=fcfg.get("max_gap", 1.0),
            display_unit=self.display_unit,
            decimals=fcfg.get("decimals", 1),
        )
        if self.kind == "continuous":
            times, values = data.series[self.name]
            self.sampler = SeriesSampler(times, values,
                                         ambiguous_intervals(times))
            self.ambiguous = self.sampler.ambiguous
        else:
            self.event_times = sorted(data.events.get(self.name, []))

    def reading(self, t: float):
        """(display_value_in_display_unit, status)"""
        v, status = self.sampler.at(t, self.sampling)
        if v is None:
            return None, status
        return convert_unit(v, self.field.unit, self.display_unit), status


def _draw_curve_panel(draw: ImageDraw.ImageDraw, box, st: _FieldState,
                      t_now: float, window_s: float):
    x0, y0, x1, y1 = box
    lo_t, hi_t = t_now - window_s, t_now
    y_min, y_max = st.cfg["y_min"], st.cfg["y_max"]

    def xy(t, v):
        x = x0 + (t - lo_t) / (hi_t - lo_t) * (x1 - x0)
        y = y1 - (v - y_min) / (y_max - y_min) * (y1 - y0)
        return x, y

    # shade missing / ambiguous runs instead of drawing through them
    pts, seg = [], []
    runs = []  # (kind, x_start, x_end)
    run_kind, run_start = None, None
    for k in range(CURVE_SAMPLES + 1):
        t = lo_t + (hi_t - lo_t) * k / CURVE_SAMPLES
        v, status = st.reading(t)
        x = x0 + (t - lo_t) / (hi_t - lo_t) * (x1 - x0)
        if status != "ok":
            if run_kind is None:
                run_kind, run_start = status, x
            run_end = x
            if seg:
                pts.append(seg)
                seg = []
        else:
            if run_kind is not None:
                runs.append((run_kind, run_start, x))
                run_kind = None
            seg.append((x, y1 - (v - y_min) / (y_max - y_min) * (y1 - y0)))
    if run_kind is not None:
        runs.append((run_kind, run_start, x1))
    if seg:
        pts.append(seg)

    overlay_runs = {"missing": (255, 60, 60, 26), "ambiguous": (255, 200, 40, 34)}
    for kind, a, b in runs:
        draw.rectangle([a, y0, b, y1], fill=overlay_runs.get(kind, (255, 60, 60, 26)))

    color = _hex(st.cfg.get("color", "#ffb000"))
    for seg in pts:
        if len(seg) >= 2:
            draw.line(seg, fill=color, width=2)
        else:
            draw.point(seg[0], fill=color)

    # current-instant marker at the right edge
    v, status = st.reading(t_now)
    if status == "ok":
        cx, cy = xy(t_now, v)
        draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=color)
    draw.rectangle(box, outline=(90, 90, 90))


def render(project: Path | str, params: dict) -> str:
    store = ProjectStore(Path(project))
    p = complete_params(params, store)
    rid = render_id_for(p)
    out = store.render_dir(rid)
    if (out / "manifest.csv").exists():
        return rid  # identical inputs: candidate already exists
    out.mkdir(parents=True, exist_ok=True)

    data = store.load_data(p["data_version"])
    anchors = load_anchors(store.anchors_path())
    alignment = fit_alignment(anchors)
    fps, W, H = p["fps"], p["width"], p["height"]
    offset, rate = p["offset"], p["rate"]
    window_s = p["window_s"]
    fields = [_FieldState(f, data) for f in p["fields"] if f.get("visible", True)]
    cont = [st for st in fields if st.kind == "continuous"]
    evts = [st for st in fields if st.kind == "event"]

    src = imageio.get_reader(str(store.source_video))
    n_src = src.count_frames()
    n_frames = min(n_src, p["frame_limit"]) if p["frame_limit"] else n_src

    font_s, font_m, font_l = _font(15), _font(19), _font(24)
    strip_h = 190 if p["layout"] == "bottom" else 150
    panel_top = H - strip_h

    rows = []
    t0 = time.time()
    writer = imageio.get_writer(
        str(out / "overlay.mp4"), fps=fps, codec="libx264", quality=8,
        pixelformat="yuv420p", macro_block_size=16)
    try:
        for fi in range(n_frames):
            frame = Image.fromarray(src.get_data(fi)).convert("RGBA")
            t_video = frame_video_time(fi, fps)
            t_data = video_to_data_time(t_video, offset, rate)
            layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            draw = ImageDraw.Draw(layer)

            # ---- header: both clocks + uncertainty range ----
            draw.rectangle([0, 0, W, 30], fill=(0, 0, 0, 150))
            head = (f"video {t_video:8.3f}s   data {t_data:8.3f}s   "
                    f"offset {offset:+.3f}s "
                    f"(anchor range {alignment.offset_lo:+.3f}..{alignment.offset_hi:+.3f}s)")
            draw.text((8, 7), head, font=font_s, fill=(230, 230, 230))

            # ---- curve strip ----
            draw.rectangle([0, panel_top, W, H], fill=(10, 14, 20, 185))
            n_pan = max(1, len(cont))
            pan_h = (strip_h - 46) // n_pan
            curve_x1 = W - (220 if p["layout"] == "bottom" else 200)
            for i, st in enumerate(cont):
                box = (10, panel_top + 8 + i * pan_h, curve_x1,
                       panel_top + 8 + (i + 1) * pan_h - 8)
                _draw_curve_panel(draw, box, st, t_data, window_s)
                draw.text((box[0] + 4, box[1] + 2),
                          f"{st.field.label} [{st.display_unit}]",
                          font=font_s, fill=(200, 200, 200))

            # event tick row under the panels
            tick_y = H - 30
            for st in evts:
                for te in events_in_window(st.event_times, t_data - window_s, t_data):
                    x = 10 + (te - (t_data - window_s)) / window_s * (curve_x1 - 10)
                    c = _hex(st.cfg.get("color", "#ff5555"))
                    draw.line([(x, tick_y), (x, tick_y + 14)], fill=c, width=2)
            draw.rectangle([10, tick_y - 2, curve_x1, tick_y + 16],
                           outline=(70, 70, 70))

            # ---- current readings (same t_data as the curves) ----
            rx = curve_x1 + 14
            ry = panel_top + 12
            row = {"frame": fi, "video_time_s": round(t_video, 4),
                   "data_time_s": round(t_data, 4)}
            any_ok = False
            for st in cont:
                v, status = st.reading(t_data)
                dec = st.sampling.decimals
                if status == "ok":
                    any_ok = True
                    txt = f"{v:.{dec}f} {st.display_unit}"
                    row[st.name] = round(v, dec)
                    row[st.name + "_status"] = "ok"
                else:
                    txt = "N/A" if status == "missing" else "rollback"
                    row[st.name] = ""
                    row[st.name + "_status"] = status
                row[st.name + "_unit"] = st.display_unit
                draw.text((rx, ry), st.field.label, font=font_s,
                          fill=(180, 180, 180))
                draw.text((rx, ry + 16), txt, font=font_l,
                          fill=_hex(st.cfg.get("color", "#ffb000"))
                          if status == "ok" else (150, 150, 150))
                ry += 62
            for st in evts:
                active = event_active(st.event_times, t_data, 0.5 / fps)
                row[st.name + "_active"] = int(active)
                if active:
                    draw.text((rx, ry), f"[{st.field.label}]", font=font_m,
                              fill=_hex(st.cfg.get("color", "#ff5555")))
                    any_ok = True
                ry += 30

            if not any_ok:
                draw.rectangle([W // 2 - 210, 44, W // 2 + 210, 76],
                               fill=(120, 20, 20, 200))
                draw.text((W // 2 - 200, 50),
                          "NO DATA - outside logger recording range",
                          font=font_m, fill=(255, 220, 220))
                row["banner"] = "no_data"
            else:
                row["banner"] = ""

            rows.append(row)
            frame = Image.alpha_composite(frame, layer).convert("RGB")
            writer.append_data(np.asarray(frame))
    finally:
        writer.close()
        src.close()

    # ---- manifest: the checkable per-frame reading list ----
    colnames = ["frame", "video_time_s", "data_time_s"]
    for st in cont:
        colnames += [st.name, st.name + "_status", st.name + "_unit"]
    for st in evts:
        colnames += [st.name + "_active"]
    colnames += ["banner"]
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=colnames)
        w.writeheader()
        w.writerows(rows)

    qc = {
        "frames": n_frames,
        "seconds": round(n_frames / fps, 3),
        "render_wall_s": round(time.time() - t0, 2),
        "missing_readings": sum(1 for r in rows for st in cont
                                if r[st.name + "_status"] == "missing"),
        "ambiguous_readings": sum(1 for r in rows for st in cont
                                  if r[st.name + "_status"] == "ambiguous"),
        "rollback_intervals_s": {st.name: st.ambiguous for st in cont},
        "data_range_s": data.time_range(),
        "alignment": alignment.to_dict(),
        "note": ("offset range brackets all manual anchors; it is an "
                 "empirical bracket, not a precision guarantee"),
    }
    (out / "manifest.json").write_text(json.dumps(
        {"render_id": rid, "params": p, "qc": qc}, indent=2))
    (out / "params.json").write_text(canonical_params(p))
    (out / "meta.json").write_text(json.dumps(
        {"render_id": rid, "created": time.strftime("%Y-%m-%dT%H:%M:%S")}))
    return rid
