"""Synthesize the demo experiment.

Scene: a camera films a rig with a rotating dial and a stimulator LED.
The camera burns in its own clock (which runs 2.0 s behind the video
timeline). The LED flashes at known video times; the data logger
records matching stimulus pulses on its own clock, offset by
+1.35 s. Sensor streams are non-uniformly sampled, contain a dropout
gap and a logger clock rollback, and stop before the video ends.

Data version v2 is a later revision: pressure re-calibrated and
re-expressed in hPa, temperature bias-corrected, and the rollback
repaired from the logger drift table.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .model import SensorData, Field
from .store import ProjectStore

FPS = 30
DURATION_S = 20.0
W, H = 1024, 576
TRUE_OFFSET = 1.35          # data clock = video clock + 1.35 s
CAMERA_CLOCK_LAG = 2.0      # camera's burned-in clock reads video_t - 2 s
FLASH_VIDEO_TIMES = [2.5, 7.0, 11.5, 16.0]
DATA_END = 18.0             # logger stops here (video runs to 20 s)
DATA_START = 2.0            # logger starts here (video already running)
TEMP_GAP = (10.0, 14.0)     # logger dropout, data-time seconds
# Pressure-channel timestamp glitch (the master clock stays correct, so
# stim pulses are unaffected): physical 15.4..16.6 s was stamped as
# 13.8..15.0 s by a faulty re-sampling buffer.
GLITCH_PHYS = (15.4, 16.6)  # physical span mis-stamped
GLITCH_STAMP = (13.8, 15.0) # stamps it was given


def pulse_times_data() -> list[float]:
    return [round(t + TRUE_OFFSET, 3) for t in FLASH_VIDEO_TIMES]


def _phys_temperature(t: float) -> float:
    return 21.5 + 0.9 * math.sin(2 * math.pi * (t - 3) / 30.0) + 0.02 * t


def _phys_pressure_kpa(t: float) -> float:
    p = 101.3
    for tp in pulse_times_data():
        if t >= tp:
            p += 8.0 * math.exp(-(t - tp) / 1.5)
    return p


def make_source_video(path: Path) -> dict:
    """Dark rig, rotating dial, burned-in camera clock, white LED flashes
    lasting two frames."""
    n_frames = int(FPS * DURATION_S)
    flash_frames = set()
    for t in FLASH_VIDEO_TIMES:
        flash_frames.add(round(t * FPS))
        flash_frames.add(round(t * FPS) + 1)
    font = ImageFont.load_default(size=18)
    writer = imageio.get_writer(str(path), fps=FPS, codec="libx264",
                                quality=7, pixelformat="yuv420p",
                                macro_block_size=16)
    try:
        for fi in range(n_frames):
            t = fi / FPS
            img = Image.new("RGB", (W, H), (24, 26, 30))
            d = ImageDraw.Draw(img)
            # bench + rig
            d.rectangle([0, H - 90, W, H], fill=(38, 40, 46))
            d.rectangle([120, 130, 380, H - 90], fill=(52, 55, 62))
            # rotating dial
            cx, cy, r = 250, 300, 110
            d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(120, 125, 135), width=4)
            ang = 2 * math.pi * t / 8.0
            d.line([cx, cy, cx + r * math.cos(ang), cy + r * math.sin(ang)],
                   fill=(200, 205, 215), width=5)
            for k in range(12):
                a = 2 * math.pi * k / 12
                d.ellipse([cx + (r - 12) * math.cos(a) - 3, cy + (r - 12) * math.sin(a) - 3,
                           cx + (r - 12) * math.cos(a) + 3, cy + (r - 12) * math.sin(a) + 3],
                          fill=(110, 115, 125))
            # stimulator LED
            led_on = fi in flash_frames
            led = (255, 250, 240) if led_on else (90, 40, 40)
            d.ellipse([620, 180, 700, 260], fill=led, outline=(60, 60, 60))
            d.text((614, 270), "STIM LED", font=font, fill=(150, 150, 150))
            if led_on:
                arr = np.asarray(img).astype(np.uint16) + 55
                img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
                d = ImageDraw.Draw(img)
            # camera's own clock, 2 s behind the true video timeline
            d.rectangle([W - 250, H - 40, W - 10, H - 10], fill=(0, 0, 0))
            d.text((W - 240, H - 36), f"CAM {t - CAMERA_CLOCK_LAG:07.3f}",
                   font=font, fill=(220, 220, 120))
            # sensor noise floor so frames are not sterile
            rng_np = np.random.default_rng(fi)
            arr = np.asarray(img).astype(np.int16)
            arr[-70:-6, -70:-6] += rng_np.integers(-4, 5, size=(64, 64, 3))
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
            writer.append_data(np.asarray(img))
    finally:
        writer.close()
    return {"fps": FPS, "width": W, "height": H,
            "frames": n_frames, "duration_s": DURATION_S,
            "camera_clock_lag_s": CAMERA_CLOCK_LAG,
            "flash_video_times_s": FLASH_VIDEO_TIMES}


def _sample_stream(rng, t_lo, t_hi, dt_lo, dt_hi):
    t, out = t_lo, []
    while t <= t_hi:
        out.append(round(t, 3))
        t += rng.uniform(dt_lo, dt_hi)
    return out


def _pressure_segments(rng):
    """Three (stamps, physical_times) segments. The middle one carries
    the mis-stamped samples; before/after it stamps equal physical time."""
    a_phys = _sample_stream(rng, DATA_START, GLITCH_PHYS[0], 0.07, 0.13)
    b_phys = _sample_stream(rng, GLITCH_PHYS[0], GLITCH_PHYS[1], 0.07, 0.13)
    c_phys = _sample_stream(rng, GLITCH_PHYS[1], DATA_END, 0.07, 0.13)
    span_p = GLITCH_PHYS[1] - GLITCH_PHYS[0]
    span_s = GLITCH_STAMP[1] - GLITCH_STAMP[0]
    b_stamp = [round(GLITCH_STAMP[0] + (p - GLITCH_PHYS[0]) / span_p * span_s, 3)
               for p in b_phys]
    return [(a_phys, a_phys), (b_stamp, b_phys), (c_phys, c_phys)]


def _build_data():
    """Shared sampling instants for both revisions."""
    rng = random.Random(11)
    tt = [t for t in _sample_stream(rng, DATA_START, DATA_END, 0.3, 0.9)
          if not (TEMP_GAP[0] <= t <= TEMP_GAP[1])]
    tv = [round(_phys_temperature(t) + rng.gauss(0, 0.05), 3) for t in tt]
    segs = _pressure_segments(rng)
    pv = [round(_phys_pressure_kpa(p) + rng.gauss(0, 0.08), 3)
          for _, phys in segs for p in phys]
    return tt, tv, segs, pv


def make_data_v1() -> SensorData:
    tt, tv, segs, pv = _build_data()
    pt = [s for stamps, _ in segs for s in stamps]
    fields = {
        "temperature": Field("temperature", "Temp", "degC", "continuous", "#ffb000"),
        "pressure": Field("pressure", "Pressure", "kPa", "continuous", "#33ccff"),
        "stim_pulse": Field("stim_pulse", "STIM", "", "event", "#ff5555"),
    }
    return SensorData(
        version="v1", fields=fields,
        series={"temperature": (tt, tv), "pressure": (pt, pv)},
        events={"stim_pulse": pulse_times_data()},
        meta={"logger": "daq-01",
              "note": ("raw export; pressure-channel timestamp glitch "
                       "(physical 15.4..16.6 s stamped as 13.8..15.0 s) "
                       "uncorrected"),
              "recorded_range_s": [DATA_START, DATA_END],
              "temperature_dropout_s": list(TEMP_GAP)})


def make_data_v2() -> SensorData:
    """Revision: pressure in hPa with +2% recalibration, temperature bias
    -0.3 degC, pressure timestamps repaired via the buffer diagnostic."""
    tt, tv, segs, pv = _build_data()
    tv2 = [round(v - 0.3, 3) for v in tv]
    pt2 = [p for _, phys in segs for p in phys]  # repaired: physical times
    pv2 = [round(v * 10 * 1.02, 3) for v in pv]  # kPa -> hPa, recalibrated
    fields = {
        "temperature": Field("temperature", "Temp", "degC", "continuous", "#ffb000"),
        "pressure": Field("pressure", "Pressure", "hPa", "continuous", "#33ccff"),
        "stim_pulse": Field("stim_pulse", "STIM", "", "event", "#ff5555"),
    }
    return SensorData(
        version="v2", fields=fields,
        series={"temperature": (list(tt), tv2), "pressure": (pt2, pv2)},
        events={"stim_pulse": pulse_times_data()},
        meta={"logger": "daq-01", "revises": "v1",
              "note": ("pressure re-calibrated +2% and expressed in hPa; "
                       "temperature bias -0.3 degC; pressure timestamp "
                       "glitch repaired via buffer diagnostic rev2"),
              "recorded_range_s": [DATA_START, DATA_END],
              "temperature_dropout_s": list(TEMP_GAP)})


def make_anchors() -> list[dict]:
    """Human-placed anchors: the frame an operator picked as the LED's
    brightest, paired with the logged pulse. Operator picks are off by up
    to one frame, which is exactly what the uncertainty range is for."""
    human_video_picks = [2.500, 7.033, 11.467, 16.000]  # ±1 frame of truth
    return [
        {"label": f"flash#{k + 1} <-> pulse#{k + 1}",
         "video_time": human_video_picks[k],
         "data_time": pulse_times_data()[k],
         "note": "operator-picked brightest LED frame vs logger pulse"}
        for k in range(4)
    ]


def synthesize(project: Path) -> None:
    store = ProjectStore(project)
    if not store.source_video.exists():
        meta = make_source_video(store.source_video)
        store.save_source_meta(meta)
    store.save_data(make_data_v1())
    store.save_data(make_data_v2())
    store.save_anchors(make_anchors())
    (project / "ground_truth.json").write_text(json.dumps({
        "true_offset_s": TRUE_OFFSET,
        "camera_clock_lag_s": CAMERA_CLOCK_LAG,
        "flash_video_times_s": FLASH_VIDEO_TIMES,
        "pulse_data_times_s": pulse_times_data(),
        "temp_gap_data_s": list(TEMP_GAP),
        "rollback_data_s": {"physical_span": list(GLITCH_PHYS),
                            "misstamped_as": list(GLITCH_STAMP)},
        "data_range_s": [DATA_START, DATA_END],
    }, indent=2))
