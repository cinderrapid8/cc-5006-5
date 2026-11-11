"""Project store: versioned data, candidate renders, immutable exports.

A render id is the hash of its full parameter set (including the data
version hash), so identical parameters always map to the same candidate.
Exporting freezes a render directory; exports are never overwritten, so
an exported result stays reproducible even after the source data is
revised and new candidates appear.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

from .model import SensorData


def canonical_params(params: dict) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def render_id_for(params: dict) -> str:
    return hashlib.sha256(canonical_params(params).encode()).hexdigest()[:12]


class ProjectStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        for sub in ("data", "renders", "exports"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    # ---- data versions ----
    def save_data(self, data: SensorData) -> None:
        data.save(self.root / "data" / f"{data.version}.json")

    def load_data(self, version: str) -> SensorData:
        return SensorData.load(self.root / "data" / f"{version}.json")

    def data_versions(self) -> list[str]:
        return sorted(p.stem for p in (self.root / "data").glob("*.json"))

    # ---- source video ----
    @property
    def source_video(self) -> Path:
        return self.root / "source.mp4"

    def save_source_meta(self, meta: dict) -> None:
        (self.root / "source_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8")

    def source_meta(self) -> dict:
        return json.loads((self.root / "source_meta.json").read_text())

    # ---- anchors ----
    def save_anchors(self, anchors: list[dict]) -> None:
        (self.root / "anchors.json").write_text(
            json.dumps({"anchors": anchors}, indent=2), encoding="utf-8")

    def anchors_path(self) -> Path:
        return self.root / "anchors.json"

    # ---- renders ----
    def render_dir(self, render_id: str) -> Path:
        return self.root / "renders" / render_id

    def list_renders(self) -> list[dict]:
        out = []
        for d in sorted((self.root / "renders").iterdir()):
            pj = d / "params.json"
            if pj.exists():
                params = json.loads(pj.read_text())
                out.append({
                    "render_id": d.name,
                    "created": params.get("_created"),
                    "data_version": params.get("data_version"),
                    "offset": params.get("offset"),
                    "done": (d / "manifest.csv").exists(),
                    "exported": self._is_exported(d.name),
                })
        return out

    def _is_exported(self, render_id: str) -> bool:
        exp = self.root / "exports"
        return any((exp / e / "render_id.txt").exists()
                   and (exp / e / "render_id.txt").read_text().strip() == render_id
                   for e in [p.name for p in exp.iterdir()] ) if exp.exists() else False

    # ---- exports ----
    def export(self, render_id: str, note: str = "") -> str:
        src = self.render_dir(render_id)
        if not (src / "manifest.csv").exists():
            raise ValueError(f"render {render_id} not finished")
        base = time.strftime("exp_%Y%m%d_%H%M%S") + "_" + render_id[:6]
        export_id, n = base, 2
        while (self.root / "exports" / export_id).exists():
            export_id, n = f"{base}_{n}", n + 1
        dst = self.root / "exports" / export_id
        shutil.copytree(src, dst)
        (dst / "render_id.txt").write_text(render_id)
        (dst / "export.json").write_text(json.dumps({
            "export_id": export_id,
            "render_id": render_id,
            "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "note": note,
        }, indent=2))
        return export_id

    def list_exports(self) -> list[dict]:
        out = []
        exp = self.root / "exports"
        for d in sorted(exp.iterdir()) if exp.exists() else []:
            ej = d / "export.json"
            if ej.exists():
                out.append(json.loads(ej.read_text()))
        return out
