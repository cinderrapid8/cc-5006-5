"""Flask backend: serves the project state, accepts render parameter
adjustments from the web page, runs renders in the background, and
freezes exports.

Run:  python -m overlay_tool.app --project projects/demo --port 8000
"""
from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from .demo import DEFAULT_PARAMS
from .model import load_anchors
from .render import render, complete_params
from .store import ProjectStore
from .timeline import fit_alignment

STATIC = Path(__file__).parent / "static"


def create_app(project: Path) -> Flask:
    app = Flask(__name__, static_folder=str(STATIC), static_url_path="/static")
    project = Path(project).resolve()
    store = ProjectStore(project)
    jobs: dict[str, dict] = {}

    @app.get("/")
    def index():
        return send_from_directory(STATIC, "index.html")

    @app.get("/api/project")
    def project_state():
        meta = store.source_meta()
        anchors = [a.__dict__ for a in load_anchors(store.anchors_path())]
        alignment = fit_alignment(load_anchors(store.anchors_path())).to_dict()
        versions = {}
        for v in store.data_versions():
            d = store.load_data(v)
            versions[v] = {
                "sha256": d.sha256()[:12],
                "meta": d.meta,
                "time_range": d.time_range(),
                "fields": {k: {"label": f.label, "unit": f.unit, "kind": f.kind,
                               "color": f.color}
                           for k, f in d.fields.items()},
            }
        return jsonify({
            "source": meta,
            "anchors": anchors,
            "alignment": alignment,
            "data_versions": versions,
            "default_params": DEFAULT_PARAMS,
            "renders": store.list_renders(),
            "exports": store.list_exports(),
        })

    @app.post("/api/render")
    def start_render():
        params = request.get_json(force=True)
        if "data_version" not in params:
            return jsonify({"error": "data_version required"}), 400
        try:
            full = complete_params(dict(params), store)
        except (ValueError, KeyError) as e:
            return jsonify({"error": str(e)}), 400
        from .store import render_id_for
        rid = render_id_for(full)
        if (store.render_dir(rid) / "manifest.csv").exists():
            return jsonify({"render_id": rid, "status": "done"})

        def work():
            jobs[rid] = {"status": "running"}
            try:
                render(project, params)
                jobs[rid] = {"status": "done"}
            except Exception as e:  # surfaced via /api/render/<rid>
                jobs[rid] = {"status": "error", "error": str(e)}

        threading.Thread(target=work, daemon=True).start()
        jobs[rid] = {"status": "running"}
        return jsonify({"render_id": rid, "status": "running"})

    @app.get("/api/render/<rid>")
    def render_status(rid):
        job = jobs.get(rid)
        done = (store.render_dir(rid) / "manifest.csv").exists()
        if done:
            return jsonify({"render_id": rid, "status": "done"})
        if job:
            return jsonify({"render_id": rid, **job})
        return jsonify({"render_id": rid, "status": "unknown"}), 404

    @app.post("/api/export")
    def export():
        body = request.get_json(force=True)
        try:
            eid = store.export(body["render_id"], note=body.get("note", ""))
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        return jsonify({"export_id": eid})

    @app.get("/api/qc")
    def qc():
        qc_file = project / "qc_report.json"
        if qc_file.exists():
            return jsonify(json.loads(qc_file.read_text()))
        return jsonify({"all_pass": None, "checks": []})

    @app.get("/files/<path:relpath>")
    def files(relpath):
        return send_from_directory(project, relpath)

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="projects/demo")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    app = create_app(Path(args.project))
    app.run(host="127.0.0.1", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
