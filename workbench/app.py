"""Flask application for the Hydrology Calculation Workbench."""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

from .catalog import ALGORITHM_BY_ID, catalog_payload, get_algorithm
from .common import (
    RUN_DIR,
    RUNTIME_DIR,
    UPLOAD_DIR,
    RunContext,
    cleanup_runtime,
    dataframe_preview,
    ensure_dirs,
    exc_message,
    locate_shapefile,
    read_csv,
    sample_path,
    safe_name,
)

from . import engineering, flood, muskingum, thiessen, unit_hydrograph

try:
    import geopandas as gpd
except Exception:  # pragma: no cover - optional at import time
    gpd = None

APP_NAME = "Hydrology Workbench"
APP_VERSION = "1.0.0"

# token id -> {"path": Path, "name": str, "kind": str, "role": str}
UPLOADS = {}
RUN_LOCK = threading.Lock()

RUNNERS = {
    "eng_linear_interp": engineering.run_linear_interp,
    "eng_multi_interp": engineering.run_multi_interp,
    "eng_max_volume": engineering.run_max_volume,
    "eng_area_integral": engineering.run_area_integral,
    "thiessen": thiessen.run_thiessen,
    "muskingum_xk": muskingum.run_muskingum_xk,
    "muskingum_route": muskingum.run_muskingum_route,
    "nash_nk": unit_hydrograph.run_nash_nk,
    "nash_duh": unit_hydrograph.run_nash_duh,
    "sherman_derive": unit_hydrograph.run_sherman_derive,
    "sherman_outlet": unit_hydrograph.run_sherman_outlet,
    "flood_curve_interp": flood.run_curve_interp,
    "flood_trial": flood.run_flood_trial,
    "flood_rk4": flood.run_flood_rk4,
}


def create_app():
    ensure_dirs()
    cleanup_runtime()
    app = Flask(__name__, static_folder="static", static_url_path="/static")
    app.config["MAX_CONTENT_LENGTH"] = 80 * 1024 * 1024
    app.json.ensure_ascii = False

    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/api/health")
    def health():
        return jsonify({"ok": True, "name": APP_NAME, "version": APP_VERSION})

    @app.get("/api/catalog")
    def catalog():
        return jsonify(catalog_payload())

    @app.get("/api/preview")
    def preview():
        token = request.args.get("token", "")
        try:
            path, meta = _resolve_token(token)
            return jsonify(_preview_file(path, meta))
        except Exception as exc:
            return jsonify({"ok": False, "error": exc_message(exc)}), 400

    @app.post("/api/upload")
    def upload():
        file = request.files.get("file")
        if not file or not file.filename:
            return jsonify({"ok": False, "error": "没有收到文件。"}), 400
        role = request.form.get("role", "")
        original_name = Path(file.filename).name
        suffix = Path(original_name).suffix.lower()
        if suffix not in (".csv", ".txt", ".zip"):
            return jsonify({"ok": False, "error": "目前支持 CSV、TXT、ZIP 文件。"}), 400
        upload_id = uuid.uuid4().hex
        folder = UPLOAD_DIR / upload_id
        folder.mkdir(parents=True, exist_ok=True)
        stored_name = safe_name(original_name)
        if not Path(stored_name).suffix:
            stored_name += suffix
        target = folder / stored_name
        file.save(target)
        kind = "shapefile" if suffix == ".zip" else "csv"
        UPLOADS[upload_id] = {"path": target, "name": original_name, "kind": kind, "role": role}
        token = f"upload:{upload_id}"
        try:
            preview_data = _preview_file(target, UPLOADS[upload_id])
            return jsonify({"ok": True, "token": token, "name": original_name, "kind": kind, **preview_data})
        except Exception as exc:
            return jsonify({"ok": False, "error": exc_message(exc), "token": token, "name": original_name, "kind": kind}), 400

    @app.post("/api/run")
    def run():
        payload = request.get_json(silent=True) or {}
        algorithm = get_algorithm(payload.get("algorithm"))
        if not algorithm:
            return jsonify({"ok": False, "error": "未知算法。"}), 404
        params = payload.get("params") or {}
        file_tokens = payload.get("files") or {}
        try:
            resolved = _resolve_run_files(algorithm, file_tokens)
        except Exception as exc:
            return jsonify({"ok": False, "error": exc_message(exc)}), 400

        runner = RUNNERS.get(algorithm["id"])
        if not runner:
            return jsonify({"ok": False, "error": "该算法还没有接入运行器。"}), 500

        ctx = RunContext(algorithm_id=algorithm["id"])
        started = time.perf_counter()
        with RUN_LOCK:
            try:
                ctx.section(algorithm["name"])
                ctx.log(algorithm.get("description", ""))
                result = runner(ctx, params, resolved)
                elapsed = time.perf_counter() - started
                return jsonify(
                    {
                        "ok": True,
                        "run_id": ctx.run_id,
                        "algorithm": algorithm["id"],
                        "duration": round(elapsed, 3),
                        "stdout": "\n".join(ctx.logs),
                        "summary": result.get("summary", []),
                        "tables": result.get("tables", []),
                        "images": result.get("images", []),
                        "visuals": ctx.visuals,
                    }
                )
            except Exception as exc:
                elapsed = time.perf_counter() - started
                ctx.log(f"运行失败：{exc}")
                return (
                    jsonify(
                        {
                            "ok": False,
                            "run_id": ctx.run_id,
                            "algorithm": algorithm["id"],
                            "duration": round(elapsed, 3),
                            "stdout": "\n".join(ctx.logs),
                            "summary": ctx.summary,
                            "tables": ctx.tables,
                            "images": ctx.images,
                            "error": exc_message(exc),
                        }
                    ),
                    400,
                )

    @app.get("/runtime/<path:subpath>")
    def runtime_files(subpath):
        return send_from_directory(RUNTIME_DIR, subpath, conditional=True)

    @app.after_request
    def no_cache_api(response):
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    return app


def _resolve_token(token):
    if not token:
        raise ValueError("未指定文件。")
    if token.startswith("sample:"):
        name = token.split(":", 1)[1]
        path = sample_path(name)
        return path, {"name": path.name, "kind": "shapefile" if path.suffix == ".zip" else "csv", "role": ""}
    if token.startswith("upload:"):
        upload_id = token.split(":", 1)[1]
        meta = UPLOADS.get(upload_id)
        if not meta:
            raise ValueError("上传文件已过期，请重新上传。")
        if not Path(meta["path"]).exists():
            raise ValueError("上传文件不存在，请重新上传。")
        return Path(meta["path"]), meta
    raise ValueError("非法的文件标识。")


def _preview_file(path: Path, meta=None):
    meta = meta or {}
    suffix = path.suffix.lower()
    if suffix in (".csv", ".txt"):
        df = read_csv(path)
        preview = dataframe_preview(df, 100)
        numeric_columns = [str(c) for c in df.columns if str(df[c].dtype) in ("int64", "float64", "Int64", "Float64")]
        return {
            "kind": "csv",
            "name": meta.get("name") or path.name,
            "columns": preview["columns"],
            "rows": preview["rows"],
            "total_rows": preview["total_rows"],
            "numeric_columns": numeric_columns,
        }
    if suffix == ".zip":
        if gpd is None:
            raise ValueError("预览 Shapefile 需要 geopandas，请安装 requirements.txt。")
        shp = locate_shapefile(path)
        gdf = gpd.read_file(shp)
        columns = [str(c) for c in gdf.columns if c != "geometry"]
        preview_df = gdf[columns].head(100) if columns else gdf[["geometry"]].head(0)
        preview = dataframe_preview(preview_df, 100)
        bounds = gdf.total_bounds.tolist() if len(gdf) else []
        return {
            "kind": "shapefile",
            "name": meta.get("name") or path.name,
            "columns": preview["columns"],
            "rows": preview["rows"],
            "total_rows": int(len(gdf)),
            "numeric_columns": [],
            "meta": {
                "geometry_type": str(gdf.geom_type.iloc[0]) if len(gdf) else "Unknown",
                "crs": str(gdf.crs) if gdf.crs else "Unknown",
                "bounds": [round(v, 6) for v in bounds] if bounds else [],
                "shp_name": shp.name,
            },
        }
    raise ValueError("不支持的文件类型。")


def _resolve_run_files(algorithm, tokens):
    resolved = {}
    for spec in algorithm.get("files", []):
        key = spec["key"]
        token = tokens.get(key)
        if not token and spec.get("required") and spec.get("sample"):
            token = f"sample:{spec['sample']}"
        if not token:
            if spec.get("required"):
                raise ValueError(f"缺少文件：{spec['label']}")
            resolved[key] = None
            continue
        path, _ = _resolve_token(token)
        resolved[key] = path
    return resolved
