"""Shared utilities for the hydrology workbench algorithms."""
from __future__ import annotations

import io
import json
import math
import os
import re
import shutil
import tempfile
import traceback
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

CYAN = "#22d3ee"
BLUE = "#38bdf8"
ORANGE = "#fb923c"
GREEN = "#34d399"
VIOLET = "#a78bfa"
PINK = "#f472b6"
RED = "#f87171"
YELLOW = "#fbbf24"
MUTED = "#94a3b8"
TEXT = "#e2e8f0"
PANEL = "#0b1728"
BG = "#07111f"
GRID = "#1e293b"
BORDER = "#334155"

PALETTE = [CYAN, ORANGE, GREEN, VIOLET, PINK, YELLOW, BLUE, RED]

_BASE = Path(__file__).resolve().parent
SAMPLES_DIR = _BASE / "samples"
RUNTIME_DIR = _BASE / "runtime"
UPLOAD_DIR = RUNTIME_DIR / "uploads"
RUN_DIR = RUNTIME_DIR / "runs"

_FONT_CACHE = None


def ensure_dirs():
    for folder in (RUNTIME_DIR, UPLOAD_DIR, RUN_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def configure_matplotlib():
    """Configure a dark, workbench-consistent Matplotlib theme."""
    global _FONT_CACHE
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    if _FONT_CACHE is None:
        available = {f.name for f in font_manager.fontManager.ttflist}
        candidates = [
            "Microsoft YaHei",
            "Microsoft YaHei UI",
            "SimHei",
            "Noto Sans CJK SC",
            "Source Han Sans SC",
            "WenQuanYi Zen Hei",
            "Arial Unicode MS",
            "DejaVu Sans",
        ]
        _FONT_CACHE = next((name for name in candidates if name in available), "DejaVu Sans")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [_FONT_CACHE, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": BG,
            "axes.facecolor": PANEL,
            "savefig.facecolor": BG,
            "text.color": TEXT,
            "axes.labelcolor": "#cbd5e1",
            "axes.titlecolor": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.edgecolor": BORDER,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.alpha": 0.7,
            "grid.linewidth": 0.7,
            "figure.dpi": 120,
            "savefig.dpi": 160,
            "legend.facecolor": "#0f1b2d",
            "legend.edgecolor": BORDER,
            "legend.labelcolor": TEXT,
            "legend.framealpha": 0.9,
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    return plt


def read_csv(path) -> pd.DataFrame:
    """Read a CSV with common Chinese/Windows encodings and clean headers."""
    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"):
        try:
            df = pd.read_csv(path, encoding=encoding)
            df.columns = [str(col).replace("\ufeff", "").strip() for col in df.columns]
            return df
        except UnicodeDecodeError as exc:
            last_error = exc
        except pd.errors.EmptyDataError as exc:
            raise ValueError("CSV 文件为空或无法识别。") from exc
        except Exception as exc:
            last_error = exc
    raise ValueError(f"读取 CSV 失败：{last_error}")


def read_table(path, *, sep=None) -> pd.DataFrame:
    """Read CSV/TXT with a best-effort delimiter."""
    if sep is not None:
        df = pd.read_csv(path, sep=sep, encoding="utf-8-sig")
        df.columns = [str(c).strip() for c in df.columns]
        return df
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"):
        for delimiter in (None, ",", "\t", r"\s+"):
            try:
                df = pd.read_csv(
                    path,
                    sep=delimiter,
                    engine="python" if delimiter is None else "c",
                    encoding=encoding,
                )
                df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
                if len(df.columns) >= 1:
                    return df
            except Exception:
                continue
    return read_csv(path)


def clean_numeric(series):
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


def find_column(df: pd.DataFrame, candidates, *, contains=None, numeric=True):
    columns = list(df.columns)
    for name in candidates:
        if name in df.columns:
            return name
    normalized = {str(c).replace(" ", "").lower(): c for c in columns}
    for name in candidates:
        key = str(name).replace(" ", "").lower()
        if key in normalized:
            return normalized[key]
    if contains:
        for col in columns:
            text = str(col).lower()
            if all(piece.lower() in text for piece in contains):
                if not numeric or pd.api.types.is_numeric_dtype(df[col]):
                    return col
    if numeric:
        for col in columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                return col
    return None


def require_columns(df, columns, label="数据"):
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{label}缺少列：{', '.join(missing)}。实际列：{', '.join(map(str, df.columns))}")


def to_number(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_duration_text(text, default=None):
    if text is None or str(text).strip() == "":
        return default or []
    values = []
    for token in str(text).replace("，", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(float(token))
        except ValueError:
            values.append(eval_fraction(token))
    return values


def eval_fraction(token):
    try:
        if "/" in token:
            numerator, denominator = token.split("/", 1)
            return float(numerator) / float(denominator)
        return float(token)
    except Exception as exc:
        raise ValueError(f"无法解析时段：{token}") from exc


def format_duration(days: float) -> str:
    if abs(days * 24 - round(days * 24)) < 1e-9 and days < 1:
        return f"{round(days * 24)} 小时"
    if abs(days - round(days)) < 1e-9:
        return f"{round(days)} 天"
    return f"{days:g} 天"


def build_datetime_series(df: pd.DataFrame) -> pd.Series:
    """Build a datetime series from common Chinese date/hour column layouts."""
    if "datetime" in df.columns:
        return pd.to_datetime(df["datetime"])
    if "时间" in df.columns:
        return pd.to_datetime(df["时间"])
    # Prefer explicit year/month/day/hour columns (e.g. interpolation output).
    cols = {str(c).lower(): c for c in df.columns}
    if all(k in cols for k in ("年", "月", "日", "时")):
        return pd.to_datetime(
            dict(
                year=clean_numeric(df[cols["年"]]),
                month=clean_numeric(df[cols["月"]]),
                day=clean_numeric(df[cols["日"]]),
                hour=clean_numeric(df[cols["时"]]),
            ),
            errors="coerce",
        )
    date_col = find_column(df, ["年月日", "日期", "date", "Date"], numeric=False)
    hour_col = find_column(df, ["时", "小时", "hour", "Hour"], numeric=True)
    if date_col and hour_col:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        hours = clean_numeric(df[hour_col]).fillna(0)
        return dates + pd.to_timedelta(hours, unit="h")
    if date_col:
        return pd.to_datetime(df[date_col], errors="coerce")
    # Fallback: a numeric time column in hours
    time_col = find_column(df, ["时间t/h", "时间(h)", "小时", "time", "Time", "t"])
    if time_col:
        return pd.to_datetime(clean_numeric(df[time_col]), unit="h", origin="unix")
    raise ValueError("无法识别时间列。请提供「年月日/时」或「时间」列。")


def read_flood_series(df: pd.DataFrame, flow_candidates=None):
    """Return (datetime Series, flow ndarray, time label)."""
    flow_col = find_column(
        df,
        (flow_candidates or []) + ["Q", "流量", "流量(m3/s)", "流量(m³/s)", "Q/(m3/s-1)", "下泄流量"],
        contains=["流量"],
    )
    if flow_col is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            raise ValueError("未找到流量列。")
        flow_col = numeric_cols[-1]
    flow = clean_numeric(df[flow_col])
    if "时间t/h" in df.columns and "年月日" not in df.columns:
        hours = clean_numeric(df["时间t/h"])
        if hours.notna().all():
            # Relative hours are enough for integration; use a synthetic epoch.
            times = pd.to_datetime(hours, unit="h", origin="1994-01-01")
            return times, flow.to_numpy(dtype=float), "时间t/h"
    times = build_datetime_series(df)
    valid = times.notna() & flow.notna()
    if not valid.any():
        raise ValueError("时间或流量列没有有效数据。")
    return times[valid].reset_index(drop=True), flow[valid].to_numpy(dtype=float), str(flow_col)


def read_time_flow_generic(df: pd.DataFrame):
    """Return numeric x and y for a generic CSV (first numeric column = x)."""
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) < 2:
        converted = {c: clean_numeric(df[c]) for c in df.columns}
        numeric_cols = [c for c, series in converted.items() if series.notna().sum() >= 2]
        if len(numeric_cols) < 2:
            raise ValueError("数据中至少需要两列数值。")
        df = df.assign(**converted)
    x_col = numeric_cols[0]
    y_col = numeric_cols[1]
    valid = df[x_col].notna() & df[y_col].notna()
    return df.loc[valid, x_col].to_numpy(dtype=float), df.loc[valid, y_col].to_numpy(dtype=float), x_col, y_col


def dataframe_preview(df: pd.DataFrame, limit=200):
    frame = df.head(limit).copy()
    for col in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[col]):
            frame[col] = frame[col].dt.strftime("%Y-%m-%d %H:%M")
        elif pd.api.types.is_float_dtype(frame[col]):
            frame[col] = frame[col].map(lambda v: None if pd.isna(v) else round(float(v), 6))
    frame = frame.where(pd.notnull(frame), None)
    frame = frame.astype(object)
    frame = frame.where(pd.notnull(frame), None)
    return {
        "columns": [str(c) for c in frame.columns],
        "rows": frame.values.tolist(),
        "total_rows": int(len(df)),
    }


def json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, 6)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.strftime("%Y-%m-%d %H:%M")
    if isinstance(value, (np.ndarray,)):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    return value


def fmt(value, digits=3):
    try:
        value = float(value)
    except Exception:
        return str(value)
    if math.isnan(value) or math.isinf(value):
        return "—"
    if abs(value) >= 1000:
        return f"{value:,.{digits}f}"
    return f"{value:,.{digits}f}"


def safe_name(name: str) -> str:
    name = Path(str(name)).name
    name = re.sub(r"[^\w.\-\u4e00-\u9fff ]+", "_", name, flags=re.UNICODE).strip()
    return name or "data"


def sample_path(name: str) -> Path:
    path = (SAMPLES_DIR / safe_name(name)).resolve()
    if not str(path).startswith(str(SAMPLES_DIR.resolve())):
        raise ValueError("非法示例文件。")
    if not path.exists():
        raise FileNotFoundError(f"示例文件不存在：{name}")
    return path


@dataclass
class RunContext:
    algorithm_id: str
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    logs: list[str] = field(default_factory=list)
    tables: list[dict] = field(default_factory=list)
    images: list[dict] = field(default_factory=list)
    summary: list[dict] = field(default_factory=list)
    started_at: float = field(default_factory=lambda: datetime.now().timestamp())

    @property
    def run_dir(self) -> Path:
        path = RUN_DIR / self.run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def log(self, message=""):
        text = str(message)
        self.logs.append(text)
        print(text)

    def section(self, title):
        self.log("")
        self.log(f"▍{title}")

    def add_summary(self, label, value, unit=""):
        self.summary.append({"label": str(label), "value": value if isinstance(value, str) else json_safe(value), "unit": unit})

    def save_csv(self, df: pd.DataFrame, filename: str, *, title=None, limit=200):
        name = safe_name(filename)
        path = self.run_dir / name
        df.to_csv(path, index=False, encoding="utf-8-sig")
        preview = dataframe_preview(df, limit)
        table = {
            "title": title or Path(name).stem,
            "columns": preview["columns"],
            "rows": preview["rows"],
            "total_rows": preview["total_rows"],
            "csv_url": f"/runtime/runs/{self.run_id}/{name}",
            "csv_name": name,
        }
        self.tables.append(table)
        return table

    def save_table(self, title, columns, rows, total_rows=None, csv_url=None):
        self.tables.append(
            {
                "title": title,
                "columns": columns,
                "rows": rows,
                "total_rows": total_rows if total_rows is not None else len(rows),
                "csv_url": csv_url,
            }
        )

    def save_fig(self, fig, filename: str, *, title=None, close=True):
        import matplotlib.pyplot as plt

        name = safe_name(filename)
        path = self.run_dir / name
        try:
            fig.tight_layout()
        except Exception:
            pass
        fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
        if close:
            plt.close(fig)
        image = {
            "title": title or Path(name).stem,
            "url": f"/runtime/runs/{self.run_id}/{name}",
            "name": name,
        }
        self.images.append(image)
        return image

    def fail(self, error):
        self.log("")
        self.log(f"✖ {error}")
        raise error


def extract_shapefile_zip(zip_path):
    """Extract a zip and return the directory containing .shp files."""
    zip_path = Path(zip_path)
    target = RUNTIME_DIR / "shp" / f"{zip_path.stem}_{uuid.uuid4().hex[:8]}"
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            member_path = (target / member.filename).resolve()
            if not str(member_path).startswith(str(target.resolve())):
                raise ValueError("压缩包包含非法路径。")
            member_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(member_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
    return target


def locate_shapefile(source):
    """Return the first .shp inside a directory or zip."""
    source = Path(source)
    if source.is_file() and source.suffix.lower() == ".zip":
        source = extract_shapefile_zip(source)
    if source.is_file() and source.suffix.lower() == ".shp":
        return source
    if source.is_dir():
        candidates = list(source.rglob("*.shp"))
        if candidates:
            return candidates[0]
    raise FileNotFoundError("未找到 .shp 文件，请上传包含完整 Shapefile 的 zip。")


def cleanup_runtime(max_age_hours=48):
    ensure_dirs()
    now = datetime.now().timestamp()
    for base in (UPLOAD_DIR, RUN_DIR):
        for child in base.iterdir() if base.exists() else []:
            try:
                if child.is_dir() and now - child.stat().st_mtime > max_age_hours * 3600:
                    shutil.rmtree(child, ignore_errors=True)
            except Exception:
                pass


def exc_message(exc):
    text = str(exc).strip()
    if not text:
        text = exc.__class__.__name__
    return text
