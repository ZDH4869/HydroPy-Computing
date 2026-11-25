"""One-command launcher for the Hydrology Workbench.

Usage:
    python workbench/start.py
    python workbench/start.py --port 8765 --no-browser
"""
from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workbench.app import create_app  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="水文学计算工作台")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    app = create_app()
    url = f"http://{args.host}:{args.port}"
    print("=" * 68)
    print("  Hydrology Workbench · 水文学计算工作台")
    print("=" * 68)
    print(f"  本地地址: {url}")
    print("  数据只在本机处理；按 Ctrl+C 停止服务。")
    print("=" * 68)

    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
