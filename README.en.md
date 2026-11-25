# HydroLab · Hydrology Workbench

[简体中文](README.md) · [Quick start](#quick-start) · [Implementation](#implementation) · [MIT license](LICENSE)

HydroLab is a local workbench for civil and hydraulic engineering coursework. Explore flood hydrographs, basin rainfall, and reservoir routing with bundled datasets, side-by-side comparisons, interactive playback, and CSV export.

![HydroLab workbench showing a bundled flood hydrograph](docs/portfolio/twin-overview.png)

## Demo

The recording follows a real session. It compares observed points with interpolated curves, inspects the computed Thiessen polygons by station, then replays the reservoir level and outflow in the RK4 example.

![A real HydroLab session showing interpolation, Thiessen polygons, and RK4 playback](docs/portfolio/twin-workbench.gif)

## Quick start

Use **Python 3.10–3.13**. The first dependency installation needs an internet connection. Flask serves the frontend and runs the calculations locally.

| Platform | Run from the repository root |
| --- | --- |
| Windows PowerShell | `.\start_workbench.bat`, or double-click the file |
| macOS / Linux | `bash start_workbench.sh` |

To start in an existing virtual environment, run the following.

```bash
python -m pip install -r requirements.txt
python workbench/start.py
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). Press `Ctrl+C` in the terminal to stop the server. Use `python workbench/start.py --port 8766` to change the port, or add `--no-browser` to skip opening a browser automatically.

## From input to results

The workbench interface is in Simplified Chinese.

1. Choose a course and then an algorithm. All 14 algorithms ship with runnable examples.
2. Use a file name in the configuration card or the input switches above the plot to inspect the data. Upload or drop your own CSV files. The Thiessen workflow takes separate Shapefile ZIPs for the basin boundary and rain gauges.
3. Set the parameters and select **运行计算** (Run), or press `Ctrl/⌘ + Enter`. Toggle a series in the legend. Hover over the plot or focus it and use the arrow keys to inspect values.
4. Use the controls below the scene to play, pause, scrub, reset, or change the playback speed. Where available, **对照** (Compare) places the two processes side by side and uses shared scales for comparable quantities.
5. Open **原始图** (Original figure) for the Python-generated analysis figure, or **结果数据与下载** (Result data and downloads) to inspect rows and export CSV. Each table preview shows up to six rows.

| Course | Methods |
| --- | --- |
| Engineering hydrology | Regular-interval linear interpolation, interpolation comparison, maximum flood volume by duration, integration check |
| Water resources informatics | Thiessen areal rainfall, Muskingum X/K calibration and downstream routing, Nash instantaneous and duration unit hydrographs, Sherman unit hydrograph derivation and outlet discharge |
| Introduction to flood control | Reservoir curve interpolation, trial-and-error routing, fourth-order Runge–Kutta (RK4) routing |

<details>
<summary>View screenshots of reservoir routing and areal rainfall</summary>

![RK4 routing with inflow and outflow hydrographs and summary metrics](docs/portfolio/twin-routing.png)

![Thiessen polygon rainfall analysis with a basin map and summary metrics](docs/portfolio/twin-thiessen.png)

</details>

## Implementation

Flask supplies the algorithm catalog, form fields, defaults, and bundled samples. A run sends parameters and file tokens to the corresponding Python runner, which returns a summary and links to its CSV and Matplotlib artifacts. The browser renders hydrographs and flood-volume comparisons using Canvas. Reservoir variables with different units have separate views. GeoPandas, Shapely, and SciPy compute the Thiessen polygons. The interactive result map uses the computed polygon geometry and projected gauge coordinates, and each station shows rainfall, control area, weight, and its weighted contribution.

```mermaid
flowchart LR
  A["Bundled or user-supplied files"] --> B["Flask preview and run API"]
  C["Algorithm catalog and defaults"] --> D["HydroLab workbench"]
  D -->|"Parameters and file tokens"| B
  B --> E["Python hydrology runners"]
  E --> F["Metrics, CSV, Matplotlib figures"]
  F --> D
```

Input plots use the API preview and label truncated data. Result plots use the full result CSV when available. Each export comes from the same Python run. The frontend handles presentation, and the Python scripts and runners handle the numerical work.

Playback follows the computed timestamps, flood-volume windows, or station records. The 3D backgrounds and reservoir section are illustrations. The displayed waterline is scaled to the current run. Interpolation playback uses the exact computation timestamps, including 30-minute steps, and the CSV retains its original whole-hour field.

### Repository layout

```text
.
├── workbench/
│   ├── static/                  # Scene images, Canvas charts, playback, and comparisons
│   ├── app.py                   # Flask API, previews, and algorithm dispatch
│   ├── catalog.py               # Courses, algorithms, fields, and sample configuration
│   ├── common.py                # Data I/O, Matplotlib theme, and run outputs
│   ├── engineering.py           # Interpolation and flood-volume integration
│   ├── muskingum.py             # Muskingum calibration and routing
│   ├── unit_hydrograph.py        # Nash and Sherman unit hydrographs
│   ├── flood.py                  # Reservoir curves and routing
│   ├── thiessen.py               # Thiessen areal rainfall
│   ├── samples/                 # Ready-to-run examples
│   └── runtime/                 # Local uploads and run outputs, ignored by Git
├── 工程水文课程-代码开发/         # Engineering hydrology coursework
├── 水利信息化课程-代码开发/       # Water resources informatics coursework
├── 防洪概论课程-代码开发/         # Flood control coursework
└── docs/portfolio/               # Screenshots and recording of the running workbench
```

The three coursework folders also contain standalone scripts and supporting material. Check file paths, fonts, and units before running a standalone script. For upload formats, start with the matching [bundled example](workbench/samples/).

## Maintenance and setup

- [Maintainer guide, API, and verification checklist (Chinese)](Agent.md)
- [Local setup prompt for a coding agent (Chinese)](一键启动部署提示词.md)

**Dehai Zhang (ZDH)** · Water Conservancy and Hydropower Engineering, Sichuan Agricultural University. HydroLab grew out of second- and third-year coursework. Released under the [MIT license](LICENSE).
