# HydroLab · Hydrology Workbench

[简体中文](README.md) · [Quick start](#quick-start) · [Implementation](#implementation) · [MIT license](LICENSE)

HydroLab is a local workbench for civil and hydraulic engineering coursework. Explore flood hydrographs, basin rainfall, and reservoir routing with bundled datasets, side-by-side comparisons and CSV export. Desktop controls, charts, and metrics fit in one viewport over the course background. On narrow screens, tabs switch between configuration, charts, and metrics.

![HydroLab workbench showing a bundled flood hydrograph](docs/portfolio/backplate-overview.png)

## Demo

The recording follows a real session. It compares observed points with interpolated curves, inspects the computed Thiessen polygons by station, then inspects the inflow, outflow, and reservoir level in the RK4 example.

![A real HydroLab session showing interpolation, Thiessen polygons, and RK4 routing](docs/portfolio/backplate-workbench.gif)

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
4. Where available, **对照** (Compare) places related processes side by side. Each plot scales to its own data and plot area, while series overlaid in a single plot share axes. **洪量区间** (Flood-volume interval) highlights the computed maximum-volume window for the selected duration. Reservoir routing also has separate level and storage views.
5. Use **原始图** (Original figure) for the Python-generated analysis figure. Input records, result data and downloads, and calculation logs open in dialogs. Each table preview shows up to six rows, and the full result is available as CSV.
6. Switch between algorithms or courses and return to your previous workspace. Open **结果对比** (Compare runs) in the header to choose any two successful runs, including different parameter sets for the same algorithm. Each chart scales independently, and Thiessen runs can be inspected on their station maps.

Comparison history is stored locally for the current browser and workbench address. Charts, maps, metrics, and run parameters remain available after a refresh. Download the original CSVs and figures if you need to keep those files.

| Course | Methods |
| --- | --- |
| Engineering hydrology | Regular-interval linear interpolation, interpolation comparison, maximum flood volume by duration, integration check |
| Water resources informatics | Thiessen areal rainfall, Muskingum X/K calibration and downstream routing, Nash instantaneous and duration unit hydrographs, Sherman unit hydrograph derivation and outlet discharge |
| Introduction to flood control | Reservoir curve interpolation, trial-and-error routing, fourth-order Runge–Kutta (RK4) routing |

<details>
<summary>View screenshots of reservoir routing and areal rainfall</summary>

![RK4 routing with inflow and outflow hydrographs and summary metrics](docs/portfolio/backplate-routing.png)

![Thiessen polygon rainfall analysis with a basin map and summary metrics](docs/portfolio/backplate-thiessen.png)

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

The page backgrounds illustrate a river gauge, a mountain drainage network, and a reservoir. Charts, flood-volume intervals, and Thiessen polygons use the current run's results. Interpolation charts use the exact computation timestamps, including 30-minute steps, and the CSV retains its original whole-hour field.

### Repository layout

```text
.
├── workbench/
│   ├── static/                  # Full-page backgrounds, Canvas charts, and comparisons
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
