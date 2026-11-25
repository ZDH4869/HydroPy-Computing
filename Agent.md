# HydroLab 维护指南

[中文 README](README.md) · [English README](README.en.md) · [本地启动提示词](一键启动部署提示词.md)

## 工作台边界

HydroLab 包含 3 门课程和 14 项算法。课程目录保存原始脚本，`workbench/` 的 Python 运行器提供每项算法的本地 Web 入口。界面维护应保留公式、单位、参数默认值、文件要求、API 以及既有 CSV 的含义。

## 启动

Windows 执行 `.\start_workbench.bat`，macOS / Linux 执行 `bash start_workbench.sh`。已有虚拟环境时，可显式安装依赖并启动。

```bash
python -m pip install -r requirements.txt
python workbench/start.py --no-browser
```

默认监听 `127.0.0.1:8765`。如需换端口，增加 `--port 8766`。GeoPandas 和 Shapely 用于泰森多边形，中文图例依赖本机可用的 CJK 字体。

## 维护入口

| 文件 | 职责 |
| --- | --- |
| `workbench/static/index.html` | 文档结构、课程导航、工作区、配置表单和原图查看对话框 |
| `workbench/static/styles.css` | 布局、响应式样式、交互状态和减少动效偏好 |
| `workbench/static/app.js` | 加载目录、算例状态、上传预览、请求运行、收起和展开结果明细 |
| `workbench/static/charts.js` | 将输入和输出表映射到 Canvas，处理量纲、图例、游标和积分窗口 |
| `workbench/static/simulation.js` | 组装回放与对照、绘制水位线和泰森控制区、管理回放控制 |
| `workbench/static/scenes/` | 流域总览、水文站、河网和水库剖面的示意背景 |
| `workbench/catalog.py` | 课程、算法、参数字段、默认值、示例文件 |
| `workbench/app.py` | Flask API、文件标识、并发控制和算法路由 |
| `workbench/common.py` | 数据读取、预览、运行上下文与文件输出 |
| `engineering.py`、`muskingum.py`、`unit_hydrograph.py`、`flood.py`、`thiessen.py` | 各课程 Python 运行器 |

上传与运行产物由应用写到 `workbench/runtime/`，该目录已忽略。当前后端在启动时会清理所属子目录中超过 48 小时的运行数据。需要长期保留的输出应下载到工作台目录之外。维护和验证可使用独立临时运行目录，保留用户已有的文件。

## API 与前端数据流

| 请求 | 用途 |
| --- | --- |
| `GET /api/health` | 查询服务状态 |
| `GET /api/catalog` | 读取 `courses` 和 `algorithms` |
| `GET /api/preview?token=…` | 读取 CSV 的列和预览行，或 Shapefile 的属性、几何类型和 CRS |
| `POST /api/upload` | 用 `multipart/form-data` 提交 `file` 与 `role` |
| `POST /api/run` | 传入 `{"algorithm": "...", "params": {...}, "files": {...}}` |
| `GET /runtime/runs/<id>/<file>` | 读取本次计算生成的 CSV 或图片 |

`files` 以配置清单中的文件键为键。值为 `sample:<name>` 或 `upload:<id>`，其中上传标识只在当前服务进程有效。运行返回 `ok`、`duration`、`stdout`、`summary`、`tables`、`images` 和 `visuals`，失败时可能包含 `error` 与部分产物。`visuals` 只携带计算过程中已有的时间、序列或投影几何，算法没有额外展示数据时返回空对象。

输入预览最多由 API 返回 100 行，表格在界面中显示前六行。结果数据由 `RunContext.save_csv()` 导出，曲线在预览截断时尝试读取同一结果的完整 CSV。读取失败时显示当前预览的范围，CSV 下载入口仍指向原始输出。

## 添加算法或结果视图

1. 扩展 `catalog.py` 和 `app.py` 的 `RUNNERS`，为字段设置可复现的默认值，并在 `workbench/samples/` 提供匹配的文件。
2. 在 Python 运行器内通过 `ctx.add_summary()`、`ctx.save_csv()`、`ctx.save_fig()` 和 `ctx.log()` 登记产物。绘图沿用现有无界面的 Matplotlib 后端。
3. 若结果表需要交互图，在 `charts.js` 的 `resultCharts()` 中明确横轴、列名和单位。不同量纲分开绘制，以原图保留空间分析与多面板图。
4. 需要额外的回放信息时，用 `ctx.add_visual_table()` 从已算得的序列登记，或用 `ctx.visuals` 记录可序列化的几何和时间。为对应算法在 `simulation.js` 配置对照和回放模式。水位线按该次算例的范围缩放。
5. 同步更新中英文 README 的算法列表和实际展示素材。当前展示文件位于 `docs/portfolio/`。

## 验收

- 请求 `/api/health` 和 `/api/catalog`，检查课程、算法和样例是否加载。
- 对目录中的 14 项算法逐项运行默认输入，核对 HTTP 状态、摘要、CSV 和图片。只改界面时，可对照改动前的摘要、结果表及 CSV 内容。
- 测试至少一例多曲线、一例三文件调洪、一例泰森地图。检查输入切换、图例、悬停或键盘数据点、对照、倍速、暂停、拖动、原图放大、日志展开和完整 CSV 下载。拖动 RK4 时间轴时核对水位线与当前时刻。点击泰森测站时核对控制区高亮和面积权重。
- 替换一个 CSV 后验证重新预览，恢复示例后核对默认输入和参数。验证运行中切换算法时，响应回到对应算例。
- 使用桌面和窄屏浏览器查看布局，确认表格有界、页面无横向溢出，检查控制台报错及 README 媒体链接。

## 数据格式

洪水过程常用 `年月日,时,Q` 或 `年,月,日,时,Q`，具体列以算法对应的示例为准。水库调洪使用入库过程、水位–库容、水位–下泄三份 CSV。泰森多边形的两个 ZIP 应包含 `.shp`、`.dbf`、`.shx`、`.prj` 配套文件，雨量站属性应包含配置中的站名和雨量字段。
