# Agent.md · HydroLab / 水文计算工作台开发与部署指南

本文件面向 AI Agent 或后续维护者，说明项目结构、启动方式、接口约定、新增算法的方法和验收流程。

## 1. 项目定位

`水利水文计算` 是一个高校水利水电工程课程实验代码仓库：

- `工程水文课程-代码开发/`
- `水利信息化课程-代码开发/`
- `防洪概论课程-代码开发/`

原有 Python 脚本可以直接命令行运行，但大多包含开发机绝对路径、硬编码数据或需要手工改参数。`workbench/` 在这些算法之上提供了一个**本地 Web 操作台**：

- 按课程 / 算法分类浏览；
- 三栏布局：参数配置 → 原始数据 → 运行结果；
- 上传 CSV / Shapefile zip，填参数，一键运行；
- 右侧展示命令行日志、结果图片、结果表格和 CSV 下载；
- 所有计算在本机 Python 进程中完成，不上传云端。

## 2. 快速启动

### Windows

```powershell
cd <repo>
.\start_workbench.bat
```

或：

```powershell
python -m pip install -r requirements.txt
python workbench\start.py
```

### macOS / Linux

```bash
cd <repo>
./start_workbench.sh
```

默认地址：`http://127.0.0.1:8765`

可选参数：

```powershell
python workbench/start.py --port 8766
python workbench/start.py --no-browser
```

## 3. 目录结构

```text
workbench/
├── app.py                 # Flask 应用和 API
├── start.py               # 一键启动入口
├── catalog.py             # 课程、算法、字段、默认示例文件目录
├── common.py              # 数据读取、Matplotlib 主题、运行上下文
├── engineering.py         # 工程水文算法
├── muskingum.py           # 马斯京根算法
├── unit_hydrograph.py     # Nash / Sherman 单位线
├── flood.py               # 曲线插值与水库调洪
├── thiessen.py            # 泰森多边形
├── samples/               # 内置示例 CSV / Shapefile zip
├── static/
│   ├── index.html         # 单页工作台
│   ├── styles.css         # 深色玻璃 UI
│   └── app.js             # 前端状态与交互
└── runtime/               # 运行生成物（已 gitignore）
    ├── uploads/
    ├── runs/
    └── shp/
```

## 4. API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/health` | 健康检查 |
| GET | `/api/catalog` | 课程、算法、表单字段、默认文件 |
| GET | `/api/preview?token=...` | CSV / Shapefile zip 预览 |
| POST | `/api/upload` | `multipart/form-data`，字段 `file`、`role` |
| POST | `/api/run` | JSON：`{algorithm, params, files}` |
| GET | `/runtime/runs/<id>/<file>` | 下载/查看结果文件 |

`files` 的值是文件 token：

- `sample:<name>`：`workbench/samples/` 下的内置示例；
- `upload:<id>`：本次进程内上传的文件（重启后失效）。

`/api/run` 返回：

```json
{
  "ok": true,
  "duration": 0.42,
  "stdout": "命令行日志",
  "summary": [{"label": "最大流量", "value": 2363, "unit": "m³/s"}],
  "tables": [{"title": "结果表", "columns": [], "rows": [], "csv_url": "/runtime/..."}],
  "images": [{"title": "过程线", "url": "/runtime/..."}]
}
```

## 5. 算法覆盖

当前 14 个算法：

### 工程水文学

- 等间隔线性插值
- 多种插值对比
- 时段最大洪量
- 曲线面积积分校验

### 水利信息化

- 泰森多边形面雨量
- 马斯京根参数率定 X/K
- 马斯京根下游流量演算
- Nash 瞬时单位线 N/K
- Nash 时段单位线
- Sherman 单位线推求
- Sherman 出口流量

### 防洪概论

- 水库曲线插值（Z-V / Z-q / 通用）
- 调洪演算 · 试算法
- 调洪演算 · RK4

## 6. 新增算法

1. 在 `workbench/` 下新增模块或复用现有模块，实现：

```python
def run_my_algorithm(ctx, params, files):
    # ctx.log("...")
    # df = read_csv(files["input_csv"])
    # ctx.save_csv(df, "结果.csv", title="结果")
    # ctx.save_fig(fig, "结果图.png", title="结果图")
    return {"summary": ctx.summary, "tables": ctx.tables, "images": ctx.images}
```

2. 在 `catalog.py` 的 `ALGORITHMS` 中增加条目：
   - `id`、`course`、`name`、`short`、`description`；
   - `fields`：`number` / `select` / `text` / `textarea` / `column`；
   - `files`：`key`、`label`、`accept`、`required`、`sample`；
   - 可以标注 `advanced: true` 将字段折叠到高级参数。

3. 在 `app.py` 的 `RUNNERS` 字典注册 `id` → 函数。

4. 把内置示例数据放入 `workbench/samples/`，并设置 `sample` 字段。

5. 运行 `python workbench/start.py --no-browser`，用浏览器点选新算法运行一次。

## 7. 数据格式约定

- CSV 支持 UTF-8 / UTF-8 BOM / GBK / GB18030 / latin1；
- 洪水过程支持：
  - `年月日` + `时` + `Q`；
  - `年` + `月` + `日` + `时` + `Q`；
  - `时间t/h` + `Q/(m3/s-1)`；
  - 通用时间列 + 流量列。
- 泰森多边形上传 zip，zip 内需包含完整 `.shp/.dbf/.shx/.prj`。
- 水库调洪三份 CSV：入库流量、水位-库容、水位-下泄流量。

## 8. 视觉与绘图约定

- 前端：深色玻璃 + 青色/紫色/橙色高亮，三栏布局；
- Matplotlib：统一使用 `common.configure_matplotlib()`，深色背景、主题色和微软雅黑；
- 图片保存到 `ctx.run_dir`，通过 `ctx.save_fig()` 自动登记；
- 不要使用 `plt.show()`；始终 `Agg` 后端。

## 9. 验收清单

```bash
python -c "from workbench.app import create_app; create_app(); print('ok')"
python workbench/start.py --no-browser
```

打开 `http://127.0.0.1:8765/`，确认：

- [ ] 顶部三个课程可切换；
- [ ] 每个课程的算法卡片可点击；
- [ ] 三栏布局完整：参数 / 原始数据 / 运行结果；
- [ ] 默认示例数据能预览；
- [ ] 至少运行一个工程水文、一个水利信息化、一个防洪算法；
- [ ] 结果区同时出现命令行日志、图片、数据表；
- [ ] 图片可以点击放大；
- [ ] 刷新页面后一切正常。

## 10. 常见问题

| 问题 | 处理 |
| --- | --- |
| `ModuleNotFoundError: flask` | `python -m pip install -r requirements.txt` |
| `geopandas` 导入失败 | 安装 `geopandas` / `shapely`，Windows 推荐 conda-forge 或 pip 新版 |
| 端口被占用 | `python workbench/start.py --port 8766` |
| 中文图片方块 | 安装微软雅黑/SimHei，或替换 `common.py` 中字体候选 |
| 上传 zip 预览失败 | 确认 zip 内包含 `.shp/.dbf/.shx/.prj`，不要把 `.shp` 单独上传 |
| 运行后无图片 | 查看命令行输出，可能是数据缺少列或 Matplotlib 绘图异常 |
| 浏览器打开空白 | 确认 Flask 服务仍在运行，访问 `/api/health` |

## 11. 约束

- 不修改用户原有课程脚本的算法含义；
- 不把运行结果、用户上传文件提交到 Git；
- 工作台只监听 `127.0.0.1`，不要暴露到公网；
- 新增算法必须提供默认示例，保证用户点开即可运行。
