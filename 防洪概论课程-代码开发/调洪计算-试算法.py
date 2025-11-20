import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm
import os

# ================================
# 用户参数设置区域
# ================================

# 输入文件路径
flood_process_file = r"E:\水电202303班\大三（上期）\课程报告或小组作业\防洪概论（调洪计算）\代码开发\原始曲线\3h入库流量过程线.csv"  # 入库流量过程线文件
storage_curve_file = r"E:\水电202303班\大三（上期）\课程报告或小组作业\防洪概论（调洪计算）\代码开发\曲线插值\插值水位-库容曲线_linear.csv"  # 水位-库容曲线文件
discharge_curve_file = r"E:\水电202303班\大三（上期）\课程报告或小组作业\防洪概论（调洪计算）\代码开发\曲线插值\插值水位-下泄流量曲线_linear.csv"  # 水位-下泄流量曲线文件
flood_encoding='utf-8' # 读取格式 文件编码 'gbk' 'utf-8' 'latin1'
storage_curve_encoding='utf-8'
discharge_curve_encoding='utf-8'
# 输出文件路径
output_file = "E:\水电202303班\大三（上期）\课程报告或小组作业\防洪概论（调洪计算）\代码开发\A1_试算法.csv"

# 第一行初始值
initial_avg_inflow = 0.0  # 时段平均入库流量第一行数值
initial_discharge = 173.9  # 下泄流量第一行数值
initial_avg_discharge = 0.0  # 时段平均下泄流量第一行数值
initial_delta_V = 0.0  # 时段内水库存水量变化第一行数值
initial_V = 6450  # 水库存水量第一行数值（示例值，请根据实际情况修改）
initial_Z = 38.0  # 水库水位第一行数值（示例值，请根据实际情况修改）

# 计算参数
time_interval = 3600 * 1  # 固定的时间隔值（小时） 第一列差值*秒*单位换算 与入库流量过程线的 时间间隔相同（插值间隔）
unit_conversion = 0.0001  # 单位换算值 (m³到万m³的转换，1万m³=10000m³，所以是1/10000=0.0001)

# 试算参数
V_tolerance = 3  # 水库存水量V绝对误差（万m³）
Z_search_min = 36.0  # 试算水位最小值
Z_search_max = 41.0  # 试算水位最大值
decimal_places = 1  # 试算取值小数位数


# ================================
# 数据读取和预处理
# ================================

def read_data():
    """读取所有输入数据"""
    try:
        # 读取入库洪水过程线
        flood_data = pd.read_csv(flood_process_file,encoding=flood_encoding)
        print(f"成功读取入库洪水过程线数据，共{len(flood_data)}行")

        # 读取水位-库容曲线
        storage_curve = pd.read_csv(storage_curve_file,encoding=storage_curve_encoding)
        print(f"成功读取水位-库容曲线数据，共{len(storage_curve)}行")

        # 读取水位-下泄流量曲线
        discharge_curve = pd.read_csv(discharge_curve_file,encoding=discharge_curve_encoding)
        print(f"成功读取水位-下泄流量曲线数据，共{len(discharge_curve)}行")

        return flood_data, storage_curve, discharge_curve
    except Exception as e:
        print(f"读取数据时出错: {e}")
        return None, None, None


def create_interpolation_functions(storage_curve, discharge_curve):
    """创建插值函数"""
    # 水位-库容插值函数
    Z_storage = storage_curve['水位Z/m'].values
    V_storage = storage_curve['库容V/万m3'].values
    storage_interp = interp1d(Z_storage, V_storage, kind='linear',
                              bounds_error=False, fill_value="extrapolate")

    # 水位-下泄流量插值函数
    Z_discharge = discharge_curve['水位Z/m'].values
    q_discharge = discharge_curve['下泄流量q/(m3·s)'].values
    discharge_interp = interp1d(Z_discharge, q_discharge, kind='linear',
                                bounds_error=False, fill_value="extrapolate")

    # 库容-水位反插值函数（用于第六步检验）
    V_Z_interp = interp1d(V_storage, Z_storage, kind='linear',
                          bounds_error=False, fill_value="extrapolate")

    return storage_interp, discharge_interp, V_Z_interp


# ================================
# 试算过程函数
# ================================

def trial_calculation(row_idx, prev_row, storage_interp, discharge_interp, V_Z_interp,
                      current_avg_inflow, time_diff):
    """
    进行试算过程
    """
    best_Z = None
    best_q = None
    best_avg_q = None
    best_delta_V = None
    best_V = None
    min_error = float('inf')

    # 第一步：在试算区间内遍历所有可能的水位值
    step = 10 ** (-decimal_places)
    Z_candidates = np.arange(Z_search_min, Z_search_max + step, step)
    Z_candidates = np.round(Z_candidates, decimal_places)

    for Z_candidate in tqdm(Z_candidates, desc=f"第{row_idx + 1}行试算", leave=False):
        # 第二步：通过水位插值得到库容和下泄流量
        try:
            V_candidate = float(storage_interp(Z_candidate))
            q_candidate = float(discharge_interp(Z_candidate))
        except:
            continue

        # 第三步：计算时段平均下泄流量
        avg_q_candidate = (prev_row['下泄流量q/(m³·s⁻¹)'] + q_candidate) / 2

        # 第四步：计算时段内水库存水量变化
        delta_V_candidate = (current_avg_inflow - avg_q_candidate) * time_diff * 3600 * unit_conversion

        # 第五步：计算当前水库存水量
        V_current = prev_row['水库存水量V/万m³'] + delta_V_candidate

        # 检验水库存水量（绝对误差）
        if abs(V_current - V_candidate) <= V_tolerance:
            # 第六步：用计算得到的水库存水量反推水位进行检验
            try:
                Z_check = float(V_Z_interp(V_current))
                # 检验水位是否在合理范围内
                if Z_search_min <= Z_check <= Z_search_max:
                    # 记录误差最小的解
                    current_error = abs(V_current - V_candidate)
                    if current_error < min_error:
                        min_error = current_error
                        best_Z = Z_candidate
                        best_q = q_candidate
                        best_avg_q = avg_q_candidate
                        best_delta_V = delta_V_candidate
                        best_V = V_current
            except:
                continue

    return best_Z, best_q, best_avg_q, best_delta_V, best_V, min_error


# ================================
# 主计算函数
# ================================

def calculate_flood_routing():
    """主计算函数"""
    # 读取数据
    flood_data, storage_curve, discharge_curve = read_data()
    if flood_data is None:
        return False

    # 创建插值函数
    storage_interp, discharge_interp, V_Z_interp = create_interpolation_functions(
        storage_curve, discharge_curve)

    # 初始化结果DataFrame
    results = pd.DataFrame({
        '时间t/h': flood_data['时间t/h'],
        '入库流量Q/(m³·s⁻¹)': flood_data['Q/(m3/s-1)'],
        '时段平均入库流量/(m³·s⁻¹)': [0.0] * len(flood_data),
        '下泄流量q/(m³·s⁻¹)': [0.0] * len(flood_data),
        '时段平均下泄流量/(m³·s⁻¹)': [0.0] * len(flood_data),
        '时段内水库存水量变化ΔV/万m³': [0.0] * len(flood_data),
        '水库存水量V/万m³': [0.0] * len(flood_data),
        '水库水位Z/m': [0.0] * len(flood_data)
    })

    # 设置第一行数值
    results.loc[0, '时段平均入库流量/(m³·s⁻¹)'] = initial_avg_inflow
    results.loc[0, '下泄流量q/(m³·s⁻¹)'] = initial_discharge
    results.loc[0, '时段平均下泄流量/(m³·s⁻¹)'] = initial_avg_discharge
    results.loc[0, '时段内水库存水量变化ΔV/万m³'] = initial_delta_V
    results.loc[0, '水库存水量V/万m³'] = initial_V
    results.loc[0, '水库水位Z/m'] = initial_Z

    # 从第二行开始计算
    print("开始进行调洪演算计算...")
    success_count = 0

    for i in tqdm(range(1, len(flood_data)), desc="总体进度"):
        # 计算时段平均入库流量
        if i == 1:
            # 第二行：第一行与第二行入库流量的平均值
            results.loc[i, '时段平均入库流量/(m³·s⁻¹)'] = (
                                                                  results.loc[i - 1, '入库流量Q/(m³·s⁻¹)'] +
                                                                  results.loc[i, '入库流量Q/(m³·s⁻¹)']
                                                          ) / 2
        else:
            # 第n行(n>=2)：第n-1行与第n行入库流量的平均值
            results.loc[i, '时段平均入库流量/(m³·s⁻¹)'] = (
                                                                  results.loc[i - 1, '入库流量Q/(m³·s⁻¹)'] +
                                                                  results.loc[i, '入库流量Q/(m³·s⁻¹)']
                                                          ) / 2

        # 获取前一行数据
        prev_row = results.iloc[i - 1]
        current_avg_inflow = results.loc[i, '时段平均入库流量/(m³·s⁻¹)']
        time_diff = results.loc[i, '时间t/h'] - results.loc[i - 1, '时间t/h']

        # 进行试算
        best_Z, best_q, best_avg_q, best_delta_V, best_V, min_error = trial_calculation(
            i, prev_row, storage_interp, discharge_interp, V_Z_interp,
            current_avg_inflow, time_diff
        )

        if best_Z is not None:
            results.loc[i, '下泄流量q/(m³·s⁻¹)'] = best_q
            results.loc[i, '时段平均下泄流量/(m³·s⁻¹)'] = best_avg_q
            results.loc[i, '时段内水库存水量变化ΔV/万m³'] = best_delta_V
            results.loc[i, '水库存水量V/万m³'] = best_V
            results.loc[i, '水库水位Z/m'] = best_Z
            success_count += 1
        else:
            print(f"\n警告：第{i + 1}行无法找到合适的解，使用近似值")
            # 如果找不到合适解，使用近似方法
            # 使用前一行值作为初始估计
            prev_Z = prev_row['水库水位Z/m']
            prev_q = prev_row['下泄流量q/(m³·s⁻¹)']

            # 计算近似的下泄流量和库容
            try:
                approx_q = float(discharge_interp(prev_Z))
                approx_V = float(storage_interp(prev_Z))
            except:
                approx_q = prev_q
                approx_V = prev_row['水库存水量V/万m³']

            # 计算其他值
            results.loc[i, '下泄流量q/(m³·s⁻¹)'] = approx_q
            results.loc[i, '时段平均下泄流量/(m³·s⁻¹)'] = (prev_q + approx_q) / 2
            results.loc[i, '时段内水库存水量变化ΔV/万m³'] = (
                                                                    current_avg_inflow - results.loc[
                                                                i, '时段平均下泄流量/(m³·s⁻¹)']
                                                            ) * time_diff * 3600 * unit_conversion
            results.loc[i, '水库存水量V/万m³'] = prev_row['水库存水量V/万m³'] + results.loc[
                i, '时段内水库存水量变化ΔV/万m³']

            # 通过库容反推水位
            try:
                results.loc[i, '水库水位Z/m'] = float(V_Z_interp(results.loc[i, '水库存水量V/万m³']))
            except:
                results.loc[i, '水库水位Z/m'] = prev_Z

    # 保存结果
    try:
        results.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n计算完成！")
        print(f"成功计算: {success_count}/{len(flood_data) - 1} 行")
        print(f"结果已保存到: {output_file}")
        print(f"结果文件包含{len(results)}行数据")

        # 显示结果统计
        print("\n结果统计:")
        print(f"最终水位: {results.iloc[-1]['水库水位Z/m']:.2f} m")
        print(f"最终库容: {results.iloc[-1]['水库存水量V/万m³']:.2f} 万m³")
        print(f"最大下泄流量: {results['下泄流量q/(m³·s⁻¹)'].max():.2f} m³/s")

        return True
    except Exception as e:
        print(f"保存结果时出错: {e}")
        return False


# ================================
# 参数验证函数
# ================================

def validate_parameters():
    """验证输入参数"""
    errors = []

    # 检查文件是否存在
    if not os.path.exists(flood_process_file):
        errors.append(f"入库洪水过程线文件不存在: {flood_process_file}")
    if not os.path.exists(storage_curve_file):
        errors.append(f"水位-库容曲线文件不存在: {storage_curve_file}")
    if not os.path.exists(discharge_curve_file):
        errors.append(f"水位-下泄流量曲线文件不存在: {discharge_curve_file}")

    # 检查参数范围
    if Z_search_min >= Z_search_max:
        errors.append("试算水位最小值必须小于最大值")

    if time_interval <= 0:
        errors.append("时间隔值必须大于0")

    if unit_conversion <= 0:
        errors.append("单位换算值必须大于0")

    if V_tolerance <= 0:
        errors.append("水库存水量误差必须大于0")

    if decimal_places < 0 or decimal_places > 6:
        errors.append("试算小数位数应在0-6之间")

    return errors


# ================================
# 主程序
# ================================

if __name__ == "__main__":
    print("水库调洪演算程序开始运行...")
    print("=" * 50)

    # 验证参数
    validation_errors = validate_parameters()
    if validation_errors:
        print("参数验证错误:")
        for error in validation_errors:
            print(f"  - {error}")
        print("请修改参数后重新运行程序。")
    else:
        print("参数验证通过")
        success = calculate_flood_routing()

        if success:
            print("程序运行成功！")
        else:
            print("程序运行失败，请检查输入参数和文件路径。")

    print("=" * 50)