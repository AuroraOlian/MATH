import pandas as pd
import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings("ignore")


def process_pregnancy_data_improved(input_file="附件.xlsx", output_file="结果.xlsx"):
    """
    改进的孕妇数据处理：
    1. 对于相同"孕妇代码"和"检测抽血次数"的数据，只保留最后一条
    2. 使用多种插值方法处理Y染色体浓度数据
    3. 找出Y染色体浓度为0.04对应的检测孕周
    4. 用检测孕周预测孕妇BMI
    5. 保存结果到新文件
    """

    # 读取Excel文件
    print(f"正在读取文件: {input_file}")
    try:
        df = pd.read_excel(input_file)
        print(f"成功读取数据，共 {len(df)} 行")
    except Exception as e:
        print(f"读取文件出错: {e}")
        return

    # 显示列名，确保列名正确
    print(f"数据列: {list(df.columns)}")

    # 检查必需的列是否存在
    required_columns = [
        "孕妇代码",
        "检测孕周",
        "Y染色体浓度",
        "孕妇BMI",
        "检测抽血次数",
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"缺少必需的列: {missing_columns}")
        return

    # 清理数据：移除空值
    original_count = len(df)
    df = df.dropna(subset=required_columns)
    print(f"移除空值后数据：{len(df)} 行 (移除了 {original_count - len(df)} 行)")

    # 处理检测孕周格式
    print("正在处理检测孕周格式...")
    df["检测孕周"] = (
        df["检测孕周"]
        .astype(str)
        .str.lower()
        .str.replace("w", "", regex=False)
        .str.split("+")
        .apply(
            lambda x: (
                int(x[0]) + int(x[1]) / 7 if len(x) > 1 and x[1] != "" else int(x[0])
            )
        )
    )

    # 数据去重：对于相同"孕妇代码"和"检测抽血次数"的数据，只保留最后一条
    print("正在进行数据去重...")
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["孕妇代码", "检测抽血次数"], keep="last")
    after_dedup = len(df)
    print(
        f"去重后数据：{after_dedup} 行 (移除了 {before_dedup - after_dedup} 条重复数据)"
    )

    # 按孕妇代码分组
    grouped = df.groupby("孕妇代码")
    print(f"共有 {len(grouped)} 个孕妇")

    results = []
    skipped_count = 0
    processed_count = 0
    method_stats = {"线性插值": 0, "PCHIP插值": 0, "多项式拟合": 0, "平均值估计": 0}

    def interpolate_y_concentration(weeks, y_concentration, target_y=0.04):
        """
        使用多种方法进行插值，选择最适合的方法
        """
        n_points = len(weeks)

        # 检查数据范围是否包含目标值
        min_y = min(y_concentration)
        max_y = max(y_concentration)

        # 如果目标值不在范围内，使用外推
        if not (min_y <= target_y <= max_y):
            print(
                f"    警告：目标值 {target_y} 不在数据范围 [{min_y:.4f}, {max_y:.4f}] 内"
            )

        try:
            # 方法1：PCHIP插值（保形插值，保证单调性）
            # if n_points >= 3:
            #     # 检查数据是否大致单调
            #     is_increasing = (
            #         np.sum(np.diff(y_concentration) > 0) >= len(y_concentration) - 2
            #     )
            #     is_decreasing = (
            #         np.sum(np.diff(y_concentration) < 0) >= len(y_concentration) - 2
            #     )

            #     if is_increasing or is_decreasing:
            #         try:
            #             pchip = PchipInterpolator(y_concentration, weeks)
            #             predicted_week = pchip(target_y)
            #             return predicted_week, "PCHIP插值"
            #         except:
            #             pass

            # 方法2：线性插值（最稳定）
            if n_points >= 2:
                try:
                    predicted_week = np.interp(target_y, y_concentration, weeks)
                    return predicted_week, "线性插值"
                except:
                    pass

            # 方法3：多项式拟合
            if n_points >= 3:
                try:
                    # 使用二次多项式拟合
                    poly_features = PolynomialFeatures(degree=min(2, n_points - 1))
                    y_poly = poly_features.fit_transform(y_concentration.reshape(-1, 1))
                    poly_model = LinearRegression()
                    poly_model.fit(y_poly, weeks)

                    target_poly = poly_features.transform([[target_y]])
                    predicted_week = poly_model.predict(target_poly)[0]
                    return predicted_week, "多项式拟合"
                except:
                    pass

            # 方法4：简单平均值估计（最后的备选）
            mean_week = np.mean(weeks)
            return mean_week, "平均值估计"

        except Exception as e:
            print(f"    插值失败: {e}")
            return None, "失败"

    # 对每个孕妇进行处理
    for pregnant_code, group in grouped:
        # 跳过只有一条数据的孕妇
        if len(group) <= 1:
            skipped_count += 1
            print(f"跳过孕妇 {pregnant_code}：数据点不足（只有 {len(group)} 条数据）")
            continue

        try:
            # 按检测孕周排序
            group_sorted = group.sort_values("检测孕周")

            weeks = group_sorted["检测孕周"].values
            y_concentration = group_sorted["Y染色体浓度"].values
            bmi = group_sorted["孕妇BMI"].values[0]

            # 检查数据质量
            if len(weeks) < 2:
                skipped_count += 1
                continue

            # 移除重复的Y染色体浓度值
            unique_indices = np.unique(y_concentration, return_index=True)[1]
            if len(unique_indices) < len(y_concentration):
                print(f"孕妇 {pregnant_code}: 移除重复的Y染色体浓度值")
                weeks = weeks[unique_indices]
                y_concentration = y_concentration[unique_indices]

            print(f"处理孕妇 {pregnant_code}: {len(weeks)} 个数据点")

            # 使用改进的插值方法
            predicted_week, method_used = interpolate_y_concentration(
                weeks, y_concentration, 0.04
            )

            if predicted_week is not None:
                method_stats[method_used] += 1

                # 计算数据质量指标
                y_range = max(y_concentration) - min(y_concentration)
                week_range = max(weeks) - min(weeks)

                results.append(
                    {
                        "孕妇代码": pregnant_code,
                        "原始BMI": bmi,
                        "Y染色体浓度0.04对应的检测孕周": predicted_week,
                        "使用的插值方法": method_used,
                        "数据点数量": len(group),
                        "Y染色体浓度范围": f"[{min(y_concentration):.4f}, {max(y_concentration):.4f}]",
                        "Y染色体变化幅度": y_range,
                        "孕周变化幅度": week_range,
                        "抽血次数范围": f"{group['检测抽血次数'].min()}-{group['检测抽血次数'].max()}",
                    }
                )

                processed_count += 1
                print(f"    成功：使用{method_used}，预测孕周 {predicted_week:.2f}")
            else:
                skipped_count += 1
                print(f"    失败：无法进行插值")

        except Exception as e:
            print(f"处理孕妇 {pregnant_code} 时出错: {e}")
            skipped_count += 1
            continue

    print(f"\n处理结果统计:")
    print(f"成功处理: {processed_count} 个孕妇")
    print(f"跳过: {skipped_count} 个孕妇")
    print(f"\n插值方法使用统计:")
    for method, count in method_stats.items():
        if count > 0:
            print(f"  {method}: {count} 次 ({count/processed_count*100:.1f}%)")

    if not results:
        print("没有成功处理的数据")
        return

    # 转换为DataFrame
    result_df = pd.DataFrame(results)

    # 建立检测孕周与BMI的预测模型
    print("\n建立检测孕周预测BMI的模型...")
    model_data = df[["检测孕周", "孕妇BMI"]].dropna()

    if len(model_data) > 10:
        X = model_data[["检测孕周"]]
        y = model_data["孕妇BMI"]

        # 使用较低次数的多项式回归，避免过拟合
        poly_model = Pipeline(
            [("poly", PolynomialFeatures(degree=2)), ("linear", LinearRegression())]
        )

        poly_model.fit(X, y)
        predicted_weeks = result_df["Y染色体浓度0.04对应的检测孕周"].values.reshape(
            -1, 1
        )
        predicted_bmi = poly_model.predict(predicted_weeks)
        result_df["预测BMI"] = predicted_bmi

        score = poly_model.score(X, y)
        print(f"BMI预测模型 R² 得分: {score:.4f}")
    else:
        mean_bmi = df["孕妇BMI"].mean()
        result_df["预测BMI"] = mean_bmi
        print(f"数据不足建模，使用平均BMI: {mean_bmi:.2f}")

    # 添加统计信息
    result_df["BMI差异"] = result_df["预测BMI"] - result_df["原始BMI"]

    # 数据质量评估
    print(f"\n数据质量评估:")
    print(f"平均Y染色体变化幅度: {result_df['Y染色体变化幅度'].mean():.4f}")
    print(f"平均孕周变化幅度: {result_df['孕周变化幅度'].mean():.2f}")
    reliable_data = result_df[result_df["数据点数量"] >= 3]
    print(
        f"数据点≥3的可靠样本: {len(reliable_data)} / {len(result_df)} ({len(reliable_data)/len(result_df)*100:.1f}%)"
    )

    # 保存结果
    try:
        result_df.to_excel(output_file, index=False)
        print(f"\n结果已保存到: {output_file}")

        # 显示结果摘要
        print("\n结果摘要:")
        print(f"处理的孕妇数量: {len(result_df)}")
        print(f"平均预测孕周: {result_df['Y染色体浓度0.04对应的检测孕周'].mean():.2f}")
        print(f"平均原始BMI: {result_df['原始BMI'].mean():.2f}")
        print(f"平均预测BMI: {result_df['预测BMI'].mean():.2f}")
        print(f"平均BMI差异: {result_df['BMI差异'].mean():.2f}")

        print("\n前5个结果:")
        display_columns = [
            "孕妇代码",
            "Y染色体浓度0.04对应的检测孕周",
            "使用的插值方法",
            "原始BMI",
            "预测BMI",
        ]
        print(result_df[display_columns].head().to_string(index=False))

    except Exception as e:
        print(f"保存文件出错: {e}")

    return result_df


# 主函数
if __name__ == "__main__":
    # 使用改进的方法处理数据
    result = process_pregnancy_data_improved("附件.xlsx", "改进处理结果.xlsx")
