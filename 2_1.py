import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings("ignore")


def process_pregnancy_data(input_file="附件.xlsx", output_file="结果.xlsx"):
    """
    处理孕妇数据：
    1. 对每个孕妇的Y染色体浓度做三次样条插值
    2. 找出Y染色体浓度为0.04对应的检测孕周
    3. 用检测孕周预测孕妇BMI
    4. 保存结果到新文件
    """

    # 读取Excel文件
    print(f"正在读取文件: {input_file}")
    try:
        df = pd.read_excel(input_file)
        print(f"成功读取数据，共 {len(df)} 行")
    except Exception as e:
        print(f"读取文件出错: {e}")
        return

    df["检测孕周"] = (
        df["检测孕周"]
        .str.lower()                         # 转为小写
        .str.replace("w", "", regex=False)   # 去掉 'w'
        .str.split("+")                      # 按 "+" 拆分
        .apply(lambda x: int(x[0]) + int(x[1]) / 7 if len(x) > 1 else int(x[0]))
    )

    # 显示列名，确保列名正确
    print(f"数据列: {list(df.columns)}")

    # 检查必需的列是否存在
    required_columns = ["孕妇代码", "检测孕周", "Y染色体浓度", "孕妇BMI"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"缺少必需的列: {missing_columns}")
        return

    # 清理数据：移除空值
    df = df.dropna(subset=required_columns)
    print(f"清理后数据：{len(df)} 行")

    # 按孕妇代码分组
    grouped = df.groupby("孕妇代码")
    print(f"共有 {len(grouped)} 个孕妇")

    results = []
    skipped_count = 0
    processed_count = 0

    # 对每个孕妇进行处理
    for pregnant_code, group in grouped:
        # 跳过只有一条数据的孕妇
        if len(group) <= 1:
            skipped_count += 1
            continue

        try:
            # 按检测孕周排序
            group_sorted = group.sort_values("检测孕周")

            weeks = group_sorted["检测孕周"].values
            y_concentration = group_sorted["Y染色体浓度"].values
            bmi = group_sorted["孕妇BMI"].values[0]  # 假设同一孕妇的BMI相同

            # 检查数据是否适合插值
            if len(weeks) < 2:
                skipped_count += 1
                continue

            # 检查Y染色体浓度是否有变化范围包含0.04
            min_y = min(y_concentration)
            max_y = max(y_concentration)

            if not (min_y <= 0.04 <= max_y):
                # 如果0.04不在数据范围内，尝试外推（但要谨慎）
                print(
                    f"孕妇 {pregnant_code}: Y染色体浓度范围 [{min_y:.4f}, {max_y:.4f}] 不包含 0.04，尝试外推"
                )

            # 创建三次样条插值
            try:
                cs = CubicSpline(y_concentration, weeks)

                # 计算Y染色体浓度为0.04对应的检测孕周
                predicted_week = cs(0.04)

                # 存储结果
                results.append(
                    {
                        "孕妇代码": pregnant_code,
                        "原始BMI": bmi,
                        "Y染色体浓度0.04对应的检测孕周": predicted_week,
                        "数据点数量": len(group),
                    }
                )

                processed_count += 1

            except Exception as e:
                print(f"孕妇 {pregnant_code} 插值失败: {e}")
                skipped_count += 1
                continue

        except Exception as e:
            print(f"处理孕妇 {pregnant_code} 时出错: {e}")
            skipped_count += 1
            continue

    print(f"成功处理 {processed_count} 个孕妇，跳过 {skipped_count} 个孕妇")

    if not results:
        print("没有成功处理的数据")
        return

    # 转换为DataFrame
    result_df = pd.DataFrame(results)

    # 建立检测孕周与BMI的预测模型
    print("建立检测孕周预测BMI的模型...")

    # 使用所有可用数据建立模型
    model_data = df[["检测孕周", "孕妇BMI"]].dropna()

    if len(model_data) > 10:  # 确保有足够的数据建模
        X = model_data[["检测孕周"]]
        y = model_data["孕妇BMI"]

        # 创建多项式回归模型（2次）
        poly_model = Pipeline(
            [("poly", PolynomialFeatures(degree=2)), ("linear", LinearRegression())]
        )

        poly_model.fit(X, y)

        # 使用模型预测BMI
        predicted_weeks = result_df["Y染色体浓度0.04对应的检测孕周"].values.reshape(
            -1, 1
        )
        predicted_bmi = poly_model.predict(predicted_weeks)

        result_df["预测BMI"] = predicted_bmi

        # 计算模型得分
        score = poly_model.score(X, y)
        print(f"BMI预测模型 R² 得分: {score:.4f}")

    else:
        # 如果数据不足，使用简单平均值
        mean_bmi = df["孕妇BMI"].mean()
        result_df["预测BMI"] = mean_bmi
        print(f"数据不足建模，使用平均BMI: {mean_bmi:.2f}")

    # 添加一些统计信息
    result_df["BMI差异"] = result_df["预测BMI"] - result_df["原始BMI"]

    # 保存结果
    try:
        result_df.to_excel(output_file, index=False)
        print(f"结果已保存到: {output_file}")

        # 显示结果摘要
        print("\n结果摘要:")
        print(f"处理的孕妇数量: {len(result_df)}")
        print(f"平均预测孕周: {result_df['Y染色体浓度0.04对应的检测孕周'].mean():.2f}")
        print(f"平均原始BMI: {result_df['原始BMI'].mean():.2f}")
        print(f"平均预测BMI: {result_df['预测BMI'].mean():.2f}")

        print("\n前5个结果:")
        print(result_df.head().to_string(index=False))

    except Exception as e:
        print(f"保存文件出错: {e}")

    return result_df


# 更详细的数据分析函数
def analyze_data_quality(input_file="附件.xlsx"):
    """分析数据质量和分布"""

    df = pd.read_excel(input_file)

    print("=== 数据质量分析 ===")
    print(f"总行数: {len(df)}")
    print(f"总列数: {len(df.columns)}")
    print(f"列名: {list(df.columns)}")

    # 检查空值
    print("\n空值统计:")
    print(df.isnull().sum())

    # 按孕妇代码分组统计
    grouped = df.groupby("孕妇代码").size()
    print(f"\n孕妇数量: {len(grouped)}")
    print(f"单条数据的孕妇: {sum(grouped == 1)}")
    print(f"多条数据的孕妇: {sum(grouped > 1)}")
    print(f"最多数据条数: {grouped.max()}")
    print(f"平均每个孕妇数据条数: {grouped.mean():.2f}")

    # Y染色体浓度统计
    print(
        f"\nY染色体浓度范围: {df['Y染色体浓度'].min():.4f} - {df['Y染色体浓度'].max():.4f}"
    )
    print(
        f"包含0.04的孕妇数量: {sum(df.groupby('孕妇代码')['Y染色体浓度'].apply(lambda x: x.min() <= 0.04 <= x.max()))}"
    )

    return df


# 主函数
if __name__ == "__main__":
    # 首先分析数据质量
    print("第一步：分析数据质量")
    analyze_data_quality("附件.xlsx")

    print("\n" + "=" * 50)
    print("第二步：处理数据")

    # 处理数据
    result = process_pregnancy_data("附件.xlsx", "处理结果.xlsx")
