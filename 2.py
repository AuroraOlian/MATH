import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
import time
import warnings
from typing import List, Tuple, Dict, Any
import logging

plt.rcParams["font.sans-serif"] = ["SimHei", "SimSong"]  # 支持中文

# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class NIFTOptimizationDebugger:
    """NIPT BMI分组优化调试器"""

    def __init__(self, data: pd.DataFrame, debug_mode: bool = True):
        self.data = data
        self.debug_mode = debug_mode
        self.optimization_history = []
        self.function_calls = 0
        self.best_solutions = []

        # 优化参数
        self.bounds = [
            (18.0, 25.0),  # b_0
            (24.0, 30.0),  # b_1
            (29.0, 35.0),  # b_2
            (34.0, 40.0),  # b_3
            (39.0, 45.0),  # b_4
            (44.0, 50.0),  # b_5
            (10.0, 24.0),  # t_0
            (10.0, 24.0),  # t_1
            (10.0, 24.0),  # t_2
            (10.0, 24.0),  # t_3
            (10.0, 24.0),  # t_4
        ]

        # 风险函数权重
        self.w_accuracy = 0.65
        self.w_timing = 0.35

    def calculate_accuracy_risk(self, y_concentration: float) -> float:
        """计算准确性风险"""
        threshold = 4.0

        if y_concentration >= threshold:
            # 达标情况：风险很低但不为零
            return 0.1 * np.exp(-(y_concentration - threshold))
        else:
            # 未达标情况：使用Sigmoid函数
            deficit = threshold - y_concentration
            risk = 1 / (1 + np.exp(-2 * deficit))
            return risk

    def calculate_timing_risk(self, detection_time: float) -> float:
        """计算时间延迟风险"""
        if detection_time <= 12:
            return 0.1
        elif detection_time <= 16:
            return 0.1 + 0.15 * (detection_time - 12) / 4
        elif detection_time <= 22:
            return 0.25 + 0.4 * (detection_time - 16) / 6
        elif detection_time <= 27:
            return 0.65 + 0.25 * (detection_time - 22) / 5
        else:
            return 0.9 + 0.1 * min((detection_time - 27) / 5, 1.0)

    def predict_y_concentration(
        self, bmi: float, detection_time: float, patient_data: Dict
    ) -> float:
        """预测Y染色体浓度（简化模型）"""
        # 基础浓度随时间增长
        base_concentration = max(0, detection_time - 8) * 0.8

        # BMI效应：BMI越高，浓度增长越慢
        bmi_effect = -0.1 * (bmi - 25)

        # 个体差异（可以基于其他特征）
        individual_factor = patient_data.get(
            "individual_factor", np.random.normal(0, 0.5)
        )

        predicted = base_concentration + bmi_effect + individual_factor
        return max(0, predicted)

    def comprehensive_risk_assessment(
        self, patient_data: Dict, detection_time: float
    ) -> float:
        """综合风险评估"""
        # 预测Y染色体浓度
        predicted_conc = self.predict_y_concentration(
            patient_data["BMI"], detection_time, patient_data
        )

        # 计算两类风险
        accuracy_risk = self.calculate_accuracy_risk(predicted_conc)
        timing_risk = self.calculate_timing_risk(detection_time)

        # 总风险
        total_risk = self.w_accuracy * accuracy_risk + self.w_timing * timing_risk

        return total_risk

    def assign_group(self, bmi: float, boundaries: List[float]) -> int:
        """根据BMI分配组别"""
        for i in range(len(boundaries) - 1):
            if boundaries[i] <= bmi < boundaries[i + 1]:
                return i
        return len(boundaries) - 2  # 最后一组

    def objective_function(self, params: np.ndarray) -> float:
        """目标函数：最小化总风险"""
        self.function_calls += 1

        try:
            # 解析参数
            boundaries = params[:6].tolist()
            time_points = params[6:11].tolist()

            # 检查边界约束
            if not self._check_boundary_constraints(boundaries):
                return 1e6  # 大的惩罚值

            total_risk = 0
            group_risks = [0] * 5
            group_counts = [0] * 5

            # 计算每个患者的风险
            for idx, row in self.data.iterrows():
                patient_data = {
                    "BMI": row["BMI"],
                    "individual_factor": np.random.normal(0, 0.1),  # 模拟个体差异
                }

                # 确定组别
                group_idx = self.assign_group(patient_data["BMI"], boundaries)
                detection_time = time_points[group_idx]

                # 计算风险
                risk = self.comprehensive_risk_assessment(patient_data, detection_time)
                total_risk += risk
                group_risks[group_idx] += risk
                group_counts[group_idx] += 1

            # 平均风险
            avg_risk = total_risk / len(self.data)

            # 组间平衡惩罚
            balance_penalty = self._calculate_balance_penalty(group_counts)
            final_risk = avg_risk + balance_penalty

            # 记录调试信息
            if self.debug_mode and self.function_calls % 100 == 0:
                self._log_optimization_step(
                    params, final_risk, group_counts, group_risks
                )

            # 保存历史
            self.optimization_history.append(
                {
                    "iteration": self.function_calls,
                    "params": params.copy(),
                    "risk": final_risk,
                    "avg_risk": avg_risk,
                    "balance_penalty": balance_penalty,
                    "group_counts": group_counts.copy(),
                    "group_risks": [
                        r / max(c, 1) for r, c in zip(group_risks, group_counts)
                    ],
                }
            )

            return final_risk

        except Exception as e:
            logger.error(f"目标函数计算错误: {e}")
            return 1e6

    def _check_boundary_constraints(self, boundaries: List[float]) -> bool:
        """检查边界约束"""
        # 检查递增性
        for i in range(len(boundaries) - 1):
            if boundaries[i + 1] <= boundaries[i]:
                return False

        # 检查最小间距
        min_gap = 1.0
        for i in range(len(boundaries) - 1):
            if boundaries[i + 1] - boundaries[i] < min_gap:
                return False

        return True

    def _calculate_balance_penalty(self, group_counts: List[int]) -> float:
        """计算组间平衡惩罚"""
        if sum(group_counts) == 0:
            return 1e6

        # 计算组间不平衡度
        ratios = [count / sum(group_counts) for count in group_counts]

        # 惩罚过小的组（< 10%）和过大的组（> 40%）
        penalty = 0
        for ratio in ratios:
            if ratio < 0.1:
                penalty += 0.5 * (0.1 - ratio)
            elif ratio > 0.4:
                penalty += 0.5 * (ratio - 0.4)

        return penalty

    def _log_optimization_step(
        self,
        params: np.ndarray,
        risk: float,
        group_counts: List[int],
        group_risks: List[float],
    ):
        """记录优化步骤"""
        boundaries = params[:6]
        time_points = params[6:11]

        logger.info(f"\n=== 迭代 {self.function_calls} ===")
        logger.info(f"当前风险: {risk:.6f}")
        logger.info(f"边界: {[f'{b:.2f}' for b in boundaries]}")
        logger.info(f"时点: {[f'{t:.1f}' for t in time_points]}")
        logger.info(f"组大小: {group_counts}")

        # 更新最佳解
        if not self.best_solutions or risk < min(
            s["risk"] for s in self.best_solutions
        ):
            self.best_solutions.append(
                {
                    "iteration": self.function_calls,
                    "params": params.copy(),
                    "risk": risk,
                    "boundaries": boundaries.copy(),
                    "time_points": time_points.copy(),
                }
            )

    def run_differential_evolution(self, **kwargs) -> Dict[str, Any]:
        """运行差分进化优化"""
        logger.info("开始差分进化优化...")

        # 默认参数
        default_params = {
            "bounds": self.bounds,
            "maxiter": 1000,
            "popsize": 50,
            "mutation": (0.5, 1),
            "recombination": 0.7,
            "seed": 42,
            "disp": True,
            "polish": True,
            "atol": 1e-6,
            "tol": 1e-6,
        }

        # 更新参数
        default_params.update(kwargs)

        # 重置计数器
        self.function_calls = 0
        self.optimization_history = []

        # 开始优化
        start_time = time.time()

        try:
            result = differential_evolution(
                func=self.objective_function, **default_params
            )

            optimization_time = time.time() - start_time

            logger.info(f"优化完成！用时: {optimization_time:.2f}秒")
            logger.info(f"函数调用次数: {self.function_calls}")
            logger.info(f"最优风险: {result.fun:.6f}")

            # 分析结果
            analysis = self._analyze_results(result)

            return {
                "scipy_result": result,
                "optimization_time": optimization_time,
                "function_calls": self.function_calls,
                "analysis": analysis,
                "history": self.optimization_history,
            }

        except Exception as e:
            logger.error(f"优化过程出错: {e}")
            raise

    def _analyze_results(self, result) -> Dict[str, Any]:
        """分析优化结果"""
        if not result.success:
            logger.warning("优化未成功收敛！")

        boundaries = result.x[:6]
        time_points = result.x[6:11]

        # 分组统计
        group_stats = []
        for i in range(5):
            group_data = self.data[
                (self.data["BMI"] >= boundaries[i])
                & (self.data["BMI"] < boundaries[i + 1])
            ]

            group_size = len(group_data)
            avg_bmi = group_data["BMI"].mean() if group_size > 0 else 0

            group_stats.append(
                {
                    "group_id": i,
                    "bmi_range": (boundaries[i], boundaries[i + 1]),
                    "detection_time": time_points[i],
                    "size": group_size,
                    "avg_bmi": avg_bmi,
                    "size_ratio": group_size / len(self.data),
                }
            )

        analysis = {
            "success": result.success,
            "final_risk": result.fun,
            "boundaries": boundaries.tolist(),
            "time_points": time_points.tolist(),
            "group_stats": group_stats,
            "convergence_info": {
                "nit": result.nit,
                "nfev": result.nfev,
                "message": result.message,
            },
        }

        return analysis

    def plot_optimization_progress(self, save_path: str = None):
        """绘制优化进程图"""
        if not self.optimization_history:
            logger.warning("没有优化历史数据可绘制")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 提取数据
        iterations = [h["iteration"] for h in self.optimization_history]
        risks = [h["risk"] for h in self.optimization_history]
        avg_risks = [h["avg_risk"] for h in self.optimization_history]
        penalties = [h["balance_penalty"] for h in self.optimization_history]

        # 1. 风险收敛曲线
        axes[0, 0].plot(iterations, risks, "b-", alpha=0.7, label="总风险")
        axes[0, 0].plot(iterations, avg_risks, "r-", alpha=0.7, label="平均风险")
        axes[0, 0].set_xlabel("函数调用次数")
        axes[0, 0].set_ylabel("风险值")
        axes[0, 0].set_title("风险收敛曲线")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 平衡惩罚项
        axes[0, 1].plot(iterations, penalties, "g-", alpha=0.7)
        axes[0, 1].set_xlabel("函数调用次数")
        axes[0, 1].set_ylabel("平衡惩罚")
        axes[0, 1].set_title("组间平衡惩罚")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 边界参数演化
        boundaries_history = np.array(
            [h["params"][:6] for h in self.optimization_history]
        )
        for i in range(6):
            axes[1, 0].plot(
                iterations, boundaries_history[:, i], label=f"b_{i}", alpha=0.7
            )
        axes[1, 0].set_xlabel("函数调用次数")
        axes[1, 0].set_ylabel("BMI边界值")
        axes[1, 0].set_title("BMI边界演化")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 时点参数演化
        timepoints_history = np.array(
            [h["params"][6:11] for h in self.optimization_history]
        )
        for i in range(5):
            axes[1, 1].plot(
                iterations, timepoints_history[:, i], label=f"t_{i}", alpha=0.7
            )
        axes[1, 1].set_xlabel("函数调用次数")
        axes[1, 1].set_ylabel("检测时点（周）")
        axes[1, 1].set_title("检测时点演化")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"优化进程图已保存到: {save_path}")

        plt.show()

    def plot_final_results(self, analysis: Dict, save_path: str = None):
        """绘制最终结果图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. BMI分组可视化
        boundaries = analysis["boundaries"]
        bmi_data = self.data["BMI"]

        axes[0, 0].hist(
            bmi_data, bins=50, alpha=0.7, color="lightblue", label="BMI分布"
        )

        for i, boundary in enumerate(boundaries):
            color = "red" if i == 0 or i == len(boundaries) - 1 else "orange"
            axes[0, 0].axvline(
                boundary, color=color, linestyle="--", label=f"边界 {i}: {boundary:.2f}"
            )

        axes[0, 0].set_xlabel("BMI")
        axes[0, 0].set_ylabel("频数")
        axes[0, 0].set_title("BMI分组结果")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 各组检测时点
        group_ids = [f"组{i}" for i in range(5)]
        time_points = analysis["time_points"]
        group_sizes = [stat["size"] for stat in analysis["group_stats"]]

        bars = axes[0, 1].bar(group_ids, time_points, color="lightgreen", alpha=0.7)

        # 在柱子上标注组大小
        for bar, size in zip(bars, group_sizes):
            height = bar.get_height()
            axes[0, 1].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.1,
                f"n={size}",
                ha="center",
                va="bottom",
            )

        axes[0, 1].set_ylabel("检测时点（周）")
        axes[0, 1].set_title("各组最佳检测时点")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 组大小分布
        axes[1, 0].pie(
            group_sizes,
            labels=group_ids,
            autopct="%1.1f%%",
            colors=["#ff9999", "#66b3ff", "#99ff99", "#ffcc99", "#ff99cc"],
        )
        axes[1, 0].set_title("各组样本分布")

        # 4. 风险函数可视化
        time_range = np.linspace(10, 25, 100)
        conc_range = np.linspace(0, 8, 100)

        # 时间风险曲线
        timing_risks = [self.calculate_timing_risk(t) for t in time_range]
        axes[1, 1].plot(time_range, timing_risks, "r-", label="时间风险", linewidth=2)

        # 准确性风险曲线（以不同浓度为例）
        accuracy_risks = [self.calculate_accuracy_risk(c) for c in conc_range]
        ax2 = axes[1, 1].twinx()
        ax2.plot(conc_range, accuracy_risks, "b-", label="准确性风险", linewidth=2)

        axes[1, 1].set_xlabel("检测时点（周）")
        axes[1, 1].set_ylabel("时间风险", color="r")
        ax2.set_xlabel("Y染色体浓度(%)")
        ax2.set_ylabel("准确性风险", color="b")
        axes[1, 1].set_title("风险函数曲线")

        # 标注各组的检测时点
        for i, t in enumerate(time_points):
            risk = self.calculate_timing_risk(t)
            axes[1, 1].plot(t, risk, "ro", markersize=8)
            axes[1, 1].annotate(
                f"组{i}", (t, risk), xytext=(5, 5), textcoords="offset points"
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"结果图已保存到: {save_path}")

        plt.show()

    def sensitivity_analysis(self, analysis: Dict, perturbation: float = 0.1) -> Dict:
        """敏感性分析"""
        logger.info("开始敏感性分析...")

        base_params = np.concatenate([analysis["boundaries"], analysis["time_points"]])
        base_risk = analysis["final_risk"]

        sensitivities = {}

        for i, param_name in enumerate(
            [
                "b_0",
                "b_1",
                "b_2",
                "b_3",
                "b_4",
                "b_5",
                "t_0",
                "t_1",
                "t_2",
                "t_3",
                "t_4",
            ]
        ):
            # 正向扰动
            perturbed_params = base_params.copy()
            perturbed_params[i] += perturbation

            # 检查约束
            if i < 6:  # 边界参数
                if not self._check_boundary_constraints(perturbed_params[:6]):
                    risk_pos = 1e6
                else:
                    risk_pos = self.objective_function(perturbed_params)
            else:  # 时点参数
                risk_pos = self.objective_function(perturbed_params)

            # 负向扰动
            perturbed_params = base_params.copy()
            perturbed_params[i] -= perturbation

            if i < 6:  # 边界参数
                if not self._check_boundary_constraints(perturbed_params[:6]):
                    risk_neg = 1e6
                else:
                    risk_neg = self.objective_function(perturbed_params)
            else:  # 时点参数
                risk_neg = self.objective_function(perturbed_params)

            # 计算敏感性
            sensitivity = (abs(risk_pos - base_risk) + abs(risk_neg - base_risk)) / (
                2 * perturbation
            )
            sensitivities[param_name] = sensitivity

        logger.info("敏感性分析完成")
        return sensitivities

    def compare_with_simple_grouping(self, analysis: Dict) -> Dict:
        """与简单分组方法比较"""
        logger.info("与简单分组方法比较...")

        # 简单均匀分组
        bmi_min, bmi_max = self.data["BMI"].min(), self.data["BMI"].max()
        simple_boundaries = np.linspace(bmi_min, bmi_max, 6)
        simple_timepoints = [11, 12, 13, 14, 15]  # 经验时点

        simple_params = np.concatenate([simple_boundaries, simple_timepoints])
        simple_risk = self.objective_function(simple_params)

        # 性能提升
        improvement = (simple_risk - analysis["final_risk"]) / simple_risk * 100

        comparison = {
            "optimized_risk": analysis["final_risk"],
            "simple_risk": simple_risk,
            "improvement_percent": improvement,
            "simple_boundaries": simple_boundaries.tolist(),
            "simple_timepoints": simple_timepoints,
            "optimized_boundaries": analysis["boundaries"],
            "optimized_timepoints": analysis["time_points"],
        }

        logger.info(f"优化后风险: {analysis['final_risk']:.6f}")
        logger.info(f"简单分组风险: {simple_risk:.6f}")
        logger.info(f"性能提升: {improvement:.2f}%")

        return comparison


def generate_demo_data(n_samples: int = 1000) -> pd.DataFrame:
    """生成演示数据"""
    np.random.seed(42)

    # 生成BMI数据（偏向高BMI，符合题目描述）
    bmi_data = np.random.gamma(2, 15) + 18  # gamma分布，均值约28
    bmi_data = np.clip(bmi_data, 18, 50)  # 限制范围

    # 其他特征
    ages = np.random.normal(30, 5, n_samples)
    ages = np.clip(ages, 20, 45)

    # 模拟Y染色体达标时间（与BMI相关）
    target_times = 8 + 0.2 * bmi_data + np.random.normal(0, 1, n_samples)
    target_times = np.clip(target_times, 9, 20)

    data = pd.DataFrame(
        {
            "BMI": bmi_data,
            "age": ages,
            "target_time": target_times,
            "individual_factor": np.random.normal(0, 0.5, n_samples),
        }
    )

    return data


def main():
    """主函数：完整的调试和分析流程"""
    print("=== NIPT BMI分组优化 - 差分进化算法调试 ===\n")

    # 1. 生成演示数据
    print("1. 生成演示数据...")

    data = generate_demo_data(1000)
    print(f"数据规模: {len(data)} 个样本")
    print(f"BMI范围: {data['BMI'].min():.2f} - {data['BMI'].max():.2f}")
    print(f"BMI均值: {data['BMI'].mean():.2f}\n")

    # 2. 初始化优化器
    print("2. 初始化优化器...")
    optimizer = NIFTOptimizationDebugger(data, debug_mode=True)

    # 3. 运行差分进化优化
    print("3. 运行差分进化优化...")
    result = optimizer.run_differential_evolution(
        maxiter=200,  # 减少迭代次数以便调试
        popsize=30,  # 减少种群大小
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=42,
    )

    # 4. 分析结果
    print("\n4. 分析优化结果...")
    analysis = result["analysis"]

    print(f"优化是否成功: {analysis['success']}")
    print(f"最终风险值: {analysis['final_risk']:.6f}")
    print(f"优化用时: {result['optimization_time']:.2f} 秒")
    print(f"函数调用次数: {result['function_calls']}")

    print("\n边界结果:")
    for i, boundary in enumerate(analysis["boundaries"]):
        print(f"  b_{i}: {boundary:.2f}")

    print("\n时点结果:")
    for i, timepoint in enumerate(analysis["time_points"]):
        print(f"  t_{i}: {timepoint:.1f} 周")

    print("\n各组统计:")
    for stat in analysis["group_stats"]:
        print(
            f"  组{stat['group_id']}: BMI[{stat['bmi_range'][0]:.2f}, {stat['bmi_range'][1]:.2f}), "
            f"时点={stat['detection_time']:.1f}周, 样本数={stat['size']}, "
            f"比例={stat['size_ratio']:.1%}"
        )

    # 5. 敏感性分析
    print("\n5. 敏感性分析...")
    sensitivities = optimizer.sensitivity_analysis(analysis)

    print("参数敏感性排序（从高到低）:")
    sorted_sensitivities = sorted(
        sensitivities.items(), key=lambda x: x[1], reverse=True
    )
    for param, sensitivity in sorted_sensitivities[:5]:  # 显示前5个最敏感的参数
        print(f"  {param}: {sensitivity:.6f}")

    # 6. 与简单分组比较
    print("\n6. 与简单分组方法比较...")
    comparison = optimizer.compare_with_simple_grouping(analysis)
    print(f"性能提升: {comparison['improvement_percent']:.2f}%")

    # 7. 绘制结果图
    print("\n7. 生成可视化图表...")
    optimizer.plot_optimization_progress("optimization_progress.png")
    optimizer.plot_final_results(analysis, "final_results.png")

    print("\n=== 调试完成 ===")

    return optimizer, result, analysis


if __name__ == "__main__":
    # 运行主程序
    optimizer, result, analysis = main()

    # 额外的调试功能演示
    print("\n=== 额外调试功能演示 ===")

    # 查看最佳解的演化历史
    print(f"\n发现的最佳解数量: {len(optimizer.best_solutions)}")
    if optimizer.best_solutions:
        best = optimizer.best_solutions[-1]
        print(f"最佳解在第 {best['iteration']} 次迭代发现")
        print(f"最佳风险: {best['risk']:.6f}")

    # 收敛性分析
    if len(optimizer.optimization_history) > 100:
        recent_risks = [h["risk"] for h in optimizer.optimization_history[-50:]]
        risk_std = np.std(recent_risks)
        print(f"最近50次迭代的风险标准差: {risk_std:.6f}")
        if risk_std < 1e-6:
            print("算法已收敛")
        else:
            print("算法可能需要更多迭代")

    # 约束违反检查
    final_boundaries = analysis["boundaries"]
    if optimizer._check_boundary_constraints(final_boundaries):
        print("最终解满足所有边界约束")
    else:
        print("警告: 最终解违反边界约束!")

    print("\n调试分析完成！")
