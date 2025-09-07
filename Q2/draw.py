import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams["font.sans-serif"] = ["SimHei", "SimSong"]

# 读取Excel文件
df = pd.read_excel("data.xlsx")

# 创建散点图
plt.figure(figsize=(10, 6))
plt.scatter(df["BMI"], df["target_time"], alpha=0.7, s=50)

# 设置图表标题和轴标签
plt.title("BMI与目标时间散点图", fontsize=16, fontweight="bold")
plt.xlabel("BMI", fontsize=12)
plt.ylabel("Target Time", fontsize=12)

# 添加网格线，使图表更易读
plt.grid(True, alpha=0.3)

# 调整布局
plt.tight_layout()

# 显示图表
plt.show()

# 可选：保存图表
# plt.savefig('bmi_scatter_plot.png', dpi=300, bbox_inches='tight')
