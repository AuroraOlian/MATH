from scipy.interpolate import CubicSpline
import numpy as np
import matplotlib.pyplot as plt

# 基本用法
x = np.array([0, 1, 2, 3, 4])
y = np.array([1, 4, 2, 5, 3])

# 创建样条对象
cs = CubicSpline(x, y)

# 插值
xx = np.linspace(0, 4, 1000)
yy = cs(xx)

# 绘图
plt.plot(x, y, 'o', xx, yy, '-')
plt.show()