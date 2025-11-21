import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
import seaborn as sns

# 设置风格，使其符合学术论文标准
sns.set(style="whitegrid")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 12

# === 数据准备 (来自你提供的截图) ===
models = ['DINO', 'DINOv2', 'CLIP', 'SigLIP', 'SAM', 'SD2.1']

# X轴数据：几何误差 (越低越好)
depth_rmse = [0.5071, 0.3307, 0.9351, 0.7187, 0.5665, 0.4801] # 来自截图1 [cite: 1]
normal_rmse = [28.35, 22.41, 34.68, 34.96, 26.89, 24.68]      # 来自截图1 [cite: 1]

# Y轴数据：Affordance Base mIoU (越高越好)
# 数据来自截图2中的灰色行 (Base)
miou = [0.477, 0.670, 0.520, 0.517, 0.546, 0.585]             # 来自截图2 [cite: 2]

# 定义颜色和标记，区分不同类型的模型
# DINO系列(蓝色), VLM(红色), SAM(绿色), Gen(紫色)
colors = ['#1f77b4', '#1f77b4', '#d62728', '#d62728', '#2ca02c', '#9467bd']
markers = ['o', 'o', '^', '^', 's', '*']

# === 绘图 ===
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

def plot_correlation(ax, x_data, y_data, x_label, title):
    # 1. 绘制散点
    for i, model in enumerate(models):
        ax.scatter(x_data[i], y_data[i], color=colors[i], marker=markers[i], s=100, label=model, edgecolors='k', alpha=0.8)
        # 给点加上文字标签（稍微偏移一点以免挡住点）
        ax.text(x_data[i], y_data[i]+0.005, model, fontsize=10, ha='center', va='bottom')

    # 2. 计算并绘制回归线
    slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)
    line_x = np.array([min(x_data), max(x_data)])
    line_y = slope * line_x + intercept
    
    # 绘制趋势线
    ax.plot(line_x, line_y, color='gray', linestyle='--', alpha=0.6, linewidth=2)
    
    # 显示相关系数 Pearson r
    # r 为负数表示负相关（RMSE越低，mIoU越高，这是我们要证明的）
    text_str = f'Pearson $r = {r_value:.2f}$'
    ax.text(0.05, 0.05, text_str, transform=ax.transAxes, fontsize=12, 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray'))

    # 3. 装饰图表
    ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
    ax.set_ylabel('Affordance mIoU', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.6)

# 绘制左图：Depth vs mIoU
plot_correlation(axes[0], depth_rmse, miou, 'Depth RMSE (m) $\downarrow$', 'Correlation: Depth Geometry vs. Affordance')

# 绘制右图：Normal vs mIoU
plot_correlation(axes[1], normal_rmse, miou, 'Normal RMSE ($^\circ$) $\downarrow$', 'Correlation: Normal Geometry vs. Affordance')

plt.tight_layout()
plt.savefig('geometry_correlation.pdf', dpi=300, bbox_inches='tight')
plt.show()