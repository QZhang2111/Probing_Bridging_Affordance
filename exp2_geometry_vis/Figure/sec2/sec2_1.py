import matplotlib.pyplot as plt
import numpy as np

# 模型及数据（来自表格）
models = ["DINO", "DINOv2", "CLIP", "SigLIP", "SAM", "SD 2.1"]

base =   np.array([0.477, 0.670, 0.520, 0.517, 0.546, 0.585])
depth =  np.array([0.534, 0.662, 0.535, 0.542, 0.544, 0.562])
normal = np.array([0.549, 0.651, 0.581, 0.556, 0.545, 0.596])
both =   np.array([0.518, 0.655, 0.573, 0.545, 0.575, 0.575])

# 横坐标
x = np.arange(len(models))
width = 0.18
offsets = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]

fig, ax = plt.subplots(figsize=(10, 5.2), dpi=150)

# 👉 A3BBDB
bars_base   = ax.bar(x + offsets[0], base,   width, label="Base", color="#7F8A8C")      # 深绿色
bars_depth  = ax.bar(x + offsets[1], depth,  width, label="+Depth", color="#AFC8DC")    # 深棕色
bars_normal = ax.bar(x + offsets[2], normal, width, label="+Normal", color="#A3BBDB")   # 深蓝色
bars_both   = ax.bar(x + offsets[3], both,   width, label="+Both", color="#7FAEB2")     # 深紫色

# 轴与标签
ax.set_ylabel("mIoU")
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15, ha="right")

# Y 轴范围从 0.3 开始
ymax = max(base.max(), depth.max(), normal.max(), both.max(), 0.746)
ax.set_ylim(0.3, min(1.0, ymax + 0.08))

# 网格线
ax.grid(axis='y', linestyle='--', alpha=0.3)

# 自动标注柱状数值
def autolabel(bars):
    for b in bars:
        height = b.get_height()
        ax.annotate(f"{height:.3f}",
                    xy=(b.get_x() + b.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

for bars in [bars_base]:
    autolabel(bars)

# 水平参考线（只显示数值，不显示"Supervised"单词）
supervised = 0.746
ax.axhline(supervised, color='black', linestyle='-', linewidth=1.0)
ax.text(x[-1]+0.7, supervised, "(0.746)", va='center', fontsize=8)

# 图例 - 使用更大的字号
ax.legend(ncol=4, frameon=False, loc="upper left", bbox_to_anchor=(0,1.0), fontsize=14)

fig.tight_layout()

# 保存
output_path = "fig3_2.png"
plt.savefig(output_path, bbox_inches="tight")
print("Saved to", output_path)