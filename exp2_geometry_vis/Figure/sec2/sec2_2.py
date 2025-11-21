import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

# ========================= 🎨 自定义颜色 =========================
color_depth  = "#F3E5AB"   # 浅蓝
color_normal = "#7FAEB2"   # 绿色（建议突出 Normal）
color_both   = "#E8AFAF"   # 橙色

# ========================= 🧩 选择模型 (SD2.1 / SAM / DINOv2) =========================
model = "DINOv2"   # ← 修改此项生成不同模型的图

# ========================= 📊 数据定义 =========================
if model == "SD2.1":
    layers = ["up0", "up1", "up2", "up3"]
    base   = np.array([0.365, 0.579, 0.449, 0.261])
    depth  = np.array([0.388, 0.596, 0.468, 0.278])
    normal = np.array([0.397, 0.600, 0.457, 0.297])
    both   = np.array([0.411, 0.591, 0.496, 0.316])

elif model == "SAM":
    layers = ["L3", "L6", "L9", "L12"]
    base   = np.array([0.378, 0.482, 0.546, 0.556])
    depth  = np.array([0.382, 0.474, 0.554, 0.545])
    normal = np.array([0.408, 0.506, 0.536, 0.550])
    both   = np.array([0.412, 0.515, 0.575, 0.562])

elif model == "DINOv2":
    layers = ["L3", "L6", "L9", "L12"]
    base   = np.array([0.416, 0.561, 0.645, 0.656])
    depth  = np.array([0.395, 0.546, 0.609, 0.635])
    normal = np.array([0.429, 0.580, 0.629, 0.628])
    both   = np.array([0.438, 0.568, 0.628, 0.624])

else:
    raise ValueError("模型名不正确，请在 'SD2.1', 'SAM', 'DINOv2' 中选择。")

# ========================= ⚙️ 计算差值 =========================
delta_depth  = depth - base
delta_normal = normal - base
delta_both   = both - base

# ========================= 📈 绘图 =========================
fig, ax = plt.subplots(figsize=(5, 3.5), dpi=150)
x = np.arange(len(layers))

ax.axhline(0, color="#888", lw=1, ls="--", alpha=0.6)
ax.plot(x, delta_depth,  marker="o", lw=2, color=color_depth,  label="+Depth")
ax.plot(x, delta_normal, marker="o", lw=2.5, color=color_normal, label="+Normal")
ax.plot(x, delta_both,   marker="o", lw=2, color=color_both,   label="+Both")

ax.set_xticks(x)
ax.set_xticklabels(layers)
ax.set_ylabel("ΔmIoU vs Base (%)")
ax.set_title(f"{model}")

ax.yaxis.set_major_formatter(PercentFormatter(1.0))

# 优化显示范围与风格
ax.grid(axis="y", linestyle="--", alpha=0.25)
ax.set_ylim(-0.12, 0.15)
ax.legend(frameon=False, loc="upper right", fontsize=9)

fig.tight_layout()
save_path = f"delta_miou_layers_{model.replace('.', '_')}.png"
plt.savefig(save_path, bbox_inches="tight")
plt.show()

print(f"✅ 图已保存为: {save_path}")