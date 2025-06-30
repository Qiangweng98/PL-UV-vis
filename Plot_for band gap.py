import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from matplotlib.font_manager import FontProperties

# === 数据路径 ===
input_csv_path = "/Users/mac/Desktop/WBG experiment/Python/PL Data analyze/241211/Remove noise csv/Output/Fitted results/For Origin/Band gap_Peak 2 Position Shift_104.csv"
data = pd.read_csv(input_csv_path)

# === 提取数据列 ===
x_data = data.iloc[:, 1]
y_data = data.iloc[:, 4]

# === 筛选数据范围 ===
valid = (x_data >= 1.48) & (x_data <= 2.02) & (y_data >= -5) & (y_data <= 50)
x_data = x_data[valid]
y_data = y_data[valid]

# === 回归分析 ===
X = sm.add_constant(x_data)
model = sm.OLS(y_data, X).fit()
slope = model.params.iloc[1]
intercept = model.params.iloc[0]
r_squared = model.rsquared
p_value = model.pvalues.iloc[1]

# === 回归预测 + CI ===
x_pred = np.linspace(1.48, 2.02, 200)
x_pred_const = sm.add_constant(x_pred)
y_pred = model.predict(x_pred_const)
ci = model.get_prediction(x_pred_const).conf_int(alpha=0.05)
lower, upper = ci[:, 0], ci[:, 1]

# === 颜色设置：所有点一致为 20-30 分组颜色 ===
facecolor = "#9a71c7"  # Color for 20-30 group
size = 60

# === 绘图 ===
plt.figure(figsize=(8, 6))

# 95% CI 填充
plt.fill_between(x_pred, lower, upper, color="#decee6", alpha=1.0, label='95% CI', zorder=1)

# 散点图
plt.scatter(x_data, y_data, s=size, facecolors=facecolor, edgecolors=facecolor,
            alpha=0.8, linewidth=1.2, zorder=2)

# 回归线
plt.plot(x_pred, y_pred, color="#482869", linewidth=3.5, label='Fit', zorder=3)

# y=0 辅助线
plt.axhline(y=0, color='black', linestyle='--', linewidth=2, zorder=4)

# 方程文本
p_text = "P < 0.001" if p_value < 0.001 else f"P = {p_value:.3e}"
if intercept < 0:
    intercept_text = f"- {abs(intercept):.2f}"
else:
    intercept_text = f"+ {intercept:.2f}"
equation_text = f"$y = {slope:.2f}x {intercept_text}$\n$R^2 = {r_squared:.3f}$\n{p_text}"
plt.text(0.05, 0.95, equation_text, transform=plt.gca().transAxes,
         fontsize=18, verticalalignment='top', fontname='Arial', color="#482869")

# 坐标轴设置
plt.xlim(1.48, 2.02)
plt.ylim(-5, 50)
plt.xlabel("Band gap (eV)", fontsize=24, fontname='Arial', weight='bold', labelpad=10)
plt.ylabel("10min Peak Shift (nm)", fontsize=24, fontname='Arial', weight='bold', labelpad=10)
plt.xticks(fontsize=18, fontname='Arial', weight='bold')
plt.yticks(fontsize=18, fontname='Arial', weight='bold')
plt.tick_params(axis='both', which='both', direction='in', width=2)
for spine in plt.gca().spines.values():
    spine.set_linewidth(2)

plt.legend(
    loc='upper right',
    bbox_to_anchor=(0.995, 0.985),
    frameon=True,
    prop={'family': 'Arial', 'size': 14}
)

plt.tight_layout()
plt.savefig("bandgap_shift_film.png", dpi=600)
plt.show()