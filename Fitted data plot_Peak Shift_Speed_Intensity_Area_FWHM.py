import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc

# 定义输入路径
input_directory = "/Users/mac/Desktop/WBG experiment/Python/PL Data analyze/241211/Remove noise csv/Output/Fitted results"
file_name = "Nr_29_fitted_results.csv"
file_path = os.path.join(input_directory, file_name)

# 检查文件是否存在
if not os.path.exists(file_path):
    raise FileNotFoundError(f"文件 {file_name} 不存在于路径 {input_directory} 中。")

# 读取csv文件（确保第一行作为列名）
data = pd.read_csv(file_path)

# 使用列的索引提取数据
x = data.iloc[:, 2]  # 第三列作为x轴 (索引从0开始)
y1 = data.iloc[:, 15]  # 第16列 (P列)
y2 = data.iloc[:, 16]  # 第17列 (Q列)
y3 = data.iloc[:, 18]  # 第19列 (S列)
y4 = data.iloc[:, 20]  # 第21列 (U列)
y5 = data.iloc[:, 23]  # 第24列 (X列)

# 设置字体
rc('font', family='Arial')

# 创建子图
fig, axes = plt.subplots(5, 1, figsize=(12, 8), sharex=True, gridspec_kw={'hspace': 0.05})

# 定义子图数据和标签
shapes = ['o', 'D', 's', 'o', 'D']  # 圆形，菱形，正方形，圆形，菱形
colors = ['#FCCDC9', '#EE9D9F', '#DE6A69', '#EE9D9F', '#FCCDC9']
edge_colors = ['#FF9999', '#CC4444', '#CC3333', '#CC4444', '#FF9999']
background_colors = ['white', '#E2F4FE', '#BBE6FA', '#E2F4FE', 'white']
data_list = [(y1, 'Shift\n(nm)'), (y2, 'Speed\n(nm/s)'), (y3, 'Intensity\n(a.u.)'), (y4, 'Area\n(a.u.)'), (y5, 'FWHM\n(a.u.)')]

for i, (y, label) in enumerate(data_list):
    axes[i].set_facecolor(background_colors[i])  # 设置底色
    axes[i].plot(x, y, marker=shapes[i], markersize=4, color=colors[i], markeredgecolor=edge_colors[i], markeredgewidth=1.5, linewidth=2)

    # 设置x轴和y轴范围，确保所有点都显示并留一定的上下范围
    y_max = y.max() + (y.max() - y.min()) * 0.3
    y_min = y.min() - (y.max() - y.min()) * 0.3 if y.min() != 0 else y.min() - (y.max() - y.min()) * 0.3
    axes[i].set_ylim(y_min, y_max)
    axes[i].set_xlim(0, 600)

    # 在每张图左侧设置标题，上下居中
    axes[i].text(-62, (y_max + y_min) / 2, label, fontsize=24, rotation=90, va='center', fontweight='bold', ha='center')

    # 去除y轴标题
    axes[i].set_ylabel('')

    # 设置y轴刻度和数字加粗（前两张图显示 y 轴刻度和数字）
    if i < 2:
        axes[i].tick_params(axis='y', direction='in', labelsize=24, labelcolor='black', width=2, which='both')
        for label in axes[i].get_yticklabels():
            label.set_fontweight('bold')
            label.set_fontsize(24)
    else:  # 后三张图不显示 y 轴刻度和数字
        axes[i].tick_params(axis='y', direction='in', left=False, right=False, labelleft=False)

    # 设置x轴刻度朝内，数字加粗
    axes[i].tick_params(axis='x', direction='in', labelsize=24, width=2)
    for label in axes[i].get_xticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(24)

    # 设置边框加粗
    for spine in ['top', 'right', 'left', 'bottom']:
        axes[i].spines[spine].set_linewidth(2)

# 设置x轴标题加粗
axes[-1].set_xlabel('Time (s)', fontsize=32, fontweight='bold')

# 调整布局，减少图形留白
plt.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.12)

# 显示图形
plt.show()
