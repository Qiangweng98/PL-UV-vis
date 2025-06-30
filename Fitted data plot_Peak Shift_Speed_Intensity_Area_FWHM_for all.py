import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc

# 定义输入和输出路径
input_directory = "/Users/mac/Desktop/WBG experiment/Python/PL Data analyze/250526/Remove noise csv/Output/Fitted results"
output_directory = "/Users/mac/Desktop/WBG experiment/Python/PL Data analyze/250526/Remove noise csv/Output/PL Fig"
os.makedirs(output_directory, exist_ok=True)

# 设置字体
rc('font', family='Arial')

# 获取路径下的所有 CSV 文件
csv_files = [f for f in os.listdir(input_directory) if f.endswith('.csv')]

# 按编号顺序处理文件
for file_name in sorted(csv_files):
    file_path = os.path.join(input_directory, file_name)

    try:
        # 读取csv文件
        data = pd.read_csv(file_path)

        # 检查列数是否足够
        if data.shape[1] < 24:
            print(f"文件 {file_name} 列数不足，已跳过。")
            continue

        # 使用列的索引提取数据
        x = data.iloc[:, 2]  # 第三列作为x轴 (索引从0开始)
        y1 = data.iloc[:, 15]  # 第16列 (P列)
        y2 = data.iloc[:, 16]  # 第17列 (Q列)
        y3 = data.iloc[:, 18]  # 第19列 (S列)
        y4 = data.iloc[:, 20]  # 第21列 (U列)
        y5 = data.iloc[:, 23]  # 第24列 (X列)

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

            # 设置y轴刻度和数字加粗
            if i < 2:  # 仅前两张图显示 y 轴刻度和数字
                axes[i].tick_params(axis='y', direction='in', labelsize=24, labelcolor='black', width=2, which='both')
            else:  # 后三张图不显示 y 轴刻度和数字
                axes[i].tick_params(axis='y', direction='in', left=False, right=False, labelleft=False)

            # 设置x轴刻度朝内，数字加粗
            axes[i].tick_params(axis='x', direction='in', labelsize=24, width=2)

            # 设置刻度标签加粗
            for label in axes[i].get_xticklabels():
                label.set_fontweight('bold')
                label.set_fontsize(24)
            if i < 2:  # 仅前两张图设置 y 轴刻度数字加粗
                for label in axes[i].get_yticklabels():
                    label.set_fontweight('bold')
                    label.set_fontsize(24)

            # 设置边框加粗
            for spine in ['top', 'right', 'left', 'bottom']:
                axes[i].spines[spine].set_linewidth(2)

        # 设置x轴标题加粗
        axes[-1].set_xlabel('Time (s)', fontsize=32, fontweight='bold')

        # 调整布局，减少图形留白
        plt.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.12)

        # 保存图形
        output_file_name = file_name.split('.')[0] + '_Fitted data plot_Peak Shift_Speed_Intensity_Area_FWHM.png'
        output_file_path = os.path.join(output_directory, output_file_name)
        plt.savefig(output_file_path, dpi=300)
        plt.close()

    except Exception as e:
        print(f"处理文件 {file_name} 时出错: {e}")

print(f"所有图片已保存到 {output_directory}")
