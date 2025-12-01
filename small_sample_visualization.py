import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb

# 示例数据
small_sample = {
    'HUST': {'1': 1.756, '2': 1.208, '3': 0.364, '4': 0.220, 'full':0.210},
    'XJTU': {'1': 1.057, '2': 0.892, '3': 0.934, '4': 0.782, 'full': 0.718},
    'MIT': {'1': 1.380, '2': 0.330, '3': 0.274, '4': 0.170, 'full': 0.086},
    # 'TJU': {'1': 11.2, '2': 10.6, '3': 9.9, '4': 9.3, 'full': 8.8}
}
# small_sample = {
#     'HUST': {'1': 0.0159, '2': 0.0107, '3': 0.0035, '4': 0.0022, 'full':0.0020},
#     'XJTU': {'1': 0.0114, '2': 0.0087, '3': 0.0090, '4': 0.0078, 'full':0.0068},
#     'MIT': {'1': 0.0161, '2': 0.0041, '3': 0.0033, '4': 0.0022, 'full': 0.0012},
#     # 'TJU': {'1': 11.2, '2': 10.6, '3': 9.9, '4': 9.3, 'full': 8.8}
# }

# 准备数据
datasets = list(small_sample.keys())
categories = ['1', '2', '3', '4', 'full']
category_labels = ['1 Bat', '2 Bat', '3 Bat', '4 Bat', 'Full']
x = np.arange(len(datasets))
width = 0.15
spacing = 0.02

# 创建颜色映射
colors = LinearSegmentedColormap.from_list('deep_blue', 
          ['#1a2f4b', '#2a4b6e', '#3a6891', '#5a8cb4', '#7ab0d7'])

# 计算最小最大值
all_values = [val for dataset in small_sample.values() for val in dataset.values()]
min_val, max_val = min(all_values), max(all_values)

# 创建图形
plt.figure(figsize=(12, 6))
ax = plt.gca()

# 绘制柱状图
for i, (cat, label) in enumerate(zip(categories, category_labels)):
    values = [small_sample[ds][cat] for ds in datasets]
    normalized_values = [(v - min_val) / (max_val - min_val) for v in values]
    bar_colors = [colors(nv) for nv in normalized_values]
    
    # 创建带透明度的柱状图颜色 (alpha=0.5)
    bar_colors_transparent = [(*to_rgb(color), 0.5) for color in bar_colors]
    
    # 绘制半透明柱状图
    bars = ax.bar(x + i*(width + spacing), values, width, color=bar_colors_transparent)
    
    # 添加数据标签和分类标签
    for rect, bar_color in zip(bars, bar_colors):  # 注意这里使用原始bar_colors
        height = rect.get_height()
        
        # 使用原始颜色作为文字颜色 (不透明)
        text_color = bar_color  
        
        # 在bar顶部添加数值标签
        ax.text(rect.get_x() + rect.get_width()/2., 
                height * 1.02,
                f'{height}%',
                ha='center',
                va='bottom',
                fontsize=15,
                color=text_color,
                fontweight='bold',
                rotation=90,
                ) 
        
        # 在bar中间添加分类标签
        ax.text(rect.get_x() + rect.get_width()/2.,
                0.01,
                label,
                ha='center',
                va='bottom',
                fontsize=12.3,
                color=text_color,
                fontweight='bold',
                transform=ax.transData,
                ) 
# 设置图形属性
ax.set_ylabel('MAPE (%)', fontsize=18, fontweight='bold')
# ax.set_ylabel('RMSE', fontsize=14, fontweight='bold')
ax.set_xticks(x + width*2)
ax.set_xticklabels(datasets, fontsize=16, fontweight='bold')
ax.tick_params(axis='y', labelsize=14)


# 设置y轴范围
ax.set_ylim(0, max_val * 1.1)

# 去除边框
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# 调整布局
plt.tight_layout()

# 保存图像
save_path = '/mnt/wenjt5/project1/plot/small_sample'
os.makedirs(save_path, exist_ok=True)
plt.savefig(f'{save_path}/small_sample_comparison.pdf', dpi=300, bbox_inches='tight')
plt.show()