import pandas as pd
import numpy as np
import sys
import os
# 加载CSV文件
train_csv_path = '/root/xzcllwx_ws/GameFormer/training_log/Exp6/train_log.csv'
df = pd.read_csv(train_csv_path)

# 获取最后一个epoch的数据
last_epoch = df.iloc[-1]
print(f"分析Epoch {int(last_epoch['epoch'])}的指标")

# 对象类型
object_types = ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']

# 指标类型
metric_types = ['minADE', 'minFDE', 'miss_rate', 'overlap_rate', 'mAP']

# K值
k_values = [5, 9, 15]

# 计算每个指标在各对象类型和K值上的均值
object_type_means = {}
for obj_type in object_types:
    print(f"\n{obj_type}类型预测结果:")
    for metric in metric_types:
        # 计算该对象类型下特定指标各K值的均值
        metric_values = []
        for k in k_values:
            col_name = f"{metric}_TYPE_{obj_type}_{k}"
            if col_name in last_epoch.index:
                metric_values.append(last_epoch[col_name])
        
        if metric_values:
            mean_value = np.mean(metric_values)
            print(f"  平均{metric}: {mean_value:.4f}")
            object_type_means.setdefault(metric, []).append(mean_value)

# 计算所有对象类型的综合均值
print("\n所有对象类型的综合均值:")
for metric in metric_types:
    if metric in object_type_means:
        overall_mean = np.mean(object_type_means[metric])
        print(f"  平均{metric}: {overall_mean:.4f}")

# 计算每个K值下各指标的均值
print("\n不同K值下的均值:")
for k in k_values:
    print(f"\nK={k}:")
    for metric in metric_types:
        # 收集所有对象类型在特定K值下该指标的值
        k_values_for_metric = []
        for obj_type in object_types:
            col_name = f"{metric}_TYPE_{obj_type}_{k}"
            if col_name in last_epoch.index:
                k_values_for_metric.append(last_epoch[col_name])
        
        if k_values_for_metric:
            k_mean = np.mean(k_values_for_metric)
            print(f"  平均{metric}: {k_mean:.4f}")
            # 将打印内容保存到result.log文件中
            
log_file_path = os.path.join(os.path.dirname(train_csv_path), 'data_analy_result.log')
with open(log_file_path, 'w') as log_file:
    sys.stdout = log_file  # 重定向标准输出到文件
    # 重新运行打印逻辑
    print(f"分析Epoch {int(last_epoch['epoch'])}的指标")
    for obj_type in object_types:
        print(f"\n{obj_type}类型预测结果:")
        for metric in metric_types:
            metric_values = []
            for k in k_values:
                col_name = f"{metric}_TYPE_{obj_type}_{k}"
                if col_name in last_epoch.index:
                    metric_values.append(last_epoch[col_name])
            if metric_values:
                mean_value = np.mean(metric_values)
                print(f"  平均{metric}: {mean_value:.4f}")
    print("\n所有对象类型的综合均值:")
    for metric in metric_types:
        if metric in object_type_means:
            overall_mean = np.mean(object_type_means[metric])
            print(f"  平均{metric}: {overall_mean:.4f}")
    print("\n不同K值下的均值:")
    for k in k_values:
        print(f"\nK={k}:")
        for metric in metric_types:
            k_values_for_metric = []
            for obj_type in object_types:
                col_name = f"{metric}_TYPE_{obj_type}_{k}"
                if col_name in last_epoch.index:
                    k_values_for_metric.append(last_epoch[col_name])
            if k_values_for_metric:
                k_mean = np.mean(k_values_for_metric)
                print(f"  平均{metric}: {k_mean:.4f}")
    sys.stdout = sys.__stdout__  # 恢复标准输出
print(f"结果已保存到 {log_file_path}")