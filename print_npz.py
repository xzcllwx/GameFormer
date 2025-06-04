# import numpy as np

# # 加载 .npz 文件
# npz_file = np.load('/home/user/alstar/xzcllwx_ws/data/nuplan/dataset/nuplan-v1.1/splits/mini_1_process/us-nv-las-vegas-strip_000e00790bc45da7.npz')

# # 打印文件中的所有数组
# for file in npz_file.files:
#     print(f"{file}: {npz_file[file]}")
import numpy as np
import sys
import os

def compare_npz_files(file1, file2):
    npz1 = np.load(file1)
    npz2 = np.load(file2)

    # 获取两个文件中的所有数组名称
    keys1 = set(npz1.files)
    keys2 = set(npz2.files)

    # 比较数组名称
    if keys1 != keys2:
        print("The files have different arrays.")
        print("File 1 arrays:", keys1)
        print("File 2 arrays:", keys2)
        return
    else:
        print("The files have the same arrays.")

    # 比较每个数组的内容
    for key in keys1:
        array1 = npz1[key]
        array2 = npz2[key]
        if array1.shape != array2.shape:
            print(f"Arrays '{key}' are different.")
            print(f"File 1 '{key}': {array1}")
            print(f"File 2 '{key}': {array2}")
        else:
            print(f"Arrays '{key}' are the same.")
            print(f"File 1 '{key}': {array1.shape}")
            print(f"File 2 '{key}': {array2.shape}")

def print_npz_to_log(npz_file, log_file):
    """将NPZ文件内容打印到指定日志文件"""
    # 重定向标准输出到日志文件
    original_stdout = sys.stdout
    with open(log_file, 'w') as f:
        sys.stdout = f
        
        try:
            # 加载NPZ文件
            data = np.load(npz_file)
            
            # 打印文件基本信息
            print(f"NPZ文件路径: {os.path.abspath(npz_file)}")
            print(f"文件大小: {os.path.getsize(npz_file) / 1024:.2f} KB")
            print(f"包含的数组: {data.files}")
            print(f"数组数量: {len(data.files)}\n")
            
            # 遍历并打印每个数组的信息
            for key in data.files:
                array = data[key]
                print(f"数组名称: {key}")
                print(f"  形状: {array.shape}")
                print(f"  数据类型: {array.dtype}")
                
                # 对于较小的数组，打印完整内容
                # if array.size <= 100:
                print(f"  值: \n{array}\n")
                # else:
                #     # 对于大数组，只打印前5个元素
                #     print(f"  前5个值: \n{array.flatten()[:5]} ... (共{array.size}个元素)\n")
        
        except Exception as e:
            print(f"处理文件时出错: {e}")
        
        # 恢复标准输出
        finally:
            sys.stdout = original_stdout

# 文件路径
npz_file = '/root/xzcllwx_ws/womd_process/testing_interactive_process/c47287bd491e8d6a_11_29_interest.npz'
log_file = '/root/xzcllwx_ws/GameFormer/npz_content.log'

# file2 = '/home/user/alstar/xzcllwx_ws/data/nuplan/dataset/nuplan-v1.1/splits/mini_2_process/us-nv-las-vegas-strip_000e00790bc45da7.npz'

# 比较文件
# compare_npz_files(file1, file2)

print_npz_to_log(npz_file, log_file)
print(f"NPZ文件内容已保存到: {log_file}")