import os
import re
import math

# 设置文件目录
directory = './'  # 更改为您的日志文件存储目录

# 准备一个字典来存储结果
results = []

# 遍历目录中的所有文件
for filename in os.listdir(directory):
    if filename.endswith("_seq.log"):
        # 构造完整的文件路径
        seq_path = os.path.join(directory, filename)
        ours_path = seq_path.replace("_seq.log", "_ours.log")

        # 检查对应的 _ours.log 文件是否存在
        if os.path.exists(ours_path):
            try:
                # 提取文件名中的 perception 和 generation 参数
                match = re.search(r'perception_([0-9_]+)_generation_([0-9_]+)', filename)
                if match:
                    # 将 _ 替换为 .，并去除任何可能的多余符号
                    perception_value = float(match.group(1).replace('_', '.').strip('.'))
                    generation_value = float(match.group(2).replace('_', '.').strip('.'))
                    # print(perception_value)
                    # print(generation_value)

                    # 计算 perception 的转换值 (-2 对应 2^-2, -3 对应 2^-3, ...)
                    perception_transformed = math.log2(perception_value)     
                    generation_transformed = math.log2(generation_value)           
                    # 读取 _seq.log 文件最后一行的 Throughput
                    with open(seq_path, 'r') as file:
                        seq_content = file.readlines()
                    seq_throughput = None
                    for line in reversed(seq_content):
                        if "Throughput" in line:
                            try:
                                seq_throughput = float(line.split()[1])
                                break
                            except ValueError:
                                continue

                    # 读取 _ours.log 文件最后一行的 Throughput
                    with open(ours_path, 'r') as file:
                        ours_content = file.readlines()
                    ours_throughput = None
                    for line in reversed(ours_content):
                        if "Throughput" in line:
                            try:
                                ours_throughput = float(line.split()[1])
                                break
                            except ValueError:
                                continue

                    # 计算 speedup
                    if seq_throughput is not None and ours_throughput is not None:
                        speedup = ours_throughput / seq_throughput
                        # 添加到结果列表
                        results.append((perception_transformed, generation_transformed, speedup))
                    else:
                        # 如果没有找到 Throughput，返回 None
                        results.append((perception_transformed, generation_transformed, None))
                else:
                    print(f"Skipping file {filename}, unable to extract perception and generation values.")
            except Exception as e:
                # 如果在处理文件时出现错误，例如文件格式不正确或无法转换数据类型，将忽略该文件
                print(f"Error processing {seq_path}: {e}")
                continue

# 输出结果
for result in results:
    print(result)





# import os
# import re

# # 设置文件目录
# directory = './'  # 更改为您的日志文件存储目录

# # 准备一个字典来存储结果
# results = []

# # 遍历目录中的所有文件
# for filename in os.listdir(directory):
#     if filename.endswith("_seq.log"):
#         # print(filename)
#         # 构造完整的文件路径
#         seq_path = os.path.join(directory, filename)
#         ours_path = seq_path.replace("_seq.log", "_ours.log")

#         # 检查对应的 _ours.log 文件是否存在
#         if os.path.exists(ours_path):
#             try:
#                 # 读取 _seq.log 文件获取 Perception params 和 Generation params
#                 with open(seq_path, 'r') as file:
#                     seq_content = file.readlines()
#                 perception_param = None
#                 generation_param = None
#                 for line in seq_content:
#                     if "Perception params:" in line:
#                         perception_param = float(line.split()[2])
#                     if "Generation params:" in line:
#                         generation_param = float(line.split()[2])

#                 if perception_param is None or generation_param is None:
#                     continue

#                 # 读取 _seq.log 文件最后一行的 Throughput
#                 seq_throughput = float(seq_content[-1].split()[1])

#                 # 读取 _ours.log 文件最后一行的 Throughput
#                 with open(ours_path, 'r') as file:
#                     ours_content = file.readlines()
#                 ours_throughput = float(ours_content[-1].split()[1])

#                 # 计算 speedup
#                 speedup = ours_throughput / seq_throughput

#                 # 保存结果
#                 results.append((perception_param, generation_param, speedup))
#             except Exception as e:
#                 # 如果在处理文件时出现错误，例如文件格式不正确或无法转换数据类型，将忽略该文件
#                 print(f"Error processing {seq_path}: {e}")
#                 continue

# # 输出结果
# for result in results:
#     print(result)


# import os
# import re

# # 设置文件目录
# directory = '/path/to/your/log/files'  # 更改为您的日志文件存储目录

# # 准备一个字典来存储结果
# results = []

# # 遍历目录中的所有文件
# for filename in os.listdir(directory):
#     if filename.endswith("_seq.log"):
#         # 获取文件前缀
#         prefix = filename[:-8]  # 移除 "_seq.log" 得到前缀
        
#         # 构造完整的文件路径
#         seq_path = os.path.join(directory, filename)
#         ours_path = seq_path.replace("_seq.log", "_ours.log")

#         # 检查对应的 _ours.log 文件是否存在
#         if os.path.exists(ours_path):
#             try:
#                 # 读取 _seq.log 文件获取 Perception params 和 Generation params
#                 with open(seq_path, 'r') as file:
#                     seq_content = file.readlines()
#                 perception_param = None
#                 generation_param = None
#                 for line in seq_content:
#                     if "Perception params:" in line:
#                         perception_param = float(line.split()[2])
#                     if "Generation params:" in line:
#                         generation_param = float(line.split()[2])

#                 if perception_param is None or generation_param is None:
#                     continue

#                 # 读取 _seq.log 文件最后一行的 Throughput
#                 seq_throughput = float(seq_content[-1].split()[1])

#                 # 读取 _ours.log 文件最后一行的 Throughput
#                 with open(ours_path, 'r') as file:
#                     ours_content = file.readlines()
#                 ours_throughput = float(ours_content[-1].split()[1])

#                 # 计算 speedup
#                 speedup = ours_throughput / seq_throughput

#                 # 保存结果
#                 results.append((prefix, perception_param, generation_param, speedup))
#             except Exception as e:
#                 # 如果在处理文件时出现错误，例如文件格式不正确或无法转换数据类型，将忽略该文件
#                 print(f"Error processing {seq_path}: {e}")
#                 continue

# # 输出结果
# for result in results:
#     print(result)
