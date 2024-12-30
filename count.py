import os
import re

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
                # 读取 _seq.log 文件获取 Perception params 和 Generation params
                with open(seq_path, 'r') as file:
                    seq_content = file.readlines()
                perception_param = None
                generation_param = None
                for line in seq_content:
                    if "Perception params:" in line:
                        perception_param = float(line.split()[2])
                    if "Generation params:" in line:
                        generation_param = float(line.split()[2])

                if perception_param is None or generation_param is None:
                    continue

                # 读取 _seq.log 文件最后一行的 Throughput
                seq_throughput = float(seq_content[-1].split()[1])

                # 读取 _ours.log 文件最后一行的 Throughput
                with open(ours_path, 'r') as file:
                    ours_content = file.readlines()
                ours_throughput = float(ours_content[-1].split()[1])

                # 计算 speedup
                speedup = ours_throughput / seq_throughput

                # 保存结果
                results.append((perception_param, generation_param, speedup))
            except Exception as e:
                # 如果在处理文件时出现错误，例如文件格式不正确或无法转换数据类型，将忽略该文件
                print(f"Error processing {seq_path}: {e}")
                continue

# 输出结果
for result in results:
    print(result)
