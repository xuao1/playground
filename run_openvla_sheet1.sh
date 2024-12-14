#!/bin/bash

# 检查是否传递了模型名称参数
if [ -z "$1" ]; then
  echo "Usage: $0 <model_name>"
  exit 1
fi

MODEL_NAME="$1"

CONFIG_FILES=("config_seq.yaml" "config_parallel.yaml" "config_ours.yaml")

# 循环遍历每个配置文件，并替换 model 字段为 "MODEL_NAME"
for file in "${CONFIG_FILES[@]}"
do
    if [ -f "$file" ]; then
        sed -i "s/model: \".*\"/model: \"$MODEL_NAME\"/" $file
        echo "Updated model to '$MODEL_NAME' in $file"
    else
        echo "Warning: $file not found, cannot update model."
    fi
done

# # ==========================================================================================================
# # ================================================ Seq =====================================================
# 定义日志文件
LOG_FILE="${MODEL_NAME}_seq.log"

# 串行运行 5 次命令
for i in {1..5}
do
    echo "Running iteration $i..." >> $LOG_FILE
    python sche_plan.py --config ./config_seq.yaml >> $LOG_FILE 2>&1
    echo "Iteration $i completed." >> $LOG_FILE
    echo "--------------------------------------" >> $LOG_FILE
done

echo "All iterations are complete. Output saved to $LOG_FILE." >> $LOG_FILE

# 计算 Query duration 的平均值
average_query_duration=$(awk '/Query duration:/ {duration_sum += $3; count++} END {print duration_sum/count}' $LOG_FILE)
average_query_duration_seconds=$(printf "%.8f" $(echo "scale=8; $average_query_duration / 1000" | bc))

# 输出平均值到日志文件
echo "Average Query Duration: $average_query_duration ms ($average_query_duration_seconds seconds)" >> $LOG_FILE

if (( $(echo "$average_query_duration_seconds > 0" | bc -l) )); then
    throughput=$(printf "%.8f" $(echo "scale=8; 1 / $average_query_duration_seconds" | bc))
    echo "Throughput: $throughput requests/second" >> $LOG_FILE
else
    echo "Throughput: Infinite (duration zero)" >> $LOG_FILE
fi


# ==========================================================================================================
# ================================================ Ours ====================================================
# 定义日志文件
LOG_FILE="${MODEL_NAME}_ours.log"

# Ours 运行 5 次命令
for i in {1..5}
do
    echo "Running iteration $i..." >> $LOG_FILE
    python sche_plan.py --config ./config_ours.yaml >> $LOG_FILE 2>&1
    echo "Iteration $i completed." >> $LOG_FILE
    echo "--------------------------------------" >> $LOG_FILE
done

echo "All iterations are complete. Output saved to $LOG_FILE." >> $LOG_FILE

# 计算 Frame interval 的平均值
average_query_duration=$(awk '/Frame interval:/ {duration_sum += $3; count++} END {print duration_sum/count}' $LOG_FILE)
average_query_duration_seconds=$(printf "%.8f" $(echo "scale=8; $average_query_duration" | bc))

# 输出平均值到日志文件
echo "Average Query Duration: $average_query_duration s ($average_query_duration_seconds seconds)" >> $LOG_FILE

# 更新 config_parallel.yaml 文件
config_file="config_parallel.yaml"
sed -i "s/req_interval: [0-9.]\+/req_interval: $average_query_duration_seconds/" $config_file

echo "config_parallel.yaml updated with new req_interval: $average_query_duration_seconds."

if (( $(echo "$average_query_duration_seconds > 0" | bc -l) )); then
    throughput=$(printf "%.8f" $(echo "scale=8; 1 / $average_query_duration_seconds" | bc))
    echo "Throughput: $throughput requests/second" >> $LOG_FILE
else
    echo "Throughput: Infinite (duration zero)" >> $LOG_FILE
fi


# ==========================================================================================================
# ============================================ Parallel ====================================================
# 定义日志文件
LOG_FILE="${MODEL_NAME}_parallel.log"

# 配置文件路径
CONFIG_FILE="./config_parallel.yaml"

# 定义 worker_num 的范围
for worker_num in 1 2 3 4 8 16
# for worker_num in {1..3}
do
    LOG_FILE="${MODEL_NAME}_parallel_worker${worker_num}.log"

    # 修改 config_parallel.yaml 中的 worker_num
    sed -i "s/worker_num: [0-9]\+/worker_num: $worker_num/" $CONFIG_FILE
    
    echo "Testing with worker_num=$worker_num" >> $LOG_FILE

    # 运行 5 次命令
    for i in {1..5}
    do
        echo "Running iteration $i with worker_num=$worker_num..." >> $LOG_FILE
        python sche_plan.py --config $CONFIG_FILE >> $LOG_FILE 2>&1
        echo "Iteration $i completed." >> $LOG_FILE
        echo "--------------------------------------" >> $LOG_FILE
    done

    echo "All iterations for worker_num=$worker_num are complete. Output saved to $LOG_FILE." >> $LOG_FILE

    # 计算 Query duration 的平均值
    average_query_duration=$(awk '/Query duration:/ {duration_sum += $3; count++} END {print duration_sum/count}' $LOG_FILE)
    average_query_duration_seconds=$(printf "%.8f" $(echo "scale=8; $average_query_duration/1000" | bc))

    # 输出平均值到日志文件
    echo "Average Query Duration: $average_query_duration ms ($average_query_duration_seconds seconds)" >> $LOG_FILE

    if (( $(echo "$average_query_duration_seconds > 0" | bc -l) )); then
        throughput=$(printf "%.8f" $(echo "scale=8; 1 / $average_query_duration_seconds" | bc))
        echo "Throughput: $throughput requests/second" >> $LOG_FILE
    else
        echo "Throughput: Infinite (duration zero)" >> $LOG_FILE
    fi

    echo "====================================" >> $LOG_FILE
done
