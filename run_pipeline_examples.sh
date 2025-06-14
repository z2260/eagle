#!/bin/bash

cd /root/workspace/eagle

source ./v_eagle/bin/activate

pip install vllm --extra-index-url https://download.pytorch.org/whl/cu125 

# 设置一个醒目的颜色，用于输出信息
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${GREEN}正在检查占用GPU的进程...${NC}"

PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | awk '/[0-9]/{print $1}')

# 检查是否有找到进程
if [ -z "$PIDS" ]; then
    echo -e "${GREEN}没有发现占用GPU的进程。显存是干净的！${NC}"
else
    echo -e "${RED}发现以下占用GPU的进程PID: ${PIDS}${NC}"
    echo "准备清理这些进程..."

    # 循环遍历所有找到的PID并杀死它们
    for PID in $PIDS; do
        # 检查PID是否真的是一个数字，防止意外
        if [[ "$PID" =~ ^[0-9]+$ ]]; then
            echo -e "正在杀死进程 ${RED}$PID${NC}..."
            # 使用 kill -9 强制杀死进程
            kill -9 $PID
            if [ $? -eq 0 ]; then
                echo -e "进程 ${GREEN}$PID 已成功清理。${NC}"
            else
                echo -e "${RED}清理进程 $PID 失败。可能需要手动干预。${NC}"
            fi
        fi
    done

    echo -e "${GREEN}所有GPU进程清理完毕！${NC}"
fi

sleep 5  # 等待进程完全结束

echo -e "${GREEN}正在检查占用GPU的进程...${NC}"
# (您原来的清理GPU进程的逻辑保持不变)
# ...

# --- 新增的清理端口逻辑 ---
echo -e "${GREEN}正在检查并清理端口 ${TARGET_PORT}...${NC}"

# 使用 lsof 命令查找占用指定端口的进程PID
# lsof -t -i:PORT 会直接返回PID，更精确
PIDS_ON_PORT=$(lsof -t -i:${TARGET_PORT})

if [ -z "$PIDS_ON_PORT" ]; then
    echo -e "${GREEN}端口 ${TARGET_PORT} 是干净的，没有进程占用。${NC}"
else
    echo -e "${RED}发现以下进程PID占用了端口 ${TARGET_PORT}: ${PIDS_ON_PORT}${NC}"
    echo "准备清理这些进程..."

    for PID in $PIDS_ON_PORT; do
        if [[ "$PID" =~ ^[0-9]+$ ]]; then
            echo -e "正在强制杀死进程 ${RED}$PID${NC}..."
            # 使用 kill -9 强制杀死进程
            kill -9 $PID
            if [ $? -eq 0 ]; then
                echo -e "进程 ${GREEN}$PID 已成功清理。${NC}"
            else
                echo -e "${RED}清理进程 $PID 失败。可能需要手动干预。${NC}"
            fi
        fi
    done
    echo -e "${GREEN}所有占用端口 ${TARGET_PORT} 的进程清理完毕！${NC}"
fi
# --- 新增逻辑结束 ---


sleep 5  # 等待进程完全结束

# 后台运行 OpenAI API 服务器
echo "启动 vLLM API 服务器（仅用于带 teacher 的数据处理）..."
nohup python3 -m vllm.entrypoints.openai.api_server \
       --model /root/workspace/TensorRT-LLM/workspace/model/Qwen3-32B \
       --port 8500 \
       --gpu-memory-utilization 0.8 \
       --dtype bfloat16 \
       --tensor-parallel-size 8 > vllm_server.log 2>&1 &

# 保存进程ID
VLLM_PID=$!
echo "vLLM 服务器已在后台启动，PID: $VLLM_PID"

# 等待服务器启动
wait_time=150
echo "等待服务器启动 (预计 $wait_time 秒)..."

# 使用 for 循环进行倒计时
for ((i=$wait_time; i>=0; i--)); do
    # 使用 \r 回到行首，-n 不换行，实现原地更新
    echo -ne "剩余时间: $i 秒... \r"
    sleep 1
done

echo -e "\n服务器启动检查完成"

# 检查服务器是否启动成功
if curl -s http://localhost:8500/v1/models > /dev/null; then
    echo "服务器启动成功！"
else
    echo "服务器启动失败，请检查日志文件 vllm_server.log"
    exit 1
fi

# 1. Qwen model with teacher (full pipeline)
echo "开始运行 Eagle 数据处理（带 teacher）..."
python eagle_data_pipeline.py \
    --spec "openai/gsm8k:20000,tatsu-lab/alpaca:20000,anthropic/hh-rlhf:40000,openai/webgpt_comparisons:20000" \
    --base-model "/root/workspace/TensorRT-LLM/workspace/model/Qwen3-32B" \
    --teacher-model "/root/workspace/TensorRT-LLM/workspace/model/Qwen3-32B" \
    --teacher-url "http://localhost:8500/v1/completions" \
    --output-dir "./eagle_qwen_data" \
    --cn-weight 0 \
    --teacher-k 20 \
    --concurrency 16 \
    --max-length 2048 \
    --resume

# 关闭 vLLM 服务器，释放显存
echo "关闭 vLLM 服务器，释放显存..."
kill $VLLM_PID
sleep 5  # 等待进程完全结束
echo "vLLM 服务器已关闭"

# # 训练 Eagle 模型
echo "开始训练 Eagle 模型..."
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch \
  train_eagle_v3.py \
    --base_model "/root/workspace/TensorRT-LLM/workspace/model/Qwen3-32B" \
    --data_path "./eagle_qwen_data/final_features" \
    --output_dir "./eagle_qwen_draft" \
    --batch_size 8 \
    --lr 5e-5 \
    --epochs 10 \
    --max_seq_len 2048 \
    --ttt_steps 3 \
    --total_steps 400000 \
    --num_decoder_layers 4 \
    --use_kl_loss \
    --kl_weight 0.7 \
    --enable_trace \
    --gradient_accumulation_steps 4

echo "训练完成！开始准备评估数据集..."

# 准备评估数据集
echo "准备评估数据集..."
python prepare_eval_datasets.py

echo "评估数据集准备完成！开始运行 EAGLE-3 评估..."

# 运行一个小样本评估进行快速验证
echo "运行快速验证评估（2个样本）..."
python eagle_v3_eval.py \
    --base_model /root/workspace/TensorRT-LLM/workspace/model/Qwen3-32B \
    --draft_ckpt ./eagle_qwen_draft/final/ \
    --dataset gsm8k \
    --data_path ./eval_datasets/gsm8k_test.jsonl \
    --num_samples 2 \
    --max_new_tokens 128 \
    --temperature 1.0 \
    --output_file results_gsm8k_quick.json

echo "所有任务已完成！评估结果已保存到 results_gsm8k_full.json 和 results_gsm8k_quick.json"

# echo "运行快速验证评估（2个样本）..."
# python eagle_v3_eval.py \
#     --base_model /root/workspace/TensorRT-LLM/workspace/model/Qwen3-32B \
#     --draft_ckpt ./eagle_qwen_draft/final/ \
#     --dataset mt-bench \
#     --data_path ./eval_datasets/mt_bench.json \
#     --num_samples 2 \
#     --max_new_tokens 128 \
#     --temperature 1.0 \
#     --output_file results_mt_bench_quick.json