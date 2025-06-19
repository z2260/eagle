#!/bin/bash
# run_pipeline.sh - 改进的EAGLE V3运行脚本
# 修复了默认温度设置和路径问题

set -e  # 遇到错误立即退出

# 配置
BASE_MODEL="Qwen/Qwen2.5-32B-Instruct"
VOCAB_SIZE=151936
EVAL_DATA="./eval_data"
BATCH_SIZE=4
TEMPERATURE=0.0  # 修复：使用低温进行首次评估

# 创建必要的目录
mkdir -p ./data/preprocessed
mkdir -p ./checkpoints
mkdir -p ./logs
mkdir -p ./results

echo "=========================================="
echo "EAGLE V3 Pipeline - 改进版"
echo "基础模型: $BASE_MODEL"
echo "评估温度: $TEMPERATURE"
echo "=========================================="

# Step 1: 准备评估数据集
echo "[Step 1] 准备评估数据集..."
if [ ! -d "$EVAL_DATA" ]; then
    cd ../ref
    python prepare_eval_datasets.py \
        --output_dir ../eagle_v3_refactored/eval_data \
        --num_samples 100 \
        --datasets mt-bench gsm8k
    cd ../eagle_v3_refactored
fi

# Step 2: 数据预处理
echo "[Step 2] 预处理数据..."
python -m data.preprocess_data \
    --model-name "$BASE_MODEL" \
    --prompts-dir "$EVAL_DATA" \
    --output-dir ./data/preprocessed \
    --save-topk 20 \
    --temperature 1.0

# Step 3: 训练草稿模型
echo "[Step 3] 训练草稿模型..."
python -m training.train_eagle_v3 \
    --vocab_size $VOCAB_SIZE \
    --data_path ./data/preprocessed \
    --output_dir ./checkpoints \
    --batch_size $BATCH_SIZE \
    --epochs 1 \
    --use_kl_loss

# Step 4: 评估模型
echo "[Step 4] 评估模型..."
python -m evaluation.eagle_v3_eval \
    --base_model "$BASE_MODEL" \
    --draft_ckpt ./checkpoints/latest.pt \
    --data_path "$EVAL_DATA" \
    --num_samples 20 \
    --temperature $TEMPERATURE \
    --max_new_tokens 64 \
    --output_file ./results/eval_results.json

echo "=========================================="
echo "Pipeline完成！"
echo "结果保存在: ./results/eval_results.json"
echo "==========================================" 