#!/bin/bash
set -e  # Exit on any error

# Change to project directory
cd /root/workspace/eagle

# Activate virtual environment
source ./v_eagle/bin/activate

cd /root/workspace/eagle/eagle_v3_refactored

# Install vLLM if needed
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu125
pip install scikit-learn

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ===== CONFIGURATION =====
BASE_MODEL_PATH="/root/workspace/TensorRT-LLM/workspace/model/Qwen3-32B"
TEACHER_MODEL_PATH="/root/workspace/TensorRT-LLM/workspace/model/Qwen3-32B"
OUTPUT_DIR="./eagle_qwen_data_v3"
VLLM_PORT=8500
EVAL_SPLIT_RATIO=0.05
BATCH_SIZE=4
TEMPERATURE=0.0 
VOCAB_SIZE=151936

# ===== UTILITY FUNCTIONS =====

cleanup_gpu() {
    echo -e "${GREEN}Checking for GPU-using processes...${NC}"
    
    PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk '/[0-9]/{print $1}' || true)
    
    if [ -z "$PIDS" ]; then
        echo -e "${GREEN}No GPU-using processes found. GPU memory is clean!${NC}"
    else
        echo -e "${RED}Found GPU-using processes: $PIDS${NC}"
        echo "Cleaning up these processes..."
        
        for PID in $PIDS; do
            if [[ "$PID" =~ ^[0-9]+$ ]]; then
                echo -e "Killing process ${RED}$PID${NC}..."
                kill -9 $PID 2>/dev/null && echo -e "Process ${GREEN}$PID cleaned.${NC}" || echo -e "${RED}Failed to kill $PID${NC}"
            fi
        done
    fi
}

cleanup_port() {
    local PORT=$1
    echo -e "${GREEN}Checking port $PORT...${NC}"
    
    PIDS_ON_PORT=$(lsof -t -i:$PORT 2>/dev/null || true)
    
    if [ -z "$PIDS_ON_PORT" ]; then
        echo -e "${GREEN}Port $PORT is clean.${NC}"
    else
        echo -e "${RED}Found processes using port $PORT: $PIDS_ON_PORT${NC}"
        for PID in $PIDS_ON_PORT; do
            if [[ "$PID" =~ ^[0-9]+$ ]]; then
                kill -9 $PID 2>/dev/null && echo -e "Process ${GREEN}$PID cleaned.${NC}"
            fi
        done
    fi
}

# ===== CLEANUP =====
echo -e "${YELLOW}=== Cleaning up resources ===${NC}"
cleanup_gpu
cleanup_port $VLLM_PORT
sleep 5

# ===== STAGE 1: BUILD DATASET WITH SPLIT =====
echo -e "${YELLOW}=== STAGE 1: Building and splitting dataset ===${NC}"
python ./data/eagle_data_pipeline.py \
    --stage build \
    --spec "openai/gsm8k:main:20000,tatsu-lab/alpaca:20000,anthropic/hh-rlhf:40000,openai/webgpt_comparisons:20000" \
    --output-dir "$OUTPUT_DIR" \
    --eval-split-ratio $EVAL_SPLIT_RATIO \
    --cn-weight 0.0 \
    --seed 42

# ===== STAGE 2: EXTRACT BASE MODEL FEATURES (WITH LOGITS) =====
echo -e "${YELLOW}=== STAGE 2: Extracting base model features and logits ===${NC}"

# Process training data
echo "Processing training data..."
python ./data/eagle_data_pipeline.py \
    --stage extract_base \
    --base-model "$BASE_MODEL_PATH" \
    --prompts-dir "${OUTPUT_DIR}/prompts/train" \
    --output-dir "${OUTPUT_DIR}/base_features/train" \
    --max-length 2048 \
    --device "cuda" \
    --num-gpus 8 \
    --save-topk 64 \
    --teacher-temperature $TEMPERATURE \
    --resume

# Process evaluation data
echo "Processing evaluation data..."
python ./data/eagle_data_pipeline.py \
    --stage extract_base \
    --base-model "$BASE_MODEL_PATH" \
    --prompts-dir "${OUTPUT_DIR}/prompts/eval" \
    --output-dir "${OUTPUT_DIR}/base_features/eval" \
    --max-length 2048 \
    --device "cuda" \
    --num-gpus 8 \
    --save-topk 64 \
    --teacher-temperature $TEMPERATURE \
    --resume

# # ===== STAGE 3: EXTRACT TEACHER FEATURES (OPTIONAL) =====
# if [ -n "$TEACHER_MODEL_PATH" ]; then
#     echo -e "${YELLOW}=== STAGE 3: Extracting teacher features ===${NC}"
    
#     # Start vLLM server
#     echo "Starting vLLM API server..."
#     nohup python3 -m vllm.entrypoints.openai.api_server \
#         --model "$TEACHER_MODEL_PATH" \
#         --port $VLLM_PORT \
#         --gpu-memory-utilization 0.8 \
#         --dtype bfloat16 \
#         --tensor-parallel-size 8 > vllm_server.log 2>&1 &
    
#     VLLM_PID=$!
#     echo "vLLM server started with PID: $VLLM_PID"
    
#     # Wait for server to be ready
#     echo "Waiting for vLLM server to start..."
#     for i in {1..150}; do
#         if curl -s http://localhost:$VLLM_PORT/v1/models > /dev/null 2>&1; then
#             echo -e "${GREEN}vLLM server is ready!${NC}"
#             break
#         fi
#         echo -ne "Waiting... $i/150\r"
#         sleep 1
#     done
    
#     # Extract teacher features for training data
#     echo "Extracting teacher features for training data..."
#     python ./data/eagle_data_pipeline.py \
#         --stage extract_teacher \
#         --teacher-model "$TEACHER_MODEL_PATH" \
#         --teacher-url "http://localhost:$VLLM_PORT/v1/completions" \
#         --prompts-dir "${OUTPUT_DIR}/prompts/train" \
#         --output-dir "${OUTPUT_DIR}/teacher_features/train" \
#         --teacher-k 20 \
#         --concurrency 16 \
#         --resume
    
#     # Extract teacher features for evaluation data
#     echo "Extracting teacher features for evaluation data..."
#     python eagle_data_pipeline.py \
#         --stage extract_teacher \
#         --teacher-model "$TEACHER_MODEL_PATH" \
#         --teacher-url "http://localhost:$VLLM_PORT/v1/completions" \
#         --prompts-dir "${OUTPUT_DIR}/prompts/eval" \
#         --output-dir "${OUTPUT_DIR}/teacher_features/eval" \
#         --teacher-k 20 \
#         --concurrency 16 \
#         --resume
    
#     # Kill vLLM server
#     echo "Shutting down vLLM server..."
#     kill $VLLM_PID
#     sleep 5
    
#     # ===== STAGE 4: MERGE FEATURES =====
#     echo -e "${YELLOW}=== STAGE 4: Merging features ===${NC}"
    
#     # Merge training features
#     python eagle_data_pipeline.py \
#         --stage merge \
#         --base-dir "${OUTPUT_DIR}/base_features/train" \
#         --teacher-dir "${OUTPUT_DIR}/teacher_features/train" \
#         --output-dir "${OUTPUT_DIR}/final_features/train"
    
#     # Merge evaluation features
#     python eagle_data_pipeline.py \
#         --stage merge \
#         --base-dir "${OUTPUT_DIR}/base_features/eval" \
#         --teacher-dir "${OUTPUT_DIR}/teacher_features/eval" \
#         --output-dir "${OUTPUT_DIR}/final_features/eval"
# else
#     echo -e "${YELLOW}=== No teacher model specified, using base features only ===${NC}"
#     cp -r "${OUTPUT_DIR}/base_features/train" "${OUTPUT_DIR}/final_features/train"
#     cp -r "${OUTPUT_DIR}/base_features/eval" "${OUTPUT_DIR}/final_features/eval"
# fi

# ===== STAGE 5: TRAIN EAGLE MODEL =====
echo -e "${YELLOW}=== STAGE 5: Training EAGLE model ===${NC}"

# Note: We no longer need --base_model parameter!
# You'll need to add --vocab_size and --hidden_size based on your model
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch train_eagle_v3.py \
    --data_path "./eagle_qwen_data_v3/final_features/train" \
    --output_dir "./eagle_qwen_draft_v3" \
    --vocab_size 151936 \
    --hidden_size 5120 \
    --batch_size 2 \
    --lr 1e-5 \
    --epochs 10 \
    --max_seq_len 2048 \
    --ttt_steps 3 \
    --total_steps 400000 \
    --num_decoder_layers 4 \
    --use_kl_loss \
    --kl_weight 0.3 \
    --enable_trace \
    --gradient_accumulation_steps 4 



# ===== STAGE 6: PREPARE EVALUATION DATASETS =====
echo -e "${YELLOW}=== STAGE 6: Preparing evaluation datasets ===${NC}"
python prepare_eval_datasets.py

# ===== STAGE 7: RUN EVALUATION =====
echo -e "${YELLOW}=== STAGE 7: Running evaluation ===${NC}"

# Quick test with 2 samples
echo "Running quick evaluation..."
python eagle_v3_eval.py \
    --base_model /root/workspace/TensorRT-LLM/workspace/model/Qwen3-32B \
    --draft_ckpt "./eagle_qwen_draft_v3/final/" \
    --dataset gsm8k \
    --data_path "./eval_datasets/gsm8k_test.jsonl" \
    --num_samples 36 \
    --max_new_tokens 1024 \
    --temperature $TEMPERATURE \
    --output_file results_gsm8k_greedy_quick.json

echo -e "${GREEN}=== All tasks completed successfully! ===${NC}"
echo "Training data: ${OUTPUT_DIR}/final_features/train"
echo "Evaluation data: ${OUTPUT_DIR}/final_features/eval"
echo "Model checkpoint: ./eagle_qwen_draft_v3/final/"
echo "Evaluation results: results_gsm8k_quick.json"