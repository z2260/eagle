
cd ~/workspace/eagle          # 进入工程根目录
python3 -m venv v_eagle       # 创建 venv（目录名 v_eagle）
source v_eagle/bin/activate   # 激活虚拟环境

python -m pip install --upgrade pip       # 建议先升级 pip
pip install datasets tqdm                 # datasets 是 Hugging Face 数据集库

pip install datasets[apache-arrow] transformers accelerate

python3 ./data/build_dataset.py \
  --spec "
    BelleGroup/train_1M_CN:6000,
    BAAI/COIG:3000,
    HuggingFaceH4/ultrachat_200k:2000,
    shareAI/ShareGPT-Chinese-English-90k:2000,
    openai/gsm8k:1000,
    MU-NLPC/Calc-gsm8k:500,
    HuggingFaceH4/CodeAlpaca_20K:1500,
    openai/openai_humaneval:500,
    philschmid/mt-bench:200,       
    ibm-research/finqa:400,
    GBaker/MedQA-USMLE-4-options:400
    " \
  --output ./data \
  --seed 42 \
  --cn-weight 1.5

pip install vllm --extra-index-url https://download.pytorch.org/whl/cu125 

python3 -m vllm.entrypoints.openai.api_server \
       --model /root/workspace/TensorRT-LLM/workspace/model/Qwen3-32B \
       --port 8000 \
       --gpu-memory-utilization 0.8 \
       --dtype bfloat16 \
       --tensor-parallel-size 8
       
python3 ./data/extract_teacher_feats.py \
  --prompts-dir ./data \
  --out-dir ./data/teacher_pt \
  --model /root/workspace/TensorRT-LLM/workspace/model/Qwen3-32B \
  --url http://localhost:8000/v1/completions \
  --k 20 --concurrency 64

python3 -m eagle.train.main \
  --basepath   /root/workspace/TensorRT-LLM/workspace/model/Qwen3-32B \
  --configpath /root/workspace/TensorRT-LLM/workspace/model/Qwen3-32B/config.json \
  --tmpdir     /root/workspace/eagle/data/teacher_pt \
  --cpdir      /root/workspace/eagle/ckpts/qwen3_eagle \
  --lr 5e-4 --bs 4 --gradient-accumulation-steps 2

580098d6f7fed4f4ffa7283704d80a3ac6f4d43d
python3 debug.py ./eagle_qwen_data/base_features

accelerate launch --multi_gpu \
    train_eagle_v3.py \
    --base_model "/root/workspace/TensorRT-LLM/workspace/model/Qwen3-32B" \
    --data_path "./eagle_qwen_data/final_features" \
    --output_dir "./eagle_qwen_draft" \
    --batch_size 8 \
    --lr 5e-4 \
    --epochs 5 \
    --max_seq_len 40960

python eagle_v3_eval.py \
    --base_model /root/workspace/TensorRT-LLM/workspace/model/Qwen3-32B \
    --draft_ckpt ./eagle_qwen_draft \
    --dataset mt-bench \
    --data_path ./eval_datasets/mt_bench.json \
    --num_samples 100 \
    --use_vllm 

python eagle_v3_eval.py \
    --base_model /root/workspace/TensorRT-LLM/workspace/model/Qwen3-32B \
    --draft_ckpt ./eagle_qwen_draft/final/ \
    --dataset gsm8k \
    --data_path ./eval_datasets/gsm8k_test.jsonl \
    --num_samples 100 \
    --max_new_tokens 256 \
    --temperature 0.0 \
    --output_file results_gsm8k.json
