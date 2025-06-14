#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
extract_teacher_feats.py  (v2)
------------------------------
Call an OpenAI-compatible *completions* endpoint provided by vLLM
and save top-k log-probs (optionally hidden states) for each prompt.

❯ python extract_teacher_feats.py \
      --prompts-dir ./data \
      --out-dir ./data/teacher_pt \
      --url http://localhost:8000/v1/completions \
      --model /root/workspace/TensorRT-LLM/workspace/model/Qwen3-32B \
      --k 20 --concurrency 64
"""
import argparse, json, pathlib, asyncio, httpx, os, torch, tqdm

# ---------- CLI ----------
p = argparse.ArgumentParser()
p.add_argument("--prompts-dir", required=True)
p.add_argument("--out-dir", required=True)
p.add_argument("--url", default="http://localhost:8000/v1/completions")  # <<< completions
p.add_argument("--model", required=True)                                 # <<< 模型字段改全路径
p.add_argument("--k", type=int, default=0, help="top-k logprobs (0=none)")
p.add_argument("--concurrency", type=int, default=32)
p.add_argument("--resume", action="store_true")
p.add_argument("--max-tokens", type=int, default=1)                      # <<< 允许自定义
args = p.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
client = httpx.AsyncClient(timeout=None)
sem = asyncio.Semaphore(args.concurrency)

# ---------- 收集 prompts ----------
def iter_jsonl(root):
    for f in pathlib.Path(root).rglob("*.jsonl"):
        for line in open(f, encoding="utf-8"):
            j = json.loads(line)
            yield j["id"], j["turns"][0]

prompts = list(iter_jsonl(args.prompts_dir))
print("Total prompts:", len(prompts))

# ---------- 调用 vLLM ----------
async def fetch(pid, text):
    save_path = f"{args.out_dir}/{pid}.pt"
    if args.resume and pathlib.Path(save_path).exists():
        return
    payload = {
        "model": args.model,                # <<< 使用完整路径
        "prompt": text,                     # <<< completions 用 prompt
        "max_tokens": args.max_tokens,
        "temperature": 0.7
    }
    if args.k > 0:
        payload["logprobs"] = args.k
        payload["echo"] = True
    async with sem:
        r = await client.post(args.url, json=payload)
        r.raise_for_status()
        data = r.json()

    choice = data["choices"][0]
    record = {"prompt": text}

    if args.k > 0:
        lp = choice["logprobs"]
        record.update({
            "tokens": lp["tokens"],
            "token_logprobs": lp["token_logprobs"],
            "top_logprobs": lp["top_logprobs"]   # list[dict], 每个 dict 有 token_ids/logprobs
        })

    torch.save(record, save_path)

async def main():
    tasks = [fetch(pid, txt) for pid, txt in prompts]   # ← 关键修正
    # 旧 tqdm 版本写法
    for fut in tqdm.tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        await fut
    # 若已升级 tqdm: await tqdm.asyncio.tqdm.gather(*tasks, total=len(tasks))
    await client.aclose()



if __name__ == "__main__":
    asyncio.run(main())
