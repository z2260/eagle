#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eagle_data_pipeline.py - Fixed multiprocessing Queue sharing with top-k probability extraction
修复字段命名和维度一致性问题
"""

import argparse
import asyncio
import json
import logging
import os
import pathlib
import sys
import subprocess
import shutil
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
import math
import torch.multiprocessing as torch_mp
import time
import threading

import torch
import torch.nn.functional as F
import httpx
from tqdm import tqdm

# Import from eagle_core for consistency
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from eagle_core import get_fusion_indices, fuse_hidden_states, FUSE_LAYERS

# We'll import the actual functions rather than copying them
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Not: from eagle_core import get_fusion_indices, fuse_hidden_states, FUSE_LAYERS
from data.build_dataset import build_dataset

logging.basicConfig(
    format="%(levelname)s | %(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


def extract_base_features_worker(args_tuple):
    """Worker function for extracting base model features on a specific GPU."""
    gpu_id, model_name, prompts_chunk, output_dir, max_length, resume, progress_queue, save_topk, temperature = args_tuple
    
    log.info(f"Worker {gpu_id}: Processing {len(prompts_chunk)} prompts on cuda:{gpu_id}")
    
    try:
        # Import here to avoid issues with multiprocessing
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch.nn.functional as F
        
        # Set device for this worker
        device = f"cuda:{gpu_id}"
        
        # Load model on specific GPU
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Load model with explicit device_map
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map={"": device},  # Force all layers to specific GPU
            trust_remote_code=True
        )
        model.eval()
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        processed = 0
        start_time = time.time()
        
        for idx, prompt_data in enumerate(prompts_chunk):
            prompt_id = prompt_data['id']
            text = prompt_data['text']
            
            save_path = pathlib.Path(output_dir) / f"{prompt_id}.pt"
            
            if resume and save_path.exists():
                processed += 1
                # Report progress
                try:
                    progress_queue.put({
                        'gpu_id': gpu_id,
                        'processed': 1,
                        'skipped': True,
                        'total_processed': processed,
                        'total_chunk': len(prompts_chunk)
                    })
                except:
                    pass  # Ignore queue errors for progress reporting
                continue
            
            try:
                # Tokenize
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=max_length,
                    truncation=True,
                    padding=True
                ).to(device)
                
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                    
                    # Extract features
                    fusion_indices = get_fusion_indices(model_name, model.config.num_hidden_layers)
                    fused_features = fuse_hidden_states(outputs.hidden_states, fusion_indices)
                    logits = outputs.logits
                    
                    # Apply temperature scaling and get top-k probabilities
                    # scaled_logits = logits / temperature
                    scaled_logits = logits.to(torch.float32) / temperature
                    scaled_logits = scaled_logits - scaled_logits.max(dim=-1, keepdim=True).values
                    probs = F.softmax(scaled_logits, dim=-1).to(torch.float16)

                    
                    # Get top-k probabilities and indices for each position
                    topk_probs, topk_indices = torch.topk(probs, k=save_topk, dim=-1)
                    
                    # ===== 关键修复：统一维度处理，去掉 batch 维度 =====
                    # 确保所有张量的第一个维度都是序列长度，而不是batch
                    input_ids_squeezed = inputs['input_ids'][0].cpu()  # [seq_len]
                    attention_mask_squeezed = inputs['attention_mask'][0].cpu()  # [seq_len]
                    fused_features_squeezed = fused_features.cpu()  # [seq_len, hidden_dim] (已经去掉了batch维度)
                    topk_probs_squeezed = topk_probs[0].cpu()  # [seq_len, k]
                    topk_indices_squeezed = topk_indices[0].cpu()  # [seq_len, k]
                    
                    # 添加一致性断言，确保所有张量的序列长度一致
                    seq_len = input_ids_squeezed.size(0)
                    assert attention_mask_squeezed.size(0) == seq_len, f"attention_mask length mismatch: {attention_mask_squeezed.size(0)} != {seq_len}"
                    assert fused_features_squeezed.size(0) == seq_len, f"fused_features length mismatch: {fused_features_squeezed.size(0)} != {seq_len}"
                    assert topk_probs_squeezed.size(0) == seq_len, f"topk_probs length mismatch: {topk_probs_squeezed.size(0)} != {seq_len}"
                    assert topk_indices_squeezed.size(0) == seq_len, f"topk_indices length mismatch: {topk_indices_squeezed.size(0)} != {seq_len}"
                    
                    # ===== 关键修复：字段命名与训练脚本保持一致 =====
                    feature_data = {
                        'hidden_states': fused_features_squeezed,
                        'num_fused_layers': FUSE_LAYERS,
                        'teacher_topk_probs': topk_probs_squeezed,    # 修复字段名：topk_probs -> teacher_topk_probs
                        'teacher_topk_indices': topk_indices_squeezed,  # 修复字段名：topk_indices -> teacher_topk_indices
                        'input_ids': input_ids_squeezed,
                        'attention_mask': attention_mask_squeezed,
                        'prompt': text,
                        'prompt_id': prompt_id,
                        'teacher_temperature': temperature,
                        'save_topk': save_topk
                    }
                    
                    torch.save(feature_data, save_path)
                    processed += 1
                    
                    # Report progress
                    try:
                        progress_queue.put({
                            'gpu_id': gpu_id,
                            'processed': 1,
                            'skipped': False,
                            'total_processed': processed,
                            'total_chunk': len(prompts_chunk)
                        })
                    except:
                        pass  # Ignore queue errors for progress reporting
                    
                    # Clear cache periodically
                    if processed % 100 == 0:
                        torch.cuda.empty_cache()
                    
            except torch.cuda.OutOfMemoryError:
                log.error(f"GPU-{gpu_id}: OOM processing {prompt_id}, skipping")
                torch.cuda.empty_cache()
                try:
                    progress_queue.put({
                        'gpu_id': gpu_id,
                        'processed': 1,
                        'skipped': False,
                        'error': True,
                        'total_processed': processed,
                        'total_chunk': len(prompts_chunk)
                    })
                except:
                    pass
            except Exception as e:
                log.error(f"GPU-{gpu_id}: Error processing {prompt_id}: {e}")
                try:
                    progress_queue.put({
                        'gpu_id': gpu_id,
                        'processed': 1,
                        'skipped': False,
                        'error': True,
                        'total_processed': processed,
                        'total_chunk': len(prompts_chunk)
                    })
                except:
                    pass
        
        # Send completion signal
        try:
            progress_queue.put({
                'gpu_id': gpu_id,
                'completed': True,
                'total_processed': processed,
                'total_chunk': len(prompts_chunk),
                'duration': time.time() - start_time
            })
        except:
            pass
        
        log.info(f"GPU-{gpu_id}: Successfully processed {processed}/{len(prompts_chunk)} prompts")
        return processed
        
    except Exception as e:
        log.error(f"GPU-{gpu_id}: Worker failed: {e}")
        try:
            progress_queue.put({
                'gpu_id': gpu_id,
                'worker_failed': True,
                'error_msg': str(e)
            })
        except:
            pass
        import traceback
        traceback.print_exc()
        return 0


def progress_monitor(progress_queue, total_prompts: int, num_workers: int):
    """Monitor progress from all workers and display unified progress bar."""
    
    pbar = tqdm(total=total_prompts, desc="Processing prompts", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    worker_stats = {}
    completed_workers = 0
    total_processed = 0
    
    while completed_workers < num_workers:
        try:
            # Get progress update with timeout
            update = progress_queue.get(timeout=1.0)
            
            gpu_id = update.get('gpu_id')
            
            # Initialize worker stats if needed
            if gpu_id not in worker_stats:
                worker_stats[gpu_id] = {
                    'processed': 0,
                    'chunk_size': 0,
                    'completed': False
                }
            
            # Handle different types of updates
            if update.get('completed'):
                # Worker completed
                completed_workers += 1
                worker_stats[gpu_id]['completed'] = True
                duration = update.get('duration', 0)
                chunk_processed = update.get('total_processed', 0)
                chunk_size = update.get('total_chunk', 0)
                
                pbar.write(f"GPU-{gpu_id}: Completed {chunk_processed}/{chunk_size} prompts in {duration:.1f}s")
                
            elif update.get('worker_failed'):
                # Worker failed
                completed_workers += 1
                error_msg = update.get('error_msg', 'Unknown error')
                pbar.write(f"GPU-{gpu_id}: Worker failed - {error_msg}")
                
            elif update.get('processed'):
                # Regular progress update
                if not update.get('error', False):
                    total_processed += update['processed']
                    pbar.update(update['processed'])
                    
                    # Update worker stats
                    worker_stats[gpu_id]['processed'] = update.get('total_processed', 0)
                    worker_stats[gpu_id]['chunk_size'] = update.get('total_chunk', 0)
                    
                    # Update description with per-GPU info
                    gpu_info = []
                    for gid, stats in worker_stats.items():
                        if not stats['completed'] and stats['chunk_size'] > 0:
                            progress_pct = (stats['processed'] / stats['chunk_size']) * 100
                            # gpu_info.append(f"GPU{gid}:{progress_pct:.0f}%")
                    
                    if gpu_info:
                        desc = f"Processing prompts ({', '.join(gpu_info)})"
                        pbar.set_description(desc)
                
        except:
            # Timeout or other error - continue waiting
            continue
    
    pbar.close()
    return total_processed


def check_gpu_availability():
    """Check available GPUs and their memory."""
    if not torch.cuda.is_available():
        log.warning("CUDA is not available")
        return []
    
    available_gpus = []
    for i in range(torch.cuda.device_count()):
        try:
            # Test if we can actually use this GPU
            torch.cuda.set_device(i)
            memory_free = torch.cuda.get_device_properties(i).total_memory
            available_gpus.append((i, memory_free))
            log.info(f"GPU {i}: {torch.cuda.get_device_name(i)} - {memory_free // (1024**3)} GB")
        except Exception as e:
            log.warning(f"GPU {i} not available: {e}")
    
    return available_gpus


def split_prompts_for_gpus(prompts: List[Dict], num_gpus: int) -> List[List[Dict]]:
    """Split prompts evenly across available GPUs."""
    chunk_size = math.ceil(len(prompts) / num_gpus)
    chunks = []
    
    for i in range(num_gpus):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(prompts))
        if start_idx < len(prompts):
            chunks.append(prompts[start_idx:end_idx])
    
    return chunks


def extract_base_features_parallel(model_name: str, prompts_dir: str, output_dir: str,
                                 max_length: int = 2048, resume: bool = True,
                                 num_gpus: int = None, save_topk: int = 20, 
                                 temperature: float = 1.0):
    """Extract base model features using multiple GPUs in parallel with unified progress."""
    
    # Check available GPUs
    available_gpus = check_gpu_availability()
    
    if not available_gpus:
        log.error("No GPUs available for processing")
        return
    
    # Use specified number of GPUs or all available
    if num_gpus is None:
        num_gpus = len(available_gpus)
    else:
        num_gpus = min(num_gpus, len(available_gpus))
    
    gpu_ids = [gpu_id for gpu_id, _ in available_gpus[:num_gpus]]
    log.info(f"Using {num_gpus} GPUs: {gpu_ids}")
    log.info(f"Extracting top-{save_topk} probabilities with temperature={temperature}")
    
    # Collect all prompts
    prompts = []
    for jsonl_file in pathlib.Path(prompts_dir).rglob("*.jsonl"):
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if 'turns' in record and record['turns']:
                        prompts.append({
                            'id': record['id'],
                            'text': record['turns'][0]
                        })
                except json.JSONDecodeError:
                    continue
    
    log.info(f"Found {len(prompts)} prompts to process")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Split prompts across GPUs
    prompt_chunks = split_prompts_for_gpus(prompts, num_gpus)
    
    # Set multiprocessing start method to spawn for CUDA compatibility
    try:
        torch_mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    # Use Manager to create shared queue
    with mp.Manager() as manager:
        progress_queue = manager.Queue()
        
        # Prepare arguments for workers
        worker_args = []
        for i, chunk in enumerate(prompt_chunks):
            if chunk:  # Only add non-empty chunks
                gpu_id = gpu_ids[i]
                worker_args.append((
                    gpu_id, model_name, chunk, output_dir, max_length, resume, 
                    progress_queue, save_topk, temperature
                ))
        
        # Start progress monitor in a separate thread
        monitor_thread = threading.Thread(
            target=progress_monitor,
            args=(progress_queue, len(prompts), len(worker_args))
        )
        monitor_thread.start()
        
        # Run parallel extraction
        total_processed = 0
        with ProcessPoolExecutor(max_workers=len(worker_args)) as executor:
            futures = [executor.submit(extract_base_features_worker, args) for args in worker_args]
            
            for future in as_completed(futures):
                try:
                    processed = future.result()
                    total_processed += processed
                except Exception as e:
                    log.error(f"Worker process failed: {e}")
        
        # Wait for progress monitor to finish
        monitor_thread.join()
    
    log.info(f"Total processed: {total_processed}/{len(prompts)} prompts")


# Simple version without progress queue for fallback
def extract_base_features_simple(model_name: str, prompts_dir: str, output_dir: str,
                                max_length: int = 2048, resume: bool = True,
                                num_gpus: int = None, save_topk: int = 20,
                                temperature: float = 1.0):
    """Simple version without complex progress tracking."""
    
    # Check available GPUs
    available_gpus = check_gpu_availability()
    
    if not available_gpus:
        log.error("No GPUs available for processing")
        return
    
    # Use specified number of GPUs or all available
    if num_gpus is None:
        num_gpus = len(available_gpus)
    else:
        num_gpus = min(num_gpus, len(available_gpus))
    
    gpu_ids = [gpu_id for gpu_id, _ in available_gpus[:num_gpus]]
    log.info(f"Using {num_gpus} GPUs: {gpu_ids}")
    log.info(f"Extracting top-{save_topk} probabilities with temperature={temperature}")
    
    # Collect all prompts
    prompts = []
    for jsonl_file in pathlib.Path(prompts_dir).rglob("*.jsonl"):
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if 'turns' in record and record['turns']:
                        prompts.append({
                            'id': record['id'],
                            'text': record['turns'][0]
                        })
                except json.JSONDecodeError:
                    continue
    
    log.info(f"Found {len(prompts)} prompts to process")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Split prompts across GPUs
    prompt_chunks = split_prompts_for_gpus(prompts, num_gpus)
    
    def simple_worker(args_tuple):
        """Simplified worker without progress queue."""
        gpu_id, model_name, prompts_chunk, output_dir, max_length, resume, save_topk, temperature = args_tuple
        
        log.info(f"Worker {gpu_id}: Processing {len(prompts_chunk)} prompts on cuda:{gpu_id}")
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch.nn.functional as F
            
            device = f"cuda:{gpu_id}"
            
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map={"": device},
                trust_remote_code=True
            )
            model.eval()
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            processed = 0
            for prompt_data in prompts_chunk:
                prompt_id = prompt_data['id']
                text = prompt_data['text']
                
                save_path = pathlib.Path(output_dir) / f"{prompt_id}.pt"
                
                if resume and save_path.exists():
                    processed += 1
                    continue
                
                try:
                    inputs = tokenizer(
                        text,
                        return_tensors="pt",
                        max_length=max_length,
                        truncation=True,
                        padding=True
                    ).to(device)
                    
                    with torch.no_grad():
                        outputs = model(**inputs, output_hidden_states=True)
                        
                        fusion_indices = get_fusion_indices(model_name, model.config.num_hidden_layers)
                        fused_features = fuse_hidden_states(outputs.hidden_states, fusion_indices)
                        
                        logits = outputs.logits
                        
                        # Apply temperature scaling and get top-k probabilities
                        scaled_logits = logits / temperature
                        probs = F.softmax(scaled_logits, dim=-1)
                        topk_probs, topk_indices = torch.topk(probs, k=save_topk, dim=-1)
                        
                        # ===== 统一维度处理 =====
                        input_ids_squeezed = inputs['input_ids'][0].cpu()
                        attention_mask_squeezed = inputs['attention_mask'][0].cpu()
                        fused_features_squeezed = fused_features.cpu()  # fuse_hidden_states已经去掉batch维度
                        topk_probs_squeezed = topk_probs[0].cpu()
                        topk_indices_squeezed = topk_indices[0].cpu()

                        seq_len = input_ids_squeezed.size(0)
                        assert attention_mask_squeezed.size(0) == seq_len
                        assert fused_features_squeezed.size(0) == seq_len
                        assert topk_probs_squeezed.size(0) == seq_len
                        assert topk_indices_squeezed.size(0) == seq_len
                        
                        # ===== 修复字段命名 =====
                        feature_data = {
                            'hidden_states': fused_features_squeezed,
                            'num_fused_layers': FUSE_LAYERS,
                            'teacher_topk_probs': topk_probs_squeezed,    # 修复字段名
                            'teacher_topk_indices': topk_indices_squeezed,  # 修复字段名
                            'input_ids': input_ids_squeezed,
                            'attention_mask': attention_mask_squeezed,
                            'prompt': text,
                            'prompt_id': prompt_id,
                            'teacher_temperature': temperature,
                            'save_topk': save_topk
                        }
                        
                        torch.save(feature_data, save_path)
                        processed += 1
                        
                        if processed % 100 == 0:
                            torch.cuda.empty_cache()
                            log.info(f"GPU-{gpu_id}: Processed {processed}/{len(prompts_chunk)}")
                        
                except Exception as e:
                    log.error(f"GPU-{gpu_id}: Error processing {prompt_id}: {e}")
            
            log.info(f"GPU-{gpu_id}: Successfully processed {processed}/{len(prompts_chunk)} prompts")
            return processed
            
        except Exception as e:
            log.error(f"GPU-{gpu_id}: Worker failed: {e}")
            return 0
    
    # Set multiprocessing start method
    try:
        torch_mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    # Prepare arguments for workers
    worker_args = []
    for i, chunk in enumerate(prompt_chunks):
        if chunk:
            gpu_id = gpu_ids[i]
            worker_args.append((
                gpu_id, model_name, chunk, output_dir, max_length, resume, save_topk, temperature
            ))
    
    # Run parallel extraction with simple progress
    total_processed = 0
    with ProcessPoolExecutor(max_workers=len(worker_args)) as executor:
        futures = [executor.submit(simple_worker, args) for args in worker_args]
        
        # Show overall progress
        for future in tqdm(as_completed(futures), total=len(worker_args), desc="GPU workers"):
            try:
                processed = future.result()
                total_processed += processed
            except Exception as e:
                log.error(f"Worker process failed: {e}")
    
    log.info(f"Total processed: {total_processed}/{len(prompts)} prompts")


# Keep threading version as alternative
def extract_base_features_threaded(model_name: str, prompts_dir: str, output_dir: str,
                                 max_length: int = 2048, resume: bool = True,
                                 num_threads: int = None, save_topk: int = 20,
                                 temperature: float = 1.0):
    """Extract base model features using threading (good for smaller models)."""
    
    if num_threads is None:
        num_threads = min(4, torch.cuda.device_count())
    
    log.info(f"Using threaded extraction with {num_threads} threads")
    log.info(f"Extracting top-{save_topk} probabilities with temperature={temperature}")
    
    # Load model once (will be shared across threads)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Collect all prompts
    prompts = []
    for jsonl_file in pathlib.Path(prompts_dir).rglob("*.jsonl"):
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if 'turns' in record and record['turns']:
                        prompts.append({
                            'id': record['id'],
                            'text': record['turns'][0]
                        })
                except json.JSONDecodeError:
                    continue
    
    log.info(f"Found {len(prompts)} prompts to process")
    os.makedirs(output_dir, exist_ok=True)
    
    def process_prompt(prompt_data):
        """Process a single prompt."""
        prompt_id = prompt_data['id']
        text = prompt_data['text']
        
        save_path = pathlib.Path(output_dir) / f"{prompt_id}.pt"
        
        if resume and save_path.exists():
            return True
        
        try:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True
            ).to(model.device)
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                
                fusion_indices = get_fusion_indices(model_name, model.config.num_hidden_layers)
                fused_features = fuse_hidden_states(outputs.hidden_states, fusion_indices)
                
                logits = outputs.logits
                
                # Apply temperature scaling and get top-k probabilities
                scaled_logits = logits / temperature
                probs = F.softmax(scaled_logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, k=save_topk, dim=-1)
                
                # ===== 统一维度处理 =====
                input_ids_squeezed = inputs['input_ids'][0].cpu()
                attention_mask_squeezed = inputs['attention_mask'][0].cpu()
                fused_features_squeezed = fused_features.cpu()
                topk_probs_squeezed = topk_probs[0].cpu()
                topk_indices_squeezed = topk_indices[0].cpu()
                
                # ===== 修复字段命名 =====
                feature_data = {
                    'hidden_states': fused_features_squeezed,
                    'num_fused_layers': FUSE_LAYERS,
                    'teacher_topk_probs': topk_probs_squeezed,    # 修复字段名
                    'teacher_topk_indices': topk_indices_squeezed,  # 修复字段名
                    'input_ids': input_ids_squeezed,
                    'attention_mask': attention_mask_squeezed,
                    'prompt': text,
                    'prompt_id': prompt_id,
                    'teacher_temperature': temperature,
                    'save_topk': save_topk
                }
                
                torch.save(feature_data, save_path)
                return True
                
        except Exception as e:
            log.error(f"Error processing {prompt_id}: {e}")
            return False
    
    # Process with thread pool and unified progress bar
    successful = 0
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_prompt, p) for p in prompts]
        
        for future in tqdm(as_completed(futures), total=len(prompts), 
                          desc="Processing prompts"):
            try:
                if future.result():
                    successful += 1
            except Exception as e:
                log.error(f"Thread failed: {e}")
    
    log.info(f"Successfully processed {successful}/{len(prompts)} prompts")


class TeacherFeatureExtractor:
    """Extract logprobs from teacher model via vLLM API."""
    
    def __init__(self, url: str, model_path: str, k: int = 20,
                 concurrency: int = 64, max_tokens: int = 1):
        self.url = url
        self.model_path = model_path
        self.k = k
        self.max_tokens = max_tokens
        self.concurrency = concurrency
        
        self.clients = [
            httpx.AsyncClient(timeout=httpx.Timeout(120.0)) 
            for _ in range(min(concurrency, 8))
        ]
        self.client_semaphore = asyncio.Semaphore(len(self.clients))
        self.request_semaphore = asyncio.Semaphore(concurrency)
    
    async def extract_features(self, prompts_dir: str, output_dir: str, resume: bool = True):
        """Extract teacher features for all prompts with unified progress."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Collect prompts
        prompts = []
        for jsonl_file in pathlib.Path(prompts_dir).rglob("*.jsonl"):
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        if 'turns' in record and record['turns']:
                            prompts.append({
                                'id': record['id'],
                                'text': record['turns'][0]
                            })
                    except json.JSONDecodeError:
                        continue
        
        log.info(f"Extracting teacher features for {len(prompts)} prompts with {self.concurrency} concurrent requests")
        
        # Process all prompts with unified progress bar
        tasks = [self._fetch_one(p['id'], p['text'], output_dir, resume) for p in prompts]
        
        completed = 0
        with tqdm(total=len(tasks), desc="Extracting teacher features") as pbar:
            for fut in asyncio.as_completed(tasks):
                try:
                    await fut
                    completed += 1
                    pbar.update(1)
                except Exception as e:
                    log.error(f"Task failed: {e}")
                    pbar.update(1)
        
        log.info(f"Successfully processed {completed}/{len(prompts)} prompts")
    
    async def _fetch_one(self, prompt_id: str, text: str, output_dir: str, resume: bool):
        """Fetch features for a single prompt."""
        save_path = pathlib.Path(output_dir) / f"{prompt_id}_teacher.pt"
        
        if resume and save_path.exists():
            return
        
        payload = {
            "model": self.model_path,
            "prompt": text,
            "max_tokens": self.max_tokens,
            "temperature": 0.7,
        }
        
        if self.k > 0:
            payload["logprobs"] = self.k
            payload["echo"] = True
        
        try:
            async with self.request_semaphore:
                async with self.client_semaphore:
                    client = self.clients[hash(prompt_id) % len(self.clients)]
                    
                    response = await client.post(self.url, json=payload)
                    response.raise_for_status()
                    data = response.json()
            
            choice = data["choices"][0]
            record = {
                "prompt": text,
                "prompt_id": prompt_id
            }
            
            if "logprobs" in choice:
                lp = choice["logprobs"]
                record.update({
                    "tokens": lp["tokens"],
                    "token_logprobs": lp["token_logprobs"],
                    "top_logprobs": lp.get("top_logprobs", [])
                })
            
            torch.save(record, save_path)
            
        except Exception as e:
            log.error(f"Error fetching features for {prompt_id}: {e}")
            raise
    
    async def close(self):
        for client in self.clients:
            await client.aclose()


def merge_with_teacher_features_parallel(base_dir: str, teacher_dir: str, output_dir: str,
                                       num_workers: int = None):
    """Merge base model hidden states with teacher logprobs using multiple threads."""
    if num_workers is None:
        num_workers = min(8, mp.cpu_count())
    
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    base_files = list(pathlib.Path(base_dir).glob("*.pt"))
    log.info(f"Found {len(base_files)} base feature files")
    log.info(f"Using {num_workers} workers for merging")
    
    def merge_single_file(base_file):
        """Merge a single file."""
        prompt_id = base_file.stem
        teacher_file = pathlib.Path(teacher_dir) / f"{prompt_id}_teacher.pt"
        
        try:
            base_data = torch.load(base_file, map_location='cpu')
            
            if teacher_file.exists():
                teacher_data = torch.load(teacher_file, map_location='cpu')
                base_data['teacher_logprobs'] = teacher_data.get('top_logprobs', [])
                base_data['teacher_tokens'] = teacher_data.get('tokens', [])
                base_data['teacher_token_logprobs'] = teacher_data.get('token_logprobs', [])
                has_teacher = True
            else:
                has_teacher = False
            
            output_file = output_path / f"{prompt_id}.pt"
            torch.save(base_data, output_file)
            
            return True, has_teacher
            
        except Exception as e:
            log.error(f"Error processing {prompt_id}: {e}")
            return False, False
    
    merged_count = 0
    missing_teacher = 0
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(merge_single_file, f): f for f in base_files}
        
        for future in tqdm(as_completed(futures), total=len(base_files), desc="Merging features"):
            try:
                success, has_teacher = future.result()
                if success:
                    merged_count += 1
                    if not has_teacher:
                        missing_teacher += 1
            except Exception as e:
                log.error(f"Merge task failed: {e}")
    
    log.info(f"Merged {merged_count} files. Missing teacher features: {missing_teacher}")


async def run_pipeline(args):
    """Run the complete EAGLE data preparation pipeline."""
    
    base_dir = pathlib.Path(args.output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    if args.stage == "build":
        log.info("=" * 70)
        log.info("Step 1: Building dataset from sources")
        log.info("=" * 70)
        
        build_dataset(
            spec=args.spec,
            output_root=base_dir / "prompts",
            seed=args.seed,
            cn_weight=args.cn_weight,
            stream=args.stream,
            eval_split_ratio=args.eval_split_ratio
        )
        
    elif args.stage == "extract_base":
        log.info("=" * 70)
        log.info("Step 2: Extracting base model hidden states and top-k probabilities")
        log.info("=" * 70)
        
        if args.use_threading:
            extract_base_features_threaded(
                model_name=args.base_model,
                prompts_dir=args.prompts_dir,
                output_dir=args.output_dir,
                max_length=args.max_length,
                resume=args.resume,
                num_threads=args.num_workers,
                save_topk=args.save_topk,
                temperature=args.teacher_temperature
            )
        elif args.simple_mode:
            extract_base_features_simple(
                model_name=args.base_model,
                prompts_dir=args.prompts_dir,
                output_dir=args.output_dir,
                max_length=args.max_length,
                resume=args.resume,
                num_gpus=args.num_gpus,
                save_topk=args.save_topk,
                temperature=args.teacher_temperature
            )
        else:
            extract_base_features_parallel(
                model_name=args.base_model,
                prompts_dir=args.prompts_dir,
                output_dir=args.output_dir,
                max_length=args.max_length,
                resume=args.resume,
                num_gpus=args.num_gpus,
                save_topk=args.save_topk,
                temperature=args.teacher_temperature
            )
        
    elif args.stage == "extract_teacher":
        if not args.teacher_model or not args.teacher_url:
            log.error("Teacher model and URL required for extract_teacher stage")
            return
            
        log.info("=" * 70)
        log.info("Step 3: Extracting teacher model features")
        log.info("=" * 70)
        
        extractor = TeacherFeatureExtractor(
            url=args.teacher_url,
            model_path=args.teacher_model,
            k=args.teacher_k,
            concurrency=args.concurrency,
            max_tokens=args.teacher_max_tokens
        )
        
        await extractor.extract_features(
            prompts_dir=args.prompts_dir,
            output_dir=args.output_dir,
            resume=args.resume
        )
        
        await extractor.close()
        
    elif args.stage == "merge":
        log.info("=" * 70)
        log.info("Step 4: Merging base and teacher features")
        log.info("=" * 70)
        
        merge_with_teacher_features_parallel(
            base_dir=args.base_dir,
            teacher_dir=args.teacher_dir,
            output_dir=args.output_dir,
            num_workers=args.num_workers
        )
        
    else:
        log.error(f"Unknown stage: {args.stage}")
        sys.exit(1)
    
    log.info("Stage completed successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="EAGLE training data preparation pipeline - Fixed multiprocessing version with top-k extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Stage control
    parser.add_argument("--stage", required=True,
                       choices=["build", "extract_base", "extract_teacher", "merge"],
                       help="Pipeline stage to execute")
    
    # Parallelization arguments
    parser.add_argument("--num-gpus", type=int, default=None,
                       help="Number of GPUs to use (auto-detect if not specified)")
    parser.add_argument("--num-workers", type=int, default=None,
                       help="Number of worker threads/processes")
    parser.add_argument("--use-threading", action="store_true",
                       help="Use threading instead of multiprocessing for base extraction")
    parser.add_argument("--simple-mode", action="store_true",
                       help="Use simple multiprocessing without complex progress tracking")
    
    # Dataset building arguments
    parser.add_argument("--spec",
                       help="Dataset specification")
    parser.add_argument("--output-dir", required=True,
                       help="Output directory for current stage")
    parser.add_argument("--seed", type=int, default=2025,
                       help="Random seed for sampling")
    parser.add_argument("--cn-weight", type=float, default=1.0,
                       help="Up-weight Chinese samples")
    parser.add_argument("--stream", action="store_true",
                       help="Use HuggingFace streaming mode")
    parser.add_argument("--eval-split-ratio", type=float, default=0.0,
                       help="Ratio of data to use for evaluation")
    
    # Base model arguments
    parser.add_argument("--base-model",
                       help="Base model name or path")
    parser.add_argument("--prompts-dir",
                       help="Directory containing .jsonl files")
    parser.add_argument("--max-length", type=int, default=2048,
                       help="Maximum sequence length")
    parser.add_argument("--device", default="auto",
                       help="Device placement for base model")
    
    # Teacher model arguments
    parser.add_argument("--teacher-model",
                       help="Teacher model path for vLLM server")
    parser.add_argument("--teacher-url",
                       default="http://localhost:8500/v1/completions",
                       help="vLLM server URL for teacher model")
    parser.add_argument("--teacher-k", type=int, default=20,
                       help="Top-k logprobs to extract from teacher")
    parser.add_argument("--teacher-max-tokens", type=int, default=1,
                       help="Max tokens to generate from teacher")
    parser.add_argument("--concurrency", type=int, default=64,
                       help="Concurrent requests to vLLM server")
    
    # Merge arguments
    parser.add_argument("--base-dir",
                       help="Directory with base features")
    parser.add_argument("--teacher-dir",
                       help="Directory with teacher features")
    
    # Other arguments
    parser.add_argument("--resume", action="store_true",
                       help="Resume from existing files")
    parser.add_argument("--save-topk", type=int, default=20,
                       help="Number of top probabilities to save (default: 20)")
    parser.add_argument("--teacher-temperature", type=float, default=1.0,
                       help="Temperature for teacher distribution (default: 1.0)")
    
    args = parser.parse_args()
    
    try:
        asyncio.run(run_pipeline(args))
    except KeyboardInterrupt:
        log.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        log.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()