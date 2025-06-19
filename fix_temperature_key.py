#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fix_temperature_key.py - 修复特征文件中的temperature键名
将'temperature'键重命名为'teacher_temperature'，以修复KeyError问题
"""

import os
import pathlib
import torch
import logging
from tqdm import tqdm
import concurrent.futures
from typing import List

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

def fix_file(file_path: str) -> bool:
    """修复单个文件的键名"""
    try:
        # 加载文件
        data = torch.load(file_path)
        
        # 检查是否已经有teacher_temperature
        if 'teacher_temperature' in data:
            return True  # 已经修复，无需操作
        
        # 检查是否有temperature
        if 'temperature' in data:
            # 重命名键
            data['teacher_temperature'] = data['temperature']
            # 删除旧键
            del data['temperature']
            # 保存修改后的文件
            torch.save(data, file_path)
            return True
        else:
            # 文件没有temperature键，需要设置默认值
            data['teacher_temperature'] = 0.0  # 默认值为0.0
            torch.save(data, file_path)
            logger.warning(f"文件 {file_path} 没有temperature键，已添加默认值")
            return True
        
    except Exception as e:
        logger.error(f"处理文件 {file_path} 时出错: {e}")
        return False

def fix_directory(directory: str, num_workers: int = 8) -> None:
    """修复目录中的所有特征文件"""
    # 查找所有.pt文件
    path = pathlib.Path(directory)
    pt_files = list(path.rglob("*.pt"))
    
    if not pt_files:
        logger.warning(f"目录 {directory} 中没有找到.pt文件")
        return
    
    logger.info(f"在目录 {directory} 中找到 {len(pt_files)} 个文件等待处理")
    
    # 使用线程池并行处理文件
    successful = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(fix_file, str(f)) for f in pt_files]
        
        # 使用tqdm显示进度
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(pt_files)):
            try:
                if future.result():
                    successful += 1
            except Exception as e:
                logger.error(f"处理文件时出错: {e}")
    
    logger.info(f"成功处理了 {successful}/{len(pt_files)} 个文件")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="修复特征文件中的temperature键名")
    parser.add_argument("--directories", nargs="+", required=True, 
                      help="要处理的目录，可以指定多个")
    parser.add_argument("--workers", type=int, default=8, 
                      help="并行工作线程数量")
    
    args = parser.parse_args()
    
    for directory in args.directories:
        logger.info(f"开始处理目录: {directory}")
        fix_directory(directory, args.workers)
    
    logger.info("所有目录处理完成")

if __name__ == "__main__":
    main() 