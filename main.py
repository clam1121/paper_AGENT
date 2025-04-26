# import os
# MODULE_PATH = "/home/guzhouhong/cxz/paper_agent"
# os.chdir(MODULE_PATH)
# from agent.agent import *
# from model.model import *
# from prompt.get_paper_name import *
# if __name__ == "__main__":
#     llm = Qwen2API()
#     pdf_path = "music_vel.pdf"
#     pdf_process = ExtractDatasetName(pdf_path)
#     dataset_sentences = pdf_process.extract_sentences()
#     text= "\n".join(dataset_sentences)
#     # prompt = f"从以下论文文本中提取实验所用的数据集信息:\n\n{text}\n\n请仅返回数据集的名称即可。"
#     prompt = GET_PAPER_NAME_PROMPT.format(text = text)
#     res, usage = llm.call(prompt)
#     res = res.split("####")[-1]
#     print(res)
#     prompt = GET_DOWNLOAD_URL.format(text = res, text_1 = text )
#     res ,usage = llm.call(prompt)
#     print(res)
#     print(res)
#     # print(usage)
#     # print(type(usage))

import os
import sys
import argparse
import logging
import json
from typing import Dict, Tuple, List, Optional, Any
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 获取模块路径
MODULE_PATH = os.path.dirname(os.path.abspath(__file__))
os.chdir(MODULE_PATH)
sys.path.append(MODULE_PATH)

# 导入所需模块
from agent.agent import ExtractDatasetName
from model.model import Qwen2API, PaperAnalyzer
from prompt.get_paper_name import GET_PAPER_NAME_PROMPT, GET_DOWNLOAD_URL
from tool.dataset_downloader import DatasetDownloader

def download_datasets(dataset_info: Dict[str, Tuple[str, str]], download_dir: str = "datasets") -> Dict[str, str]:
    """使用DatasetDownloader下载数据集
    
    Args:
        dataset_info: 数据集信息字典，格式为 {"数据集名称": ("平台", "URL")}
        download_dir: 下载目录
    
    Returns:
        下载结果字典，格式为 {"数据集名称": "结果消息"}
    """
    if not dataset_info:
        logger.warning("没有数据集信息可供下载")
        return {}
    
    logger.info(f"准备下载{len(dataset_info)}个数据集到目录: {download_dir}")
    downloader = DatasetDownloader(download_dir=download_dir)
    
    results = {}
    for name, info in dataset_info.items():
        logger.info(f"下载数据集: {name}")
        try:
            # 检查是否已经在下载历史中
            if name in downloader.history and os.path.exists(downloader.history[name].get("path", "")):
                logger.info(f"数据集 {name} 已存在于下载历史，跳过下载")
                results[name] = f"已存在: {downloader.history[name]['path']}"
                continue
                
            result = downloader.download(info)
            results[name] = result
        except Exception as e:
            logger.error(f"下载 {name} 时出错: {str(e)}")
            results[name] = f"下载失败: {str(e)}"
    
    return results

def process_pdf(pdf_path: str, download: bool = False, download_dir: str = "datasets", verbose: bool = False) -> Tuple[str, Dict[str, Any]]:
    """处理单个PDF文件，提取数据集信息并可选下载
    
    Args:
        pdf_path: PDF文件路径
        download: 是否下载数据集
        download_dir: 数据集下载目录
        verbose: 是否显示详细日志
    
    Returns:
        数据集名称和下载信息元组
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f"处理PDF: {pdf_path}")
    
    # 1. 初始化LLM客户端
    llm = Qwen2API()
    
    # 2. 创建论文分析器
    analyzer = PaperAnalyzer(pdf_path, llm)
    
    # 3. 提取数据集名称
    dataset_names = analyzer.extract_dataset_names()
    logger.info(f"发现数据集: {dataset_names}")
    
    # 4. 提取与数据集相关的句子作为上下文
    extractor = ExtractDatasetName(pdf_path)
    dataset_sentences = extractor.extract_sentences()
    context = "\n".join(dataset_sentences)
    
    # 5. 获取数据集下载信息
    download_info = analyzer.get_dataset_download_info(dataset_names, context)
    
    # 将下载信息存储到结果中
    download_results = {}
    
    if download_info:
        logger.info(f"数据集下载信息:")
        for name, info in download_info.items():
            logger.info(f"  {name}: {info}")
        
        # 6. 可选：下载数据集
        if download:
            logger.info("开始下载数据集...")
            download_results = download_datasets(download_info, download_dir)
            
            logger.info("下载结果:")
            for name, result in download_results.items():
                logger.info(f"  {name}: {result}")
    else:
        logger.warning("未找到数据集下载信息")
    
    return dataset_names, {"download_info": download_info, "download_results": download_results}

def process_directory(dir_path: str, download: bool = False, download_dir: str = "datasets", verbose: bool = False) -> Dict[str, Dict]:
    """处理目录下的所有PDF文件
    
    Args:
        dir_path: 目录路径
        download: 是否下载数据集
        download_dir: 数据集下载目录
        verbose: 是否显示详细日志
    
    Returns:
        处理结果字典
    """
    results = {}
    pdf_files = list(Path(dir_path).glob("*.pdf"))
    
    if not pdf_files:
        logger.warning(f"目录中未找到PDF文件: {dir_path}")
        return {}
    
    logger.info(f"找到{len(pdf_files)}个PDF文件")
    for pdf_file in pdf_files:
        try:
            logger.info(f"处理: {pdf_file.name}")
            dataset_names, info = process_pdf(str(pdf_file), download, download_dir, verbose)
            results[pdf_file.name] = {
                "dataset_names": dataset_names,
                "download_info": info["download_info"],
                "download_results": info["download_results"]
            }
        except Exception as e:
            logger.error(f"处理 {pdf_file.name} 失败: {str(e)}")
            results[pdf_file.name] = {"error": str(e)}
    
    return results

def save_results(results: Dict, output_path: str):
    """保存结果到JSON文件
    
    Args:
        results: 处理结果字典
        output_path: 输出文件路径
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"结果已保存到: {output_path}")
    except Exception as e:
        logger.error(f"保存结果失败: {str(e)}")

def main():
    """主函数：解析命令行参数并处理PDF文件"""
    parser = argparse.ArgumentParser(description="论文数据集提取与下载工具")
    parser.add_argument("path", help="PDF文件或包含PDF文件的目录路径")
    parser.add_argument("--download", "-d", action="store_true", help="自动下载发现的数据集")
    parser.add_argument("--download-dir", type=str, default="datasets", help="数据集下载目录")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细日志")
    parser.add_argument("--output", "-o", help="将结果保存到JSON文件")
    parser.add_argument("--batch", "-b", action="store_true", help="批处理模式，处理目录下所有PDF")
    
    args = parser.parse_args()
    
    # 检查路径是否存在
    if not os.path.exists(args.path):
        logger.error(f"路径不存在: {args.path}")
        return 1
    
    try:
        results = {}
        
        # 判断是处理单个文件还是目录
        if os.path.isdir(args.path) or args.batch:
            # 处理目录
            logger.info(f"批处理目录: {args.path}")
            results = process_directory(args.path, args.download, args.download_dir, args.verbose)
        else:
            # 处理单个PDF文件
            pdf_path = args.path
            if not pdf_path.lower().endswith('.pdf'):
                logger.error(f"文件不是PDF格式: {pdf_path}")
                return 1
                
            dataset_names, info = process_pdf(pdf_path, args.download, args.download_dir, args.verbose)
            results = {
                "pdf": pdf_path,
                "dataset_names": dataset_names,
                "download_info": info["download_info"],
                "download_results": info["download_results"]
            }
        
        # 保存结果到JSON文件
        if args.output:
            save_results(results, args.output)
            
        return 0
        
    except Exception as e:
        logger.error(f"处理过程中出错: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    # 如果直接运行，使用默认参数
    if len(sys.argv) == 1:
        # 使用默认示例PDF文件
        pdf_files = [f for f in os.listdir(".") if f.endswith(".pdf")]
        if pdf_files:
            default_pdf = pdf_files[1]  # 使用第一个找到的PDF文件
            logger.info(f"未提供参数，使用默认PDF文件: {default_pdf}")
            print(pdf_files)
            
            # 直接调用process_pdf函数处理并下载
            dataset_names, info = process_pdf(default_pdf, download=True)
            
            print(f"\n处理结果摘要:")
            print(f"PDF文件: {default_pdf}")
            print(f"提取的数据集: {dataset_names}")
            
            if info["download_info"]:
                print("\n下载信息:")
                for name, dl_info in info["download_info"].items():
                    print(f"  {name}: {dl_info}")
            
            if info["download_results"]:
                print("\n下载结果:")
                for name, result in info["download_results"].items():
                    print(f"  {name}: {result}")
        else:
            print("未找到PDF文件，请指定文件路径")
            parser = argparse.ArgumentParser(description="论文数据集提取与下载工具")
            parser.print_help()
    else:
        # 使用命令行参数
        sys.exit(main())