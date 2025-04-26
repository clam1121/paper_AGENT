import os
import logging
import json
from typing import Dict, Tuple, List, Union, Optional
from tqdm import tqdm
import requests
import importlib.util

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetDownloader:
    def __init__(self, download_dir="datasets"):
        self.download_dir = download_dir
        self.dataset_mapping = {
            "HumanEval": ("huggingface", "openai/human-eval"),
            "HotPotQA": ("huggingface", "hotpot_qa"),
            "MBPP": ("huggingface", "mbpp"),
            "AlfWorld": ("git", "https://github.com/alfworld/alfworld.git"),
            "WebShop": ("git", "https://github.com/princeton-nlp/WebShop"),
            # 可以添加更多预设的数据集映射
        }
        os.makedirs(self.download_dir, exist_ok=True)
        
        # 记录下载历史
        self.history_file = os.path.join(self.download_dir, "download_history.json")
        self.load_history()

    def load_history(self):
        """加载下载历史"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    self.history = json.load(f)
            except Exception as e:
                logger.error(f"加载历史记录失败: {str(e)}")
                self.history = {}
        else:
            self.history = {}

    def save_history(self):
        """保存下载历史"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            logger.error(f"保存历史记录失败: {str(e)}")

    def check_dependencies(self) -> List[str]:
        """检查并返回缺少的依赖项"""
        missing = []
        dependencies = {
            "datasets": "datasets",
            "git": "GitPython",
            "kaggle": "kaggle"
        }
        
        for module, package in dependencies.items():
            if importlib.util.find_spec(module) is None:
                missing.append(package)
                
        return missing

    def install_dependencies(self, packages: List[str]):
        """安装缺少的依赖项"""
        if not packages:
            return
            
        try:
            import pip
            for package in packages:
                logger.info(f"安装依赖: {package}")
                pip.main(['install', package])
        except Exception as e:
            logger.error(f"安装依赖失败: {str(e)}")
            logger.info(f"请手动安装以下依赖: {', '.join(packages)}")

    def download_from_huggingface(self, dataset_path: str) -> str:
        """从HuggingFace或镜像站点下载数据集"""
        try:
            # 设置环境变量使用镜像站点
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            
            # 处理完整 URL 的情况
            if dataset_path.startswith(("http://", "https://")):
                # 从 URL 提取数据集 ID
                if "huggingface.co/datasets/" in dataset_path:
                    dataset_path = dataset_path.split("huggingface.co/datasets/")[-1]
                elif "hf-mirror.com/datasets/" in dataset_path:
                    dataset_path = dataset_path.split("hf-mirror.com/datasets/")[-1]
                else:
                    logger.warning(f"无法从URL提取数据集ID: {dataset_path}")
                    # 尝试使用最后一部分作为数据集ID
                    dataset_path = dataset_path.rstrip("/").split("/")[-1]
            
            # 首先尝试使用huggingface_hub的snapshot_download功能
            try:
                logger.info(f"尝试从HF镜像站点下载数据集: {dataset_path}")
                from huggingface_hub import snapshot_download
                
                # 确定本地保存路径
                save_path = os.path.join(self.download_dir, dataset_path.replace("/", "_"))
                os.makedirs(save_path, exist_ok=True)
                
                # 使用镜像站点下载
                with tqdm(desc=f"从HF镜像下载 {dataset_path}") as pbar:
                    def progress_callback(progress):
                        pbar.update(progress - pbar.n)
                    
                    snapshot_download(
                        repo_id=dataset_path,
                        repo_type="dataset",
                        local_dir=save_path,
                        local_dir_use_symlinks=False,
                        resume_download=True,
                        max_workers=4,  # 增加并行下载的工作线程数
                        tqdm_class=lambda *args, **kwargs: pbar  # 使用现有进度条
                    )
                
                # 更新历史
                self.history[dataset_path] = {
                    "source": "huggingface_mirror",
                    "path": save_path,
                    "date": self._get_current_timestamp()
                }
                self.save_history()
                
                return f"数据集已下载至: {save_path}"
                
            except Exception as e:
                logger.warning(f"使用HF镜像直接下载失败: {str(e)}，尝试使用datasets库")
                
                # 如果直接下载失败，尝试使用datasets库
                import datasets
                
                # 规范化数据集名称
                normalized_name = dataset_path.replace("-", "_").lower()
                logger.info(f"使用datasets库从镜像下载数据集: {normalized_name}")
                
                # 创建进度条
                with tqdm(desc=f"下载 {normalized_name}", unit="MB") as pbar:
                    def progress_callback(dl_size, total_size):
                        if total_size:
                            pbar.total = total_size / 1024 / 1024
                        pbar.update((dl_size / 1024 / 1024) - pbar.n)
                    
                    # 下载数据集
                    try:
                        dataset = datasets.load_dataset(normalized_name, download_mode="force_redownload")
                    except Exception as inner_error:
                        # 尝试使用原始路径
                        logger.warning(f"使用规范化路径失败: {str(inner_error)}，尝试使用原始路径")
                        dataset = datasets.load_dataset(dataset_path, download_mode="force_redownload")
                    
                    save_path = os.path.join(self.download_dir, normalized_name.split('/')[-1])
                    dataset.save_to_disk(save_path)
                    
                    # 更新历史
                    self.history[normalized_name] = {
                        "source": "huggingface",
                        "path": save_path,
                        "date": dataset.info.download_timestamp
                    }
                    self.save_history()
                    
                    return f"数据集已保存至 {save_path}"
                    
        except Exception as e:
            logger.error(f"HuggingFace下载失败: {str(e)}")
            return f"HuggingFace下载失败: {str(e)}" 

    def download_from_git(self, repo_url: str) -> str:
        """从Git仓库克隆数据集（简化版，无进度报告）"""
        try:
            # 检查GitPython是否已安装
            missing = self.check_dependencies()
            if "GitPython" in missing:
                self.install_dependencies(["GitPython"])
                
            from git import Repo
            repo_name = repo_url.split("/")[-1].replace(".git", "")
            clone_path = os.path.join(self.download_dir, repo_name)
            
            if os.path.exists(clone_path):
                logger.info(f"更新已有仓库: {repo_name}")
                repo = Repo(clone_path)
                origin = repo.remotes.origin
                origin.pull()
                msg = f"仓库已更新: {clone_path}"
            else:
                logger.info(f"克隆新仓库: {repo_url}")
                # 禁用进度报告
                Repo.clone_from(repo_url, clone_path)
                msg = f"仓库已克隆至: {clone_path}"
            
            # 更新历史
            self.history[repo_name] = {
                "source": "git",
                "path": clone_path,
                "url": repo_url,
                "date": self._get_current_timestamp()
            }
            self.save_history()
            
            return msg
        except Exception as e:
            logger.error(f"Git操作失败: {str(e)}")
            return f"Git操作失败: {str(e)}"

    def download_from_kaggle(self, dataset_identifier: str) -> str:
        """从Kaggle下载数据集"""
        try:
            missing = self.check_dependencies()
            if "kaggle" in missing:
                self.install_dependencies(["kaggle"])
                
            import kaggle
            dataset_path = os.path.join(self.download_dir, "kaggle", dataset_identifier.replace("/", "_"))
            os.makedirs(dataset_path, exist_ok=True)
            
            logger.info(f"从Kaggle下载数据集: {dataset_identifier}")
            kaggle.api.dataset_download_files(
                dataset_identifier, 
                path=dataset_path, 
                unzip=True
            )
            
            # 更新历史
            self.history[dataset_identifier] = {
                "source": "kaggle",
                "path": dataset_path,
                "date": self._get_current_timestamp()
            }
            self.save_history()
            
            return f"Kaggle数据集已下载至: {dataset_path}"
        except Exception as e:
            logger.error(f"Kaggle下载失败: {str(e)}")
            return f"Kaggle下载失败: {str(e)}"

    def download_from_url(self, url: str, filename: Optional[str] = None) -> str:
        """通用URL下载方法"""
        try:
            if not filename:
                filename = url.split("/")[-1]
                
            save_path = os.path.join(self.download_dir, filename)
            logger.info(f"从URL下载文件: {url} -> {save_path}")
            
            # 流式下载并显示进度
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(save_path, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            # 更新历史
            self.history[filename] = {
                "source": "url",
                "path": save_path,
                "url": url,
                "date": self._get_current_timestamp()
            }
            self.save_history()
            
            return f"文件已下载至: {save_path}"
        except Exception as e:
            logger.error(f"URL下载失败: {str(e)}")
            return f"URL下载失败: {str(e)}"

    def download(self, dataset_info: Union[str,  List[str]]) -> str:
        """增强版下载方法，支持多种格式
    
        Args:
            dataset_info: 数据集信息，可以是以下格式：
                - 字符串：数据集名称或URL
                - 元组：(source, path)
                - 列表：[source, path]
        
        Returns:
            下载结果描述
        """
        # 检查依赖
        missing = self.check_dependencies()
        if missing:
            self.install_dependencies(missing)
        
        # 处理元组格式 (source, path)
        if isinstance(dataset_info, tuple) and len(dataset_info) == 2:
            source, path = dataset_info
            return self._process_by_source(source, path)
        
        # 处理列表格式 [source, path]
        if isinstance(dataset_info, list) and len(dataset_info) == 2:
            source, path = dataset_info
            return self._process_by_source(source, path)
        
        # 处理字符串格式
        if isinstance(dataset_info, str):
            # 检查预设映射
            if dataset_info in self.dataset_mapping:
                return self.download(self.dataset_mapping[dataset_info])
            
            # 自动识别格式
            if "/" in dataset_info and not dataset_info.startswith(("http://", "https://")):
                # 可能是huggingface格式
                if "huggingface.co" in dataset_info:
                    path = dataset_info.split("huggingface.co/datasets/")[-1]
                    return self.download_from_huggingface(path)
                else:
                    return self.download_from_huggingface(dataset_info)
            elif dataset_info.startswith(("http://", "https://")):
                # URL链接
                return self.download_from_url(dataset_info)
            elif "github.com" in dataset_info.lower():
                # GitHub链接
                if not dataset_info.endswith(".git"):
                    dataset_info += ".git"
                return self.download_from_git(dataset_info)
        
        return f"未识别的数据集格式，请手动处理: {dataset_info}"

    def _process_by_source(self, source: str, path: str) -> str:
        """根据来源类型处理下载
        
        Args:
            source: 来源类型
            path: 路径或URL
        
        Returns:
            下载结果
        """
        source = source.lower()
        
        # 标准来源类型
        if source == "huggingface":
            return self.download_from_huggingface(path)
        elif source == "git" or source == "github":
            return self.download_from_git(path)
        elif source == "kaggle":
            return self.download_from_kaggle(path)
        elif source in ["url", "official", "官方网站", "官方出版物", "官方数据库"]:
            return self.download_from_url(path)
        
        # 对于不支持自动下载的数据集类型，返回信息
        if "官方" in source or "订阅" in source or "非公开" in source:
            return f"需要手动获取的数据集: {source} - {path}"
        
        # 默认尝试作为URL处理
        logger.warning(f"未知来源类型 '{source}'，尝试作为URL处理")
        return self.download_from_url(path)

    def download_multiple(self, dataset_dict: Dict[str, Tuple[str, str]]) -> Dict[str, str]:
        """批量下载多个数据集
    
        Args:
            dataset_dict: 数据集字典，格式为 {"数据集名称": (source, path)} 或 {"数据集名称": [source, path]}
        
        Returns:
            下载结果字典
        """
        results = {}
        if not dataset_dict:
            return {"error": "空数据集字典"}
            
        for name, info in dataset_dict.items():
            logger.info(f"下载数据集: {name}")
            try:
                result = self.download(info)
                results[name] = result
            except Exception as e:
                logger.error(f"下载 {name} 时出错: {str(e)}")
                results[name] = f"下载失败: {str(e)}"
            
        return results
    def _get_current_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()