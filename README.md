# 自动下载论文相关数据集的Agent

## 简介

这个Agent旨在自动从PDF论文中提取与数据集相关的信息，并下载相应的数据集。它结合了PDF文本提取、自然语言处理和多种数据集下载方法，帮助用户快速获取论文中提到的数据集。

## 功能

- 从PDF中提取与数据集相关的句子和信息。
- 使用LLM（大语言模型）分析提取的数据集名称。
- 自动识别数据集的下载链接。
- 支持从HuggingFace、Git、Kaggle等平台下载数据集。
- 记录下载历史，避免重复下载。

## 文件结构

- `agent/agent.py`: 负责从PDF中提取与数据集相关的句子。
- `model/model.py`: 包含LLM客户端类，用于调用大语言模型API。
- `prompt/get_paper_name.py`: 包含用于生成提取数据集名称的提示。
- `tool/dataset_downloader.py`: 提供多种数据集下载方法。
- `main.py`: 主程序入口，处理命令行参数并执行数据集提取和下载。

## 使用方法

1. 确保安装了所有依赖项。
2. 运行`main.py`，并提供PDF文件或包含PDF文件的目录路径。
3. 使用`--download`选项自动下载发现的数据集。
4. 使用`--output`选项将结果保存到JSON文件。

## 依赖项

- `fitz` (PyMuPDF): 用于PDF文本提取。
- `requests`: 用于HTTP请求。
- `tqdm`: 用于显示进度条。
- `GitPython`: 用于从Git仓库下载数据集。
- `kaggle`: 用于从Kaggle下载数据集。
- `datasets`: 用于从HuggingFace下载数据集。

## 注意事项

- 确保在运行前配置好API密钥和相关设置。
- 下载数据集时，请确保网络连接正常。

## 贡献

欢迎提交问题和贡献代码！请通过GitHub提交Pull Request或Issue。
