import fitz  # PyMuPDF
import re
import os
import logging
from tqdm import tqdm
MODULE_PATH = "/home/guzhouhong/cxz/paper_agent"

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.chdir(MODULE_PATH)


# class ExtractDatasetName:
#     def __init__(self, pdf_path):
#         self.doc = fitz.open(pdf_path)
#         # 预编译正则表达式模式
#         self.dataset_pattern = re.compile(r'\b(dataset(s)?|training data|test set|benchmark)\b', re.IGNORECASE)
#         self.url_pattern = re.compile(r'https?://\S+', re.IGNORECASE) 
    
#     def extract_sentences(self):
#         dataset_sentences = []
#         for page in self.doc:
#             text = page.get_text("text").replace('\n', ' ').strip()
#             # 分割句子并过滤
#             sentences = re.split(r'[.!?]', text)
#             dataset_sentences.extend([
#                 sentence.strip() 
#                 for sentence in sentences 
#                 if sentence.strip() and (self.dataset_pattern.search(sentence) or
#                 self.url_pattern.search(sentence))
#             ])
#         return dataset_sentences
    
#     def __del__(self):
#         self.doc.close()


class ExtractDatasetName:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        try:
            self.doc = fitz.open(pdf_path)
            logger.info(f"成功打开PDF文件: {pdf_path}, 共{len(self.doc)}页")
        except Exception as e:
            logger.error(f"打开PDF文件失败: {str(e)}")
            raise
            
        # 预编译正则表达式模式 - 增强数据集识别能力
        self.dataset_pattern = re.compile(r'\b(dataset(s)?|data(\s+)?set|corpus|benchmark|training\s+data|test\s+set|repository|collection|evaluation\s+data)\b', re.IGNORECASE)
        self.url_pattern = re.compile(r'https?://\S+', re.IGNORECASE)
        self.reference_pattern = re.compile(r'\b(github|huggingface|kaggle|zenodo|figshare|uci|openml)\b', re.IGNORECASE)
    
    def extract_sentences(self, max_sentences=None):
        """提取与数据集相关的句子，增加进度条显示和更多筛选条件"""
        dataset_sentences = []
        
        # 使用tqdm添加进度条
        for page in tqdm(self.doc, desc="处理PDF页面"):
            text = page.get_text("text").replace('\n', ' ').strip()
            # 分割句子
            sentences = re.split(r'[.!?]', text)
            
            # 过滤与数据集相关的句子
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                # 增加对表格和引用部分的识别
                if (self.dataset_pattern.search(sentence) or 
                    self.url_pattern.search(sentence) or 
                    self.reference_pattern.search(sentence)):
                    dataset_sentences.append(sentence)
                    
                # 如果设置了限制则提前返回
                if max_sentences and len(dataset_sentences) >= max_sentences:
                    break
        
        logger.info(f"已提取{len(dataset_sentences)}个相关句子")
        return dataset_sentences
    
    def extract_tables(self):
        """提取PDF中的表格数据"""
        table_data = []
        try:
            for page in tqdm(self.doc, desc="提取表格"):
                # 表格通常包含数据集信息
                tables = page.find_tables()
                if tables and tables.tables:
                    for table in tables.tables:
                        cells = []
                        for row in table.cells:
                            if row:
                                row_text = [cell.text.strip() for cell in row if cell]
                                cells.append(row_text)
                        if cells:
                            table_data.append(cells)
            return table_data
        except Exception as e:
            logger.warning(f"表格提取失败: {str(e)}")
            return []

    def extract_metadata(self):
        """提取PDF的元数据"""
        try:
            metadata = self.doc.metadata
            return {
                "title": metadata.get("title", ""),
                "author": metadata.get("author", ""),
                "subject": metadata.get("subject", ""),
                "keywords": metadata.get("keywords", "")
            }
        except Exception as e:
            logger.warning(f"元数据提取失败: {str(e)}")
            return {}
            
    def __del__(self):
        try:
            if hasattr(self, 'doc') and self.doc:
                self.doc.close()
                logger.debug(f"已关闭PDF文件: {self.pdf_path}")
        except Exception as e:
            logger.error(f"关闭PDF文件失败: {str(e)}")


if __name__ == "__main__":
    # ... existing test code ...
    extractor = ExtractDatasetName("music_vel.pdf")
    dataset_sentences = extractor.extract_sentences()
    print(f"Found {len(dataset_sentences)} dataset-related sentences:")
    for i, sentence in enumerate(dataset_sentences, 1):  # 打印前10个
        print(f"{i}. {sentence}")
    # ... existing print loop ...