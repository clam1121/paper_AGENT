import fitz  # PyMuPDF
import re
import json
import requests
import os
import sys
import logging
from typing import Tuple, Dict, Any, Optional
import time

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 获取模块路径
MODULE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(MODULE_PATH)  # 添加项目根目录到Python路径

from prompt.get_paper_name import (
    GET_PAPER_NAME_PROMPT,
    GET_DOWNLOAD_URL
)

class LLMClient:
    """基础LLM客户端类，可以扩展支持不同的模型"""
    def __init__(self):
        pass
        
    def call(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """调用LLM API"""
        raise NotImplementedError("子类必须实现call方法")

    

class Qwen2API(LLMClient):
    
    def __init__(self, api_key="your_api_key", 
                engine_name="chatgpt-4o-latest", max_retries=3, retry_delay=2):
        super().__init__()
        self.api_key = api_key
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # 解析模型名称和温度
        if "#" in engine_name:
            temp = engine_name.split("#")
            self.engine_name = temp[0]
            self.temperature = float(temp[1])
        else:
            self.engine_name = engine_name
            self.temperature = 0.7  # 默认温度

    def call(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """调用API并处理重试逻辑"""
        params = {
            "messages": [{"role": "user", "content": prompt}],
            "model": self.engine_name,
            "temperature": self.temperature
        }

        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"调用API，尝试 {attempt+1}/{self.max_retries}")
                response = requests.post(
                    "your_api_url",
                    headers=headers,
                    json=params,
                    stream=False,
                    timeout=30  # 添加超时设置
                )
                
                if response.status_code != 200:
                    logger.warning(f"API返回非200状态码: {response.status_code}, {response.text}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        return f"API调用失败: {response.status_code}", {"error": response.text}
                
                res = response.json()
                message = res["choices"][0]["message"]["content"]
                usage = res["usage"]
                
                return message, usage
                
            except Exception as e:
                logger.error(f"API调用异常: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    return f"API调用异常: {str(e)}", {"error": str(e)}

    def parse_dataset_response(self, response: str) -> Dict[str, Tuple[str, str]]:
        """解析API返回的数据集信息"""
        try:
            # 提取格式化部分
            if "####" in response:
                print("*"*30)
                print(response)
                print("*"*30)
                response = response.split("####")[-2].strip()
                print("*"*30)
                print(response)
                print("*"*30)
            
            # 尝试解析JSON字典
            return json.loads(response)
        except Exception as e:
            logger.error(f"解析数据集响应失败: {str(e)}")
            logger.debug(f"原始响应: {response}")
            return {}

class PaperAnalyzer:
    """论文分析器类，整合PDF提取和LLM分析"""
    def __init__(self, pdf_path: str, llm_client: Optional[LLMClient] = None):
        self.pdf_path = pdf_path
        self.llm_client = llm_client or Qwen2API()
        
    def extract_dataset_names(self) -> str:
        """从PDF提取数据集名称"""
        from agent.agent import ExtractDatasetName
        
        try:
            extractor = ExtractDatasetName(self.pdf_path)
            dataset_sentences = extractor.extract_sentences()
            text = "\n".join(dataset_sentences)
            
            prompt = GET_PAPER_NAME_PROMPT.format(text=text)
            response, _ = self.llm_client.call(prompt)
            
            # 提取格式化部分
            if "####" in response:
                response = response.split("####")[-1].strip()
                
            logger.info(f"提取的数据集名称: {response}")
            return response
        except Exception as e:
            logger.error(f"提取数据集名称失败: {str(e)}")
            return f"错误: {str(e)}"
            
    def get_dataset_download_info(self, dataset_names: str, context_text: str) -> Dict[str, Tuple[str, str]]:
        """获取数据集下载信息"""
        try:
            prompt = GET_DOWNLOAD_URL.format(text=dataset_names, text_1=context_text)
            response, _ = self.llm_client.call(prompt)
            
            # 解析响应
            if isinstance(self.llm_client, Qwen2API):
                return self.llm_client.parse_dataset_response(response)
            else:
                # 简单解析
                if "####" in response:
                    response = response.split("####")[1].strip()
                try:
                    return json.loads(response)
                except:
                    logger.error("无法解析下载信息JSON")
                    return {}
        except Exception as e:
            logger.error(f"获取下载信息失败: {str(e)}")
            return {}


if __name__ == "__main__":
    text = """1. The Music Maestro or The Musically Challenged, A Massive Music Evaluation Benchmark for Large Language Models Jiajia Li1,2, Lu Yang3, Mingni Tang3, Cong Chen4, Zuchao Li3,∗, Ping Wang1,2,∗, Hai Zhao5 1School of Information Management, Wuhan University, Wuhan, China 2Key Laboratory of Archival Intelligent Development and Service, NAAC 3School of Computer Science, Wuhan University, Wuhan, China 4School of Music, Shenyang Conservatory of Music, Shenyang, China 5Department of Computer Science and Engineering, Shanghai Jiao Tong University {cantata, yang_lu, minnie-tang, zcli-charlie, wangping}@whu
2. cn Abstract Benchmark plays a pivotal role in assessing the advancements of large language models (LLMs)
3. To address this gap, we present ZIQI-Eval, a com- prehensive and large-scale music benchmark specifically designed to evaluate the music- related capabilities of LLMs
4. The dataset is available at GitHub1 and HuggingFace2
5. Benchmark evaluation has played a crucial role in assessing and quantifying the performance of LLMs across different domains
6. 1https://github
7. com/zcli-charlie/ZIQI-Eval 2https://huggingface
8. co/datasets/MYTH-Lab/ZIQI-Eval ing (Austin et al
9. Therefore, we present ZIQI-Eval, an extensive and comprehensive music benchmark specifically crafted to assess the music-related abilities of LLMs
10. In addition, this music benchmark actively contributes to the recognition of female music composers
11. Therefore, we pro- pose ZIQI-Eval benchmark, a manually curated, large-scale, and comprehensive benchmark for eval- uating music-related capabilities
12. (2023) extracts music-related information from an open-source music dataset and uses instruction- tuning to instruct their proposed model LLark to do music understanding, music captioning, and music reasoning
13. Benchmark Evaluations Benchmark evaluation plays a crucial role in assessing the development of LLMs
14. Therefore, we pro- pose ZIQI-EVAL, a benchmark for evaluating the musical abilities of LLMs, to fill the gap in bench- mark evaluations of LLMs’ musical capabilities
15. 3 ZIQI-Eval Benchmark 3
16. 1 Dataset Curation General Principle This dataset integrates the renowned music literature database Répertoire In- ternational de Littérature Musicale (RILM), provid- ing a broad research perspective and profound aca- demic insights
17. The inclusion of "The New Grove Dictionary of Music and Musicians" injects the essence of musical humanism into the dataset
18. , 2023), collectively enhance the data integrity and reliability of the benchmark
19. Data Statistics ZIQI-Eval dataset consists of two parts: music comprehension question bank and music generation question bank
20. Addi- tionally, the dataset adopts a decentralized design philosophy, fully showcasing the diversity and in- clusiveness of global music cultures
21. We conduct a comprehensive evaluation of LLMs’ music capabilities across the entire dataset
22. It is worth mentioning that this music dataset has made positive contributions in highlighting female music composers
23. This initiative not only reflects the benchmark’s profound recognition of gender equal- ity issues but also demonstrates its efforts in ad- vancing the diversification of the music field
24. Overall, the performance of all LLMs on the ZIQI-Eval benchmark is poor
25. This glaring discrepancy highlights the inadequate consideration given to music abilities within current LLM models and un- derscores the formidable challenges posed by the ZIQI-Eval benchmark
26. 5 Analysis In addition to the overall evaluation of LLMs on the dataset, we are also interested in the models’ ability for specific categories
27. To address this gap, we intro- duce ZIQI-Eval, a comprehensive benchmark that encompasses 10 major categories and 56 subcate- gories, comprising over 14,000 data entries
28. No- tably, this benchmark also actively contributes to the acknowledgment of female music composers, rectifying the gender disparity and promoting inclu- sivity
29. We intend to create a multimodal benchmark to evaluate the musical expertise of LLMs in the future
30. One limitation of our current music benchmark is the absence of multi- modal data
31. While the benchmark may excel in evaluating and comparing the quality and creativ- ity of musical compositions based on audio data alone, it fails to incorporate other essential aspects of the music experience, such as visual elements or textual information
32. Lawbench: Benchmark- ing legal knowledge of large language models
33. Measuring mathematical problem solving with the math dataset
34. MultiSpanQA: A dataset for multi-span question answering
35. https://github
36. Marble: Music audio representation benchmark for universal evalua- tion
37. Arcmmlu: A library and information science benchmark for large language models"""
    # prompt = f"从以下论文文本中提取实验所用的数据集信息:\n\n{text}\n\n请仅返回数据集的名称即可。"
    prompt = GET_PAPER_NAME_PROMPT.format(text = text)
    agent = Qwen2API()
    message, usage = agent.call(prompt)
    message = message.split("####")[-1]
    print(message)
    prompt = GET_DOWNLOAD_URL.format(text= message, text_1 = text)
    message, usage = agent.call(prompt)

    print(message)
    print(usage)