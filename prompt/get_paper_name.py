GET_PAPER_NAME_PROMPT = """
从以下论文文本中提取实验所用的数据集信息:\n{text}\n请仅返回数据集的名称即可。
格式要求如下：
#### name: xxx,xxx,xxx,xxx
"""

# GET_DOWNLOAD_URL = """
# 请根据我给出的数据集名称和上下文信息，汇总他们的下载方式和下载url，
# 数据名称为:\n{text}\n,
# 上下文信息：\n{text_1}\n,
# 参考的输出格式如下(是python中的字典格式):
# ####
# {{
#     \"HumanEval\": (\"huggingface\", \"openai/human-eval\"),
#     \"AlfWorld\": (\"git\", \"https://github.com/alfworld/alfworld.git\"),
#     \"WebShop\": (\"git\", \"https://github.com/princeton-nlp/WebShop\")
# }}
# ####
# """

# 增强提示模板

# GET_PAPER_NAME_PROMPT = """
# 请通过分析以下来自学术论文的文本片段，提取该论文中提到的所有数据集名称：

# {text}

# 请注意以下几点：
# 1. 只需提取确切的数据集名称，不要包含描述性文字
# 2. 区分数据集名称和模型名称（我们只需要数据集）
# 3. 对于有名字缩写的数据集，同时提供完整名称和缩写
# 4. 如果有多个数据集，使用逗号分隔

# 格式要求如下：
# #### name: xxx,xxx,xxx,xxx
# """

GET_DOWNLOAD_URL = """
请帮我找出以下数据集的下载方式和URL链接。我提供的信息包括数据集名称和相关上下文。

数据集名称：
{text}

相关上下文内容：
{text_1}

对于每个数据集，请：
1. 确定其主要托管平台（如Huggingface、Github、Kaggle等）
2. 提供完整的下载URL
3. 如果上下文中明确提到了下载链接，请优先使用该链接
4. 如果没有明确链接，请提供该数据集最官方、最可靠的来源

请以有效的JSON格式返回结果，格式为含双引号的JSON字典：
####
{{
    "数据集名称1": ["平台", "URL"],
    "数据集名称2": ["平台", "URL"],
    "数据集名称3": ["平台", "URL"]
}}
####

例如：
####
{{
    "HumanEval": ["huggingface", "openai/human-eval"],
    "AlfWorld": ["git", "https://github.com/alfworld/alfworld.git"],
    "WebShop": ["git", "https://github.com/princeton-nlp/WebShop"]
}}
####

注意：请确保所有键和值都使用双引号，以确保返回有效的JSON格式。
"""

DATASET_DESCRIPTION_PROMPT = """
请基于以下来自学术论文的文本片段，为每个提取出的数据集提供简短描述：

数据集名称：
{dataset_names}

论文上下文：
{context}

请提供以下信息（如果能从上下文中获取）：
1. 数据集的主要用途
2. 包含的数据类型
3. 数据量大小（如有提及）
4. 创建者/组织（如有提及）
5. 该数据集在论文中的应用方式

格式如下：
####
{{
    "数据集名称1": "简短描述...",
    "数据集名称2": "简短描述..."
}}
####
"""
