# RAG系统

## 简介

这是一个基于Retrieval-Augmented Generation (RAG)技术的问答系统，专门针对《西游记》和《三国演义》等古典文学作品进行设计。系统能够根据用户提出的问题，从相关文档中检索信息，并利用本地部署的大语言模型生成答案。

## 功能特点

- **文档加载**: 支持从本地文件系统加载文本数据
- **文档切分**: 使用LangChain的RecursiveCharacterTextSplitter进行文档预处理，支持自定义chunk大小和重叠
- **向量索引**: 使用Faiss-CPU构建高效的向量索引，支持CPU和GPU环境
- **GPU加速**: 支持使用sentence-transformers本地模型进行向量化，可利用GPU加速处理
- **智能检索**: 实现向量相似性搜索，并结合重排序技术提高检索精度
- **问答生成**: 调用本地Ollama部署的deepseek-r1:1.5b大模型生成自然语言答案
- **进度显示**: 向量化过程中提供实时进度反馈

## 系统架构

本系统主要由以下核心模块组成：

1. **数据加载模块** (`core/data_loader.py`): 负责从文件系统加载文档数据并进行切分
2. **文档切分模块** (`core/text_splitter.py`): 使用LangChain进行文档切分，chunk_size=1000，chunk_overlap=100
3. **索引构建模块** (`core/index_builder.py`): 使用Faiss-CPU构建和管理向量索引
4. **向量化模块** (`core/vectorizer.py`): 处理文档和查询向量化，支持API和本地模型两种方式
5. **检索模块** (`core/retriever.py`): 执行向量检索并对结果进行重排序
6. **问答生成模块** (`core/qa_engine.py`): 调用本地Ollama deepseek-r1:1.5b大模型生成最终答案

## 环境依赖

- Python 3.8+
- Faiss-CPU
- LangChain
- Sentence-Transformers (用于GPU加速)
- PyTorch (用于GPU加速)
- NumPy
- Requests
- Ollama (用于本地部署大模型)

## 安装与配置

1. 安装Python依赖:
   ```
   pip install -r requirements.txt
   ```

2. 安装并配置Ollama:
   - 下载并安装Ollama
   - 拉取所需的大语言模型:
     ```
     ollama pull dengcao/Qwen3-Embedding-0.6B:Q8_0  # 用于向量化
     ollama pull deepseek-r1:1.5b                   # 用于问答生成
     ```

3. 准备文档数据:
   - 将《西游记》和《三国演义》等文档放置在docs目录

## 使用方法

1. 启动Ollama服务
2. 运行系统主程序
3. 选择向量化方式:
   - API方式 (默认，使用Ollama服务，输出1536维向量)
   - 本地模型方式 (可利用GPU加速，输出384维向量，需设置use_local_model=True)
4. 输入你想查询的问题
5. 系统将自动检索相关文档并生成答案

## 配置GPU加速

如需启用GPU加速，请在main.py中修改以下设置：

```python
use_local_model = True  # 启用本地模型
```

本地模型将自动检测GPU并使用CUDA进行加速处理。

## 项目结构

```
ragsystem/
├── core/
│   ├── data_loader.py
│   ├── index_builder.py
│   ├── retriever.py
│   ├── vectorizer.py
│   ├── text_splitter.py
│   └── qa_engine.py
├── docs/
│   ├── journey_to_the_west.txt
│   └── three_kingdoms.txt
└── requirements.txt
```

## 扩展性说明

本系统采用模块化设计，每个核心组件都封装在独立的类中，便于维护和扩展。例如，可以轻松替换不同的索引构建算法、重排序模型或大语言模型。

## 注意事项

- 如使用GPU加速，请确保PyTorch已正确安装并支持CUDA
- 确保Ollama服务正在运行，并且所需模型已下载
- 本地模型方式与API方式的向量维度不同，需保持index_builder和vectorizer维度一致
- 文档切分参数可根据实际需求调整，平衡精确性和性能