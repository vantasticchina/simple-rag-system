import numpy as np
from typing import List
import requests
from langchain.schema import Document

# 尝试导入GPU支持的库
try:
    from sentence_transformers import SentenceTransformer
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False
    SentenceTransformer = None

class Vectorizer:
    """
    文档和查询向量化处理器
    支持使用Ollama API或本地模型进行向量化
    """
    
    def __init__(self, 
                 dimension: int = 1536, 
                 base_url: str = "http://localhost:11434", 
                 embedding_model: str = "dengcao/Qwen3-Embedding-0.6B:Q8_0",
                 use_local_model: bool = False,
                 local_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        初始化向量化器
        
        Args:
            dimension: 向量维度
            base_url: Ollama API的URL
            embedding_model: Ollama模型名称
            use_local_model: 是否使用本地模型进行向量化
            local_model_name: 本地模型名称
        """
        self.dimension = dimension
        self.base_url = base_url
        self.embedding_model = embedding_model
        self.use_local_model = use_local_model
        self.local_model_name = local_model_name
        self.local_model = None
        
        # 如果启用本地模型，则加载模型
        if self.use_local_model and SentenceTransformer is not None:
            print(f"正在加载本地模型 {self.local_model_name}...")
            try:
                self.local_model = SentenceTransformer(self.local_model_name)
                if GPU_AVAILABLE:
                    self.local_model = self.local_model.to('cuda')
                    print(f"检测到GPU，将使用GPU进行向量化。")
                else:
                    print(f"未检测到GPU，将使用CPU进行向量化。")
            except Exception as e:
                print(f"加载本地模型失败: {e}")
                print("将回退到Ollama API进行向量化")
                self.use_local_model = False
    
    def vectorize_documents(self, documents) -> List[List[float]]:
        """
        将文档列表转换为向量表示
        
        Args:
            documents: 文档内容列表，可以是字符串列表或LangChain Document对象列表
            
        Returns:
            向量表示的列表
        """
        # 提取文档内容
        texts = []
        for doc in documents:
            # 检查是否为Document对象，如果是则提取内容
            if isinstance(doc, Document):
                text = doc.page_content
            else:
                text = doc
            texts.append(text)
        
        if self.use_local_model and self.local_model is not None:
            # 使用本地模型进行向量化（支持GPU加速）
            return self._vectorize_with_local_model(texts)
        else:
            # 使用Ollama API进行向量化
            return self._vectorize_with_api(texts)
    
    def _vectorize_with_local_model(self, texts: List[str]) -> List[List[float]]:
        """
        使用本地模型向量化文本
        
        Args:
            texts: 文本列表
            
        Returns:
            向量表示的列表
        """
        # 批量处理以提高效率
        total_docs = len(texts)
        
        # 分批处理
        batch_size = 32  # 可根据GPU内存调整
        embeddings = []
        
        for i in range(0, total_docs, batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # 显示进度
            progress = min(i + batch_size, total_docs) / total_docs * 100
            print(f"\r正在使用本地模型embedding文档 {min(i + batch_size, total_docs)}/{total_docs} ({progress:.1f}%)...", end="", flush=True)
            
            # 批量向量化
            batch_embeddings = self.local_model.encode(batch_texts).tolist()
            
            # 调整维度
            for embedding in batch_embeddings:
                if len(embedding) != self.dimension:
                    if len(embedding) > self.dimension:
                        embedding = embedding[:self.dimension]
                    else:
                        embedding.extend([0.0] * (self.dimension - len(embedding)))
                embeddings.append(embedding)
        
        print()  # 换行
        return embeddings
    
    def _vectorize_with_api(self, texts: List[str]) -> List[List[float]]:
        """
        使用API向量化文本
        
        Args:
            texts: 文本列表
            
        Returns:
            向量表示的列表
        """
        # 批处理文档以提高性能，并显示进度
        embeddings = []
        total_docs = len(texts)
        for idx, text in enumerate(texts):
            # 显示进度
            progress = (idx + 1) / total_docs * 100
            print(f"\r正在使用API embedding文档 {idx + 1}/{total_docs} ({progress:.1f}%)...", end="", flush=True)
            
            embedding = self._get_embedding(text)
            if embedding is not None:
                embeddings.append(embedding)
            else:
                # 如果API调用失败，回退到随机向量
                embeddings.append(self._generate_random_vector())
        
        # 换行以保持输出整洁
        print()  # 换行
        
        return embeddings
    
    def vectorize_query(self, query: str) -> List[float]:
        """
        将查询转换为向量表示
        
        Args:
            query: 查询字符串
            
        Returns:
            查询的向量表示
        """
        print("正在embedding查询...")
        
        if self.use_local_model and self.local_model is not None:
            try:
                # 使用本地模型向量化查询
                embedding = self.local_model.encode([query])[0].tolist()
                
                # 调整维度
                if len(embedding) != self.dimension:
                    if len(embedding) > self.dimension:
                        embedding = embedding[:self.dimension]
                    else:
                        embedding.extend([0.0] * (self.dimension - len(embedding)))
                
                print("查询embedding完成")
                return embedding
            except Exception as e:
                print(f"本地模型查询向量化失败: {e}")
                print("将使用Ollama API进行向量化")
        
        # 使用Ollama API作为备选方案
        embedding = self._get_embedding(query)
        if embedding is not None:
            print("查询embedding完成")
            return embedding
        else:
            # 如果API调用失败，回退到随机向量
            print("查询embedding失败，使用随机向量")
            return self._generate_random_vector()
    
    def _get_embedding(self, text: str) -> List[float]:
        """
        通过Ollama API获取文本嵌入向量
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量列表
        """
        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model": self.embedding_model,
            "prompt": text
        }
        
        try:
            response = requests.post(url, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                embedding = result.get("embedding", [])
                # 如果嵌入维度与预期不符，尝试调整
                if len(embedding) != self.dimension and len(embedding) > 0:
                    # 如果维度不匹配，尝试截取或填充
                    if len(embedding) > self.dimension:
                        embedding = embedding[:self.dimension]
                    else:
                        # 用0填充到指定维度
                        embedding.extend([0.0] * (self.dimension - len(embedding)))
                return embedding
            else:
                print(f"获取嵌入向量失败，状态码：{response.status_code}")
                # 返回随机向量作为回退方案
                return self._generate_random_vector()
        except requests.exceptions.RequestException as e:
            print(f"调用Ollama嵌入API时出错：{e}")
            # 返回随机向量作为回退方案
            return self._generate_random_vector()

    def _generate_random_vector(self) -> List[float]:
        """
        生成随机向量（占位符实现）
        
        Returns:
            随机向量
        """
        return np.random.rand(self.dimension).tolist()