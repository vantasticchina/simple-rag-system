"""
索引构建模块
负责使用Faiss CPU版本构建和管理向量索引
"""

import faiss
import numpy as np

class IndexBuilder:
    """索引构建类，用于创建和管理Faiss向量索引"""

    def __init__(self, dimension):
        """
        初始化IndexBuilder
        
        Args:
            dimension (int): 向量维度
        """
        self.dimension = dimension
        self.index = None
        self.documents = []  # 用于存储文档内容，与索引中的向量对应

    def build_index(self, vectors, documents=None):
        """
        使用Faiss CPU版本构建索引
        
        Args:
            vectors (np.array): 文档向量数组
            documents (list): 对应的文档内容列表，可选
            
        Returns:
            faiss.Index: 构建的索引对象
        """
        # 将向量转换为float32类型
        vectors = np.array(vectors).astype('float32')
        
        # 创建CPU索引
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # 添加向量到索引
        self.index.add(vectors)
        
        # 如果提供了文档，则保存它们
        if documents is not None:
            self.documents = documents
        
        return self.index

    def save_index(self, file_path):
        """
        保存索引到文件
        
        Args:
            file_path (str): 保存索引的文件路径
        """
        if self.index is not None:
            faiss.write_index(self.index, file_path)
        else:
            print("索引为空，无法保存")

    def load_index(self, file_path):
        """
        从文件加载索引
        
        Args:
            file_path (str): 索引文件路径
            
        Returns:
            faiss.Index: 加载的索引对象
        """
        try:
            # 读取CPU索引
            self.index = faiss.read_index(file_path)
            
            return self.index
        except Exception as e:
            print(f"加载索引时出错: {e}")
            return None

    def add_documents(self, documents):
        """
        添加文档到索引
        
        Args:
            documents (list): 要添加的文档列表
        """
        self.documents.extend(documents)