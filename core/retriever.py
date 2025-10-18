"""
检索模块
负责向量检索和结果重排序
"""

import numpy as np

class Retriever:
    def __init__(self, index_builder):
        """
        初始化检索器
        
        Args:
            index_builder (IndexBuilder): 索引构建器实例
        """
        self.index_builder = index_builder
    
    def search(self, query_vector, k=5):
        """
        搜索最相似的向量
        
        Args:
            query_vector (np.array): 查询向量
            k (int): 返回结果数量
            
        Returns:
            tuple: (距离数组, 索引数组)
        """
        query_vector = np.array(query_vector).reshape(1, -1).astype('float32')
        distances, indices = self.index_builder.index.search(query_vector, k)
        return distances[0], indices[0]
    
    def retrieve_documents(self, query_vector, k=5):
        """
        检索最相关的文档
        
        Args:
            query_vector (np.array): 查询向量
            k (int): 返回结果数量
            
        Returns:
            list: 检索到的文档列表
        """
        distances, indices = self.search(query_vector, k)
        retrieved_docs = []
        
        # 根据索引获取对应的文档
        for idx in indices:
            if idx < len(self.index_builder.documents):
                doc = self.index_builder.documents[idx]
                retrieved_docs.append({
                    'document': doc,
                    'distance': distances[list(indices).index(idx)]
                })
        
        return retrieved_docs
    
    def rerank(self, query_text, documents):
        """
        对检索结果进行重排序
        
        Args:
            query_text (str): 查询文本
            documents (list): 文档列表
            
        Returns:
            list: 重排序后的文档列表
        """
        # 这里应该实现具体的重排序逻辑
        # 暂时返回原始列表作为示例
        return documents