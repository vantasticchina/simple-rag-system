"""
数据加载模块
负责从各种来源加载文档数据
"""
from langchain.schema import Document
from .text_splitter import DocumentSplitter


class DataLoader:
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        """
        初始化DataLoader
        
        Args:
            chunk_size: 文档切分块大小
            chunk_overlap: 文档切分重叠大小
        """
        self.document_splitter = DocumentSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def load_from_file(self, file_path):
        """
        从文件加载文本数据
        
        Args:
            file_path (str): 文件路径
            
        Returns:
            str: 加载的文本内容
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except Exception as e:
            print(f"加载文件时出错: {e}")
            return ""

    def load_and_split_from_file(self, file_path):
        """
        从文件加载文本数据并切分
        
        Args:
            file_path (str): 文件路径
            
        Returns:
            list: 切分后的Document对象列表
        """
        content = self.load_from_file(file_path)
        if not content:
            return []
        
        # 创建Document对象
        document = Document(page_content=content, metadata={"source": file_path})
        
        # 切分文档
        split_documents = self.document_splitter.split_documents([document])
        return split_documents

    def load_multiple_files(self, file_paths):
        """
        从多个文件加载文本数据
        
        Args:
            file_paths (list): 文件路径列表
            
        Returns:
            list: 加载的文本内容列表
        """
        contents = []
        for file_path in file_paths:
            content = self.load_from_file(file_path)
            if content:
                contents.append(content)
        return contents

    def load_and_split_multiple_files(self, file_paths):
        """
        从多个文件加载文本数据并切分
        
        Args:
            file_paths (list): 文件路径列表
            
        Returns:
            list: 切分后的Document对象列表
        """
        all_split_documents = []
        for file_path in file_paths:
            split_documents = self.load_and_split_from_file(file_path)
            all_split_documents.extend(split_documents)
        return all_split_documents