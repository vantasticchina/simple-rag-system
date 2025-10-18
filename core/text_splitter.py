from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocumentSplitter:
    """文档切分器，使用LangChain的RecursiveCharacterTextSplitter"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        """
        初始化文档切分器
        
        Args:
            chunk_size: 每个块的最大字符数
            chunk_overlap: 块之间的重叠字符数
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
    
    def split_text(self, text: str) -> list:
        """
        切分文本为多个块
        
        Args:
            text: 输入文本
            
        Returns:
            切分后的文本块列表
        """
        return self.text_splitter.split_text(text)
    
    def split_documents(self, documents: list) -> list:
        """
        切分文档列表
        
        Args:
            documents: LangChain Document对象列表
            
        Returns:
            切分后的Document对象列表
        """
        return self.text_splitter.split_documents(documents)