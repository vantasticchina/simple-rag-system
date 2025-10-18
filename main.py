"""
RAG系统入口文件
整合各模块功能，提供完整的问答流程
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.data_loader import DataLoader
from core.index_builder import IndexBuilder
from core.retriever import Retriever
from core.qa_engine import QAEngine
from core.vectorizer import Vectorizer

def main():
    """主函数，运行RAG系统"""
    # 初始化各模块
    data_loader = DataLoader(chunk_size=1000, chunk_overlap=100)  # 使用更大的切分参数以减少chunks数量
    
    # 选择向量化方式：
    # 如果启用了本地模型，维度应为384（all-MiniLM-L6-v2输出）
    # 如果使用Ollama API，维度应为1536（Qwen3-Embedding输出）
    use_local_model = False  # 设为True以启用本地模型（如果安装了相关依赖且有GPU）
    
    if use_local_model:
        vectorizer = Vectorizer(
            dimension=384,  # all-MiniLM-L6-v2输出384维向量
            use_local_model=True,
            local_model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        index_builder = IndexBuilder(dimension=384)  # 维度需匹配
    else:
        vectorizer = Vectorizer(
            dimension=1536,  # Ollama Qwen3-Embedding输出1536维向量
            use_local_model=False
        )
        index_builder = IndexBuilder(dimension=1536)  # 维度需匹配
    
    retriever = Retriever(index_builder)
    qa_engine = QAEngine()
    
    # 加载文档数据（现在会自动切分）
    file_paths = [
        "docs/journey_to_the_west.txt",
        "docs/three_kingdoms.txt"
    ]
    
    print("正在加载并切分文档数据...")
    split_documents = data_loader.load_and_split_multiple_files(file_paths)
    if not split_documents:
        print("未能加载或切分任何文档，请检查文件路径。")
        return
    
    print(f"成功加载并切分 {len(split_documents)} 个文档块。")
    
    # 提取文档内容用于向量化
    document_contents = [doc.page_content for doc in split_documents]
    
    # 文档向量化处理（包含进度显示）
    print("正在向量化文档...")
    vectors = vectorizer.vectorize_documents(document_contents)
    
    print("正在构建索引...")
    # 构建索引并存储文档
    index = index_builder.build_index(vectors, split_documents)
    print("索引构建完成")
    
    # 进入问答循环
    while True:
        query = input("\n请输入您的问题 (输入 'quit' 退出): ")
        if query.lower() == 'quit':
            break
            
        print(f"正在处理查询: {query}")

        # 查询向量化处理
        print("正在将查询转换为向量...")
        query_vector = vectorizer.vectorize_query(query)
        print("查询向量化完成")
        
        # 执行检索
        print("正在检索相关文档...")
        retrieved_docs = retriever.retrieve_documents(query_vector, k=3)
        print(f"检索完成，找到 {len(retrieved_docs)} 个相关文档")
        
        # 获取相关文档内容
        print("正在获取相关文档内容...")
        relevant_docs = [doc['document'].page_content for doc in retrieved_docs]
        context = "\n".join(relevant_docs)
        print(f"获取到 {len(relevant_docs)} 个文档内容")
        
        # 生成答案
        print("正在生成答案...")
        answer = qa_engine.generate_answer(query, context)
        print("答案生成完成")

        # 输出答案
        print(f"答案: {answer}")

        
if __name__ == "__main__":
    main()