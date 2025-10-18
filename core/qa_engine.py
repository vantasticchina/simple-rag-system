"""
问答生成模块
负责调用本地Ollama部署的大模型生成答案
"""

import requests

class QAEngine:
    def __init__(self, model_name="deepseek-r1:1.5b", base_url="http://localhost:11434"):
        """
        初始化问答引擎
        
        Args:
            model_name (str): Ollama模型名称
            base_url (str): Ollama服务基础URL
        """
        self.model_name = model_name
        self.base_url = base_url
    
    def generate_answer(self, query, context):
        """
        根据查询和上下文生成答案
        
        Args:
            query (str): 用户查询
            context (str): 检索到的上下文信息
            
        Returns:
            str: 生成的答案
        """
        # 构造提示词
        prompt = f"根据以下上下文回答问题：\n\n{context}\n\n问题：{query}\n\n答案："
        
        # 调用Ollama API
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post(url, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                print(f"API调用失败，状态码：{response.status_code}")
                print(f"响应内容：{response.text}")
                return ""
        except requests.exceptions.ConnectionError:
            print("无法连接到Ollama服务，请确保Ollama正在运行")
            return ""
        except requests.exceptions.Timeout:
            print("请求Ollama服务超时，请检查服务状态")
            return ""
        except Exception as e:
            print(f"调用Ollama API时出错：{e}")
            return ""