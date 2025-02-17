from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
import os
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# 初始化rich console
console = Console()

# 从环境变量获取 API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY in environment variables")

# 配置OpenAI客户端
client = OpenAI(
    api_key=GOOGLE_API_KEY,
    base_url="https://xly-gemini-prox-666.deno.dev"
)

# 1. 读取文本文件
def load_text(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            content = file.read()
            print(f"Successfully loaded file: {filepath}")
            print(f"Content length: {len(content)} characters")
            return content
    except FileNotFoundError:
        print(f"Error: File not found at path: {filepath}")
        raise
    except Exception as e:
        print(f"Error loading file: {str(e)}")
        raise

# 使用相对路径或绝对路径
text = load_text(r"test_txt\意大利面树保护法.txt")

# 2. 文本分块（Chunking）
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = text_splitter.split_text(text)

# 3. 生成文本块对象
documents = [Document(page_content=chunk) for chunk in chunks]

# 4. 使用代理API生成 Embedding
class ProxyEmbeddings(Embeddings):
    def __init__(self, client):
        self.client = client

    def embed_documents(self, texts):
        """Embed search docs."""
        embeddings = []
        for text in texts:
            response = self.client.embeddings.create(
                model="models/text-embedding-004",
                input=text
            )
            embeddings.append(response.data[0].embedding)
        return embeddings

    def embed_query(self, text):
        """Embed query text."""
        response = self.client.embeddings.create(
            model="models/text-embedding-004",
            input=text
        )
        return response.data[0].embedding

# 创建embedding实例
embedding_model = ProxyEmbeddings(client)

# 在文本加载后，添加缓存检查函数
def check_faiss_cache(cache_dir="faiss_cache"):
    """检查是否存在FAISS缓存"""
    if os.path.exists(cache_dir):
        try:
            # 添加 allow_dangerous_deserialization 参数
            return FAISS.load_local(
                cache_dir, 
                embedding_model, 
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            print(f"加载缓存失败: {str(e)}")
            return None
    return None

# 5. 创建或加载 FAISS 向量数据库
def get_vectorstore(documents, update_faiss=False, cache_dir="faiss_cache"):
    """获取向量存储，支持缓存"""
    if not update_faiss:
        cached_vectorstore = check_faiss_cache(cache_dir)
        if cached_vectorstore is not None:
            print("使用现有的FAISS缓存")
            return cached_vectorstore

    print("创建新的FAISS向量存储")
    vectorstore = FAISS.from_documents(documents, embedding_model)
    
    # 保存到缓存
    os.makedirs(cache_dir, exist_ok=True)
    vectorstore.save_local(cache_dir)
    
    return vectorstore

# 修改主流程中的向量存储创建
vectorstore = get_vectorstore(documents, update_faiss=False)

# 6. Define RAG process
def rag_query(query):
    console.print("\n" + "="*80, style="blue")
    console.print(Panel(f"[yellow]Query:[/yellow] {query}"), style="bold")
    
    # Retrieve relevant text chunks from FAISS
    retrieved_docs = vectorstore.similarity_search_with_score(query, k=4)
    
    # 展示相关文档和分数
    console.print("\n[blue]Related Context Chunks:[/blue]")
    context_parts = []
    for i, (doc, score) in enumerate(retrieved_docs, 1):
        console.print(f"\n[green]Chunk {i} (Similarity: {1 - score:.4f})[/green]")
        # 将文档内容按句号分割，只取前3句
        sentences = doc.page_content.strip().split('。')
        preview_text = '。'.join(sentences[:3]) + '。...' if len(sentences) > 3 else doc.page_content.strip()
        console.print(Panel(preview_text, border_style="dim"))
        context_parts.append(doc.page_content)
    
    context = "\n".join(context_parts)
    
    prompt = f"""Based on the following context, please answer the question:

Context:
{context}

Question:
{query}

Please provide a detailed but concise answer in Chinese."""

    console.print("\n[blue]Generating Response...[/blue]")
    
    # 使用代理API生成响应
    response = client.chat.completions.create(
        model="gemini-2.0-flash",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    answer = response.choices[0].message.content
    console.print("\n[blue]Generated Answer:[/blue]")
    console.print(Panel(answer, border_style="green"))
    console.print("="*80 + "\n", style="blue")
    
    return answer

if __name__ == "__main__":
    try:
        console.print("[bold green]Starting RAG Query Demo[/bold green]")

        query = "为什么偷猎者的行为会破坏意大利面树？"

        answer = rag_query(query)
    except Exception as e:
        console.print(f"[bold red]Error during execution:[/bold red] {str(e)}")