import os
from openai import OpenAI

def test_gemini_api():
    # 从环境变量获取API密钥
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("请设置GOOGLE_API_KEY环境变量")

    # 创建客户端实例
    client = OpenAI(
        api_key=api_key,
        base_url="https://xly-gemini-prox-666.deno.dev"
    )

    try:
        response = client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[
                {"role": "user", "content": "Hello, how are you?"}
            ]
        )
        return response
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        return None

if __name__ == "__main__":
    response = test_gemini_api()
    if response:
        print("API连接成功！")
        print(f"响应内容: {response.choices[0].message.content}")
        print(f"总token数: {response.usage.total_tokens}")
    else:
        print("API请求失败")