"""阿里云 DashScope 文本向量化（qwen3.7-text-embedding）。

复用现有 settings.openai_api_key：因为阿里云 DashScope 的 API Key 同时
支持 chat（OpenAI 兼容协议）和 embedding（DashScope 原生协议）。
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import DashScopeEmbeddings

from src.config.settings import settings


# DashScope embedding 模型
EMBEDDING_MODEL = "qwen3.7-text-embedding"


def get_embeddings() -> Embeddings:
    """构造 DashScope embedding 实例。

    复用 settings.openai_api_key 作为 DashScope API Key。
    """
    return DashScopeEmbeddings(
        model=EMBEDDING_MODEL,
        dashscope_api_key=settings.openai_api_key,
    )


if __name__ == "__main__":
    # 自测：embedding 一段文本，看返回维度
    import numpy as np

    emb = get_embeddings()
    test_texts = ["茅台 2024 年营收", "贵州茅台酒股份有限公司年度报告"]
    vectors = emb.embed_documents(test_texts)
    print(f"模型: {EMBEDDING_MODEL}")
    print(f"测试文本数: {len(test_texts)}")
    print(f"返回向量数: {len(vectors)}")
    print(f"向量维度: {len(vectors[0])}")
    print(f"向量前 5 维: {vectors[0][:5]}")

    # 计算相似度（应该较高）
    v1, v2 = np.array(vectors[0]), np.array(vectors[1])
    sim = float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    print(f"两段文本余弦相似度: {sim:.4f}")
