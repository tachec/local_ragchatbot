# -*- coding: utf-8 -*-
"""CLI 版 RAG 自测：检索 -> rerank -> LLM 流式回答（与 app.py 同一条链路）。

运行: python main.py "你的问题"
"""
import sys

from app import GENERAL_SYSTEM, RAG_SYSTEM, retrieve
from llm import get_llm_response


def main():
    import sys
    data = sys.argv[1] if len(sys.argv) > 1 else "大模型人工智能一般使用什么编程语言"

    # 混合检索 + rerank
    materials, sources, best = retrieve(data)
    rag_mode = bool(sources)  # retrieve 已按阈值过滤
    sys_prompt = (RAG_SYSTEM.format(materials=materials) if rag_mode else GENERAL_SYSTEM)
    messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": data}]

    print(f"[模式] {'RAG 知识库' if rag_mode else '通用知识'}（最高相关度 {best:.2f}）")
    for s in sources:
        print(f"  来源{s['n']}: 《{s['doc_name']}》 片段{s['chunk_idx']} score={s['score']:.2f}")

    print("\n" + "=" * 20 + "完整回复" + "=" * 20)
    for chunk in get_llm_response(messages):
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        content = getattr(delta, "content", None)
        if content:
            print(content, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
