 # -*- coding: utf-8 -*-
"""大模型调用封装（阿里百炼 OpenAI 兼容接口）。

相对 demo 的变化：
- get_llm_response 改为接收 messages 列表（支持多轮对话）；
- 新增 trim_messages：只保留最近 N 轮（默认 3 轮）发给大模型；
- 新增 stream_chat_content：流式生成器，逐段产出回答文本（SSE 聊天用）；
- 新增 describe_image：视觉模型把图片转文字（附件展示、图片知识入库共用）；
- 新增 get_image_embedding / get_query_embedding：qwen3-vl-embedding 多模态向量，
  文本与图片共用同一语义空间；qwen3.7-text-embedding-flash 仍走旧版纯文本接口；
- 模型名均可通过 .env 覆盖（LLM_MODEL / EMBED_MODEL / RERANK_MODEL / VISION_MODEL）。
"""
import base64
import io
import os

import dashscope
import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path=".env", encoding="utf-8")

url = os.getenv("ALI_URL")
key = os.getenv("ALI_TOKEN")

LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.8-flash")
# 多模态向量模型（文本 / 图片同一语义空间）；如需纯文本模型，
# 在 .env 中改为 qwen3.7-text-embedding-flash，会自动走旧版 OpenAI embedding 接口。
EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3-vl-embedding")
RERANK_MODEL = os.getenv("RERANK_MODEL", "qwen3.7-text-rerank")
VISION_MODEL = os.getenv("VISION_MODEL", "").strip()  # 未配置时图片入库不可用
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))
EMBED_API_URL = os.getenv("EMBED_API_URL", "").strip()
MAX_HISTORY_ROUNDS = int(os.getenv("MAX_HISTORY_ROUNDS", "3"))
MULTIMODAL_EMBED_BATCH = 16   # 多模态接口单次 contents 上限 20，留余量
TEXT_EMBED_BATCH = 10         # 纯文本 embedding 接口单批上限 20，留余量

llm_client = OpenAI(api_key=key, base_url=url)
dash_client = dashscope.Client(api_key=key, base_url=os.getenv("ALI_RERANK_URL"))


def trim_messages(messages: list, max_rounds: int = MAX_HISTORY_ROUNDS) -> list:
    """只保留 system 消息 + 最近 max_rounds 轮（一轮 = 1 条 user + 1 条 assistant）。"""
    system = [m for m in messages if m.get("role") == "system"]
    convo = [m for m in messages if m.get("role") in ("user", "assistant")]
    return system + convo[-2 * max_rounds:]


def get_llm_response(messages: list, stream: bool = True, think: bool = True,
                     enable_search: bool = False):
    """调用千问大模型。

    :param messages: OpenAI 格式消息列表（多轮）
    :param stream: 是否开启流式传输
    :param think: 是否开启思考模式
    :param enable_search: 是否允许联网搜索（默认关闭，优先使用本地知识库）
    """
    return llm_client.chat.completions.create(
        model=LLM_MODEL,
        extra_body={"enable_thinking": think, "enable_search": enable_search},
        messages=messages,
        stream=stream,
        timeout=120,
    )


def stream_chat_content(messages: list, think: bool = False):
    """流式生成器：逐段 yield 回答正文（忽略思考过程），供 SSE 流式刷新。"""
    for chunk in get_llm_response(messages, stream=True, think=think):
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        content = getattr(delta, "content", None)
        if content:
            yield content


def is_multimodal_embedding_model(model: str = None) -> bool:
    """当前向量模型是否为文本 / 图片多模态模型。

    qwen3-vl-embedding 等走 DashScope 多模态 embedding 接口；
    qwen3.7-text-embedding-flash 等纯文本模型继续走 OpenAI 兼容接口。
    """
    name = (model or EMBED_MODEL or "").strip().lower()
    if not name or "text-embedding" in name:
        return False
    return any(token in name for token in ("vl-embedding", "vision", "multimodal"))


def embedding_signature() -> str:
    """向量空间签名：模型或维度变化时，已有知识库需要重建索引。"""
    kind = "multimodal" if is_multimodal_embedding_model() else "text"
    return f"{EMBED_MODEL}|{EMBED_DIM}|{kind}"


def _multimodal_endpoint() -> str:
    """多模态 embedding 的 HTTP 地址。"""
    if EMBED_API_URL:
        return EMBED_API_URL
    base = (os.getenv("ALI_RERANK_URL") or "").strip().rstrip("/")
    if not base:
        # 只配置了 OpenAI 兼容地址时，推导同一工作区的 DashScope 地址
        compat = (url or "").strip().rstrip("/")
        if compat.endswith("/compatible-mode/v1"):
            base = compat[:-len("/compatible-mode/v1")] + "/api/v1"
    if base:
        return base + "/services/embeddings/multimodal-embedding/multimodal-embedding"
    return (
        "https://dashscope.aliyuncs.com/api/v1"
        "/services/embeddings/multimodal-embedding/multimodal-embedding"
    )


def _normalize_image_mime(mime: str) -> str:
    kind = (mime or "png").lower()
    if kind.startswith("image/"):
        kind = kind[6:]
    if kind == "jpg":
        kind = "jpeg"
    if kind not in ("png", "jpeg", "webp", "bmp", "gif", "tiff"):
        kind = "png"
    return kind


def image_data_url(data: bytes, mime: str = "image/png") -> str:
    """把图片字节编码成多模态接口可识别的 data URL。

    多模态 embedding 不直接支持 GIF，遇到时取首帧转成 PNG。
    """
    if not data:
        raise ValueError("图片内容为空")
    kind = _normalize_image_mime(mime)
    if kind == "gif":
        try:
            from PIL import Image  # noqa: PLC0415

            with Image.open(io.BytesIO(data)) as image:
                image.seek(0)
                frame = image.convert("RGB")
                buffer = io.BytesIO()
                frame.save(buffer, format="PNG")
                data = buffer.getvalue()
                kind = "png"
        except Exception as exc:  # noqa: BLE001
            raise ValueError("GIF 图片转换失败") from exc
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/{kind};base64,{b64}"


def _normalize_query_image(image: str) -> str:
    """把前端传来的 data URL 规范化（GIF 转 PNG）；也允许直接传 http 图片地址。"""
    image = (image or "").strip()
    if image.startswith(("http://", "https://")):
        return image
    if not image.startswith("data:"):
        raise ValueError("图片数据格式不正确")
    header, sep, b64 = image.partition(",")
    if not sep or ";base64" not in header.lower():
        raise ValueError("图片数据格式不正确")
    mime = header[5:].split(";", 1)[0].strip().lower() or "image/png"
    try:
        data = base64.b64decode(b64, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise ValueError("图片 base64 解码失败") from exc
    return image_data_url(data, mime)


def _multimodal_embed(contents: list, dimensions: int, enable_fusion: bool = False) -> list:
    """调用 DashScope 多模态 embedding 接口，返回与 contents 顺序一致的向量列表。"""
    payload = {
        "model": EMBED_MODEL,
        "input": {"contents": contents},
        "parameters": {"dimension": dimensions},
    }
    if enable_fusion:
        payload["parameters"]["enable_fusion"] = True
    try:
        resp = httpx.post(
            _multimodal_endpoint(),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
            timeout=180,
        )
    except httpx.HTTPError as exc:
        raise RuntimeError(f"多模态向量接口连接失败：{exc}") from exc

    if resp.status_code != 200:
        detail = (resp.text or "").strip().replace("\n", " ")[:400]
        raise RuntimeError(f"多模态向量接口返回 HTTP {resp.status_code}：{detail}")
    try:
        data = resp.json()
    except ValueError as exc:
        raise RuntimeError("多模态向量接口返回了非 JSON 内容") from exc

    items = ((data.get("output") or {}).get("embeddings")) or []
    vectors = []
    for item in items:
        vec = item.get("embedding") if isinstance(item, dict) else None
        if not vec:
            raise RuntimeError("多模态向量接口未返回向量")
        vectors.append(vec)

    expected = 1 if enable_fusion else len(contents)
    if len(vectors) != expected:
        raise RuntimeError(
            f"多模态向量接口返回条数异常（期望 {expected}，实际 {len(vectors)}）"
        )
    return vectors


def _embed_batch(texts: list, dimensions: int) -> list:
    """旧版纯文本 embedding 接口（OpenAI 兼容）。"""
    resp = llm_client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
        dimensions=dimensions,
        encoding_format="float",
    )
    return [d.embedding for d in resp.data]


def get_embedding(input_data, dimensions: int = EMBED_DIM) -> list:
    """把文本转成向量。input_data 为 str 或 list[str]，返回单条向量或向量列表。

    自动按当前模型选择接口：
    - qwen3-vl-embedding：DashScope 多模态 embedding（文本与图片共用同一空间）；
    - qwen3.7-text-embedding-flash：OpenAI 兼容的纯文本 embedding。
    """
    single = isinstance(input_data, str)
    texts = [input_data] if single else [str(x) for x in input_data]
    texts = [t if t.strip() else " " for t in texts]
    if not texts:
        return []

    vectors = []
    if is_multimodal_embedding_model():
        for i in range(0, len(texts), MULTIMODAL_EMBED_BATCH):
            batch = texts[i:i + MULTIMODAL_EMBED_BATCH]
            vectors.extend(_multimodal_embed([{"text": t} for t in batch], dimensions))
    else:
        for i in range(0, len(texts), TEXT_EMBED_BATCH):
            vectors.extend(_embed_batch(texts[i:i + TEXT_EMBED_BATCH], dimensions))
    return vectors[0] if single else vectors


def get_image_embedding(data: bytes, mime: str = "image/png",
                        dimensions: int = EMBED_DIM) -> list:
    """把图片转成向量（仅多模态向量模型可用）。"""
    if not is_multimodal_embedding_model():
        raise RuntimeError(
            f"当前向量模型 {EMBED_MODEL} 不支持图片向量化，请改用 qwen3-vl-embedding"
        )
    return _multimodal_embed(
        [{"image": image_data_url(data, mime)}],
        dimensions,
    )[0]


def get_image_embedding_from_path(path: str, dimensions: int = EMBED_DIM) -> list:
    """读取本地图片文件并做向量化。"""
    with open(path, "rb") as fh:
        data = fh.read()
    ext = os.path.splitext(path)[1].lstrip(".").lower() or "png"
    return get_image_embedding(data, ext, dimensions)


def get_query_embedding(text: str = None, image_data_url: str = None,
                        dimensions: int = EMBED_DIM) -> list:
    """生成检索查询向量。

    - 只有文本：普通文本向量；
    - 只有图片：图片向量（以图搜库）；
    - 图文同时存在：qwen3-vl-embedding 的融合向量（enable_fusion）。
    """
    text = (text or "").strip()
    image = (image_data_url or "").strip()
    if not image:
        if not text:
            raise ValueError("检索查询不能为空")
        return get_embedding(text, dimensions)
    if not is_multimodal_embedding_model():
        raise RuntimeError(
            f"当前向量模型 {EMBED_MODEL} 不支持图片检索，请改用 qwen3-vl-embedding"
        )

    contents = []
    if text:
        contents.append({"text": text[:6000]})
    contents.append({"image": _normalize_query_image(image)})
    return _multimodal_embed(contents, dimensions, enable_fusion=len(contents) > 1)[0]


def get_rerank(query_data: str, doc_data: list, top_n: int = 3):
    """调用重排序模型，返回结果列表（每项含 index / relevance_score）。"""
    resp = dash_client.rerank.create(
        model=RERANK_MODEL,
        query=query_data,
        documents=doc_data,
        top_n=top_n,
    )
    return resp.results


def _vision_prompt(prompt: str = None) -> str:
    return prompt or (
        "请详细描述这张图片的全部内容：若有文字请逐字提取（OCR），若有图表请说明其含义，"
        "输出为纯文本，供知识库检索和问答使用。"
    )


def describe_image_bytes(data: bytes, mime: str = "image/png", prompt: str = None) -> str:
    """用视觉模型解析图片字节，返回文字描述 / OCR 结果。

    需要 .env 配置 VISION_MODEL（当前工作区可用的视觉模型名）。
    图片附件与知识库图片入库共用这条链路。
    """
    if not VISION_MODEL:
        raise RuntimeError(
            "未配置 VISION_MODEL，无法解析图片。请在 .env 中设置当前工作区可用的视觉模型名。"
        )
    if not data:
        raise ValueError("图片内容为空")

    kind = _normalize_image_mime(mime)

    b64 = base64.b64encode(data).decode("utf-8")
    resp = llm_client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/{kind};base64,{b64}"}},
                {"type": "text", "text": _vision_prompt(prompt)},
            ],
        }],
        stream=False,
        timeout=180,
    )
    return resp.choices[0].message.content or ""


def describe_image_data(data_url: str, prompt: str = None) -> str:
    """解析前端传来的 data:image/...;base64,... 图片。"""
    if not isinstance(data_url, str) or not data_url.startswith("data:"):
        raise ValueError("图片数据格式不正确")
    header, sep, b64 = data_url.partition(",")
    if not sep or ";base64" not in header.lower():
        raise ValueError("图片数据格式不正确")
    mime = header[5:].split(";", 1)[0].strip().lower() or "image/png"
    try:
        data = base64.b64decode(b64, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise ValueError("图片 base64 解码失败") from exc
    return describe_image_bytes(data, mime, prompt)


def describe_image(image_path: str, prompt: str = None) -> str:
    """兼容旧调用：读取本地图片文件后用视觉模型解析。"""
    with open(image_path, "rb") as f:
        data = f.read()
    ext = os.path.splitext(image_path)[1].lstrip(".").lower() or "png"
    return describe_image_bytes(data, ext, prompt)
