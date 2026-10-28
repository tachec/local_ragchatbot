# -*- coding: utf-8 -*-
"""知识库 ingestion：PDF / Markdown / TXT / 图片 -> 分块 -> 向量化 -> 本地 milvus-lite。

分块策略按文件格式区分：
- PDF：按页提取文本块，页内按段落打包，块上限约 900 字，保留页码；
- Markdown：先按标题层级切分章节，再按段落/代码块/表格切块，块首带标题路径；
- TXT / LOG：按空行段落与句子边界切块，块上限约 700 字；
- 图片：视觉模型生成描述；多模态向量模型下直接使用图片向量（以图搜库）。
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from collections import Counter
from pathlib import Path

import pymupdf as fitz  # PyMuPDF 2.x 官方导入名

from llm import (
    EMBED_DIM,
    describe_image,
    embedding_signature,
    get_embedding,
    get_image_embedding_from_path,
    is_multimodal_embedding_model,
)
from milvus import (
    COLLECTION_NAME,
    DB_PATH,
    delete_document,
    drop_collection,
    ensure_collection,
    has_collection,
    insert_rows,
    query_all_rows,
    rename_collection,
)

log = logging.getLogger("zhiyuan.kb")

# 每种格式的切分参数（字符数 / 重叠字符数）
CHUNK_PROFILES = {
    "pdf": {"size": 900, "overlap": 120},
    "md": {"size": 700, "overlap": 100},
    "txt": {"size": 700, "overlap": 100},
    "image": {"size": 1200, "overlap": 0},
}

SUPPORTED_EXTS = {
    ".pdf": "pdf",
    ".md": "md", ".markdown": "md",
    ".txt": "txt", ".log": "txt",
    ".png": "image", ".jpg": "image", ".jpeg": "image",
    ".webp": "image", ".bmp": "image", ".gif": "image",
}

_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s{0,3}(```+|~~~+)")
_FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", re.S)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])\s*|(?<=\.)\s+|\n+")


def detect_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTS:
        raise ValueError(
            f"不支持的文件类型 {ext or '（无扩展名）'}，支持：PDF / Markdown / TXT / 常见图片格式"
        )
    return SUPPORTED_EXTS[ext]


def _profile(src_type: str) -> dict:
    return CHUNK_PROFILES.get(src_type, CHUNK_PROFILES["txt"])


# ---------------- 文本清洗与通用分块 ----------------
def clean_text(text: str) -> str:
    """统一换行、去掉控制字符与多余空行，保留代码缩进。"""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_txt(path: str) -> str:
    """读取文本文件，自动尝试常见编码。"""
    raw = open(path, "rb").read()
    for enc in ("utf-8", "utf-8-sig", "gb18030", "big5", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("无法识别文本文件编码")


def _split_sentences(block: str) -> list:
    return [part.strip() for part in _SENTENCE_SPLIT_RE.split(block) if part.strip()]


def _hard_split(text: str, size: int) -> list:
    """没有可用边界时按长度硬切。"""
    return [text[i:i + size].strip() for i in range(0, len(text), size) if text[i:i + size].strip()]


def _split_plain_block(block: str, size: int) -> list:
    """把超长段落按句子边界拆成不超过 size 的片段。"""
    if len(block) <= size:
        return [block]
    pieces: list[str] = []
    current = ""
    for sentence in _split_sentences(block):
        if len(sentence) > size:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(_hard_split(sentence, size))
            continue
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= size:
            current = f"{current} {sentence}"
        else:
            pieces.append(current)
            current = sentence
    if current:
        pieces.append(current)
    return pieces


def _split_code_block(block: str, size: int) -> list:
    """超长代码块按行切分，并为每段补回围栏，保证 Markdown 结构完整。"""
    if len(block) <= size:
        return [block]
    lines = block.splitlines()
    opener = lines[0].strip()
    fence = opener[:3]
    body = lines[1:]
    if body and body[-1].strip().startswith(fence):
        body = body[:-1]
    limit = max(size - len(opener) - len(fence) - 2, 240)
    pieces: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in body:
        line_len = len(line) + 1
        if current and current_len + line_len > limit:
            pieces.append(f"{opener}\n" + "\n".join(current) + f"\n{fence}")
            current, current_len = [], 0
        current.append(line)
        current_len += line_len
    if current:
        pieces.append(f"{opener}\n" + "\n".join(current) + f"\n{fence}")
    return pieces or [block]


def _is_table_block(block: str) -> bool:
    lines = [line for line in block.splitlines() if line.strip()]
    return len(lines) >= 2 and all("|" in line for line in lines)


def _split_table_block(block: str, size: int) -> list:
    """超长表格按行切分，每段都保留表头与分隔行。"""
    if len(block) <= size:
        return [block]
    lines = block.splitlines()
    if len(lines) < 3:
        return _split_plain_block(block, size)
    header = lines[:2]
    header_text = "\n".join(header)
    limit = max(size - len(header_text) - 1, 200)
    pieces: list[str] = []
    current: list[str] = []
    current_len = 0
    for row in lines[2:]:
        row_len = len(row) + 1
        if current and current_len + row_len > limit:
            pieces.append(header_text + "\n" + "\n".join(current))
            current, current_len = [], 0
        current.append(row)
        current_len += row_len
    if current:
        pieces.append(header_text + "\n" + "\n".join(current))
    return pieces or [block]


def _split_block(block: str, size: int) -> list:
    stripped = block.lstrip()
    if stripped.startswith(("```", "~~~")):
        return _split_code_block(block, size)
    if _is_table_block(block):
        return _split_table_block(block, size)
    return _split_plain_block(block, size)


def _overlap_tail(text: str, overlap: int) -> str:
    """取上一块末尾的重叠片段，尽量从句子开头开始。"""
    if overlap <= 0 or not text:
        return ""
    tail = text[-overlap:]
    matches = list(re.finditer(r"[。！？!?；;\n]", tail))
    if matches and matches[-1].end() < len(tail):
        tail = tail[matches[-1].end():]
    return tail.strip()


def _is_structural_block(text: str) -> bool:
    """代码块 / 表格不适合做跨块重叠，避免破坏结构。"""
    stripped = (text or "").lstrip()
    if stripped.startswith(("```", "~~~")):
        return True
    return _is_table_block(text)


def pack_blocks(blocks: list, size: int, overlap: int) -> list:
    """把多个段落打包成块；块间保留少量重叠，避免语义被切断。"""
    expanded: list[str] = []
    for block in blocks:
        block = (block or "").strip()
        if block:
            expanded.extend(_split_block(block, size))

    chunks: list[str] = []
    current = ""
    for piece in expanded:
        if not current:
            current = piece
            continue
        if len(current) + 2 + len(piece) <= size:
            current = f"{current}\n\n{piece}"
            continue

        chunks.append(current)
        if _is_structural_block(current) or _is_structural_block(piece):
            tail = ""
        else:
            tail = _overlap_tail(current, overlap)
        candidate = f"{tail}\n\n{piece}".strip() if tail else piece
        if len(candidate) > size:
            candidate = piece
        current = candidate
    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if chunk.strip()]


def split_paragraphs(text: str) -> list:
    """按空行切段；没有空行时退化为按非空行切分。"""
    text = clean_text(text)
    if not text:
        return []
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if len(paragraphs) <= 1:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) > 1:
            paragraphs = lines
    return paragraphs


def chunk_text(text: str, size: int = None, overlap: int = None,
               src_type: str = "txt") -> list:
    """通用文本分块（TXT / LOG，也用于图片描述的纯文本回退）。"""
    profile = _profile(src_type)
    size = size or profile["size"]
    overlap = profile["overlap"] if overlap is None else overlap
    return pack_blocks(split_paragraphs(text), size, overlap)


# ---------------- PDF ----------------
def _strip_repeated_margins(pages: list) -> list:
    """删除多页重复出现的页眉 / 页脚（例如文档名、页码行）。"""
    if len(pages) < 4:
        return pages

    def norm(text: str) -> str:
        return re.sub(r"\d+", "#", text.strip())[:100]

    counter: Counter = Counter()
    for _, blocks in pages:
        for block in {blocks[0], blocks[-1]}:
            if 0 < len(block) <= 80:
                counter[norm(block)] += 1
    threshold = max(3, int(len(pages) * 0.6))
    noise = {key for key, count in counter.items() if count >= threshold}
    if not noise:
        return pages

    cleaned = []
    for page_no, blocks in pages:
        kept = []
        for index, block in enumerate(blocks):
            is_margin = index in (0, len(blocks) - 1)
            if is_margin and len(block) <= 80 and norm(block) in noise:
                continue
            kept.append(block)
        if kept:
            cleaned.append((page_no, kept))
    return cleaned


def load_pdf_pages(path: str) -> list:
    """按页提取 PDF 文本块，返回 [(page_no, [block, ...])]。"""
    pages = []
    with fitz.open(path) as doc:
        for page_no, page in enumerate(doc, start=1):
            blocks = []
            for block in page.get_text("blocks", sort=True):
                if len(block) >= 7 and block[6] != 0:  # 跳过图片块
                    continue
                text = clean_text(block[4] if len(block) > 4 else "")
                if text:
                    blocks.append(text)
            if not blocks:
                text = clean_text(page.get_text("text"))
                if text:
                    blocks = [text]
            if blocks:
                pages.append((page_no, blocks))
    if not pages:
        raise ValueError("PDF 中未提取到文本（可能是纯扫描件，可转成图片上传）")
    return _strip_repeated_margins(pages)


def extract_pdf_chunks(path: str) -> list:
    profile = _profile("pdf")
    out = []
    for page_no, blocks in load_pdf_pages(path):
        for chunk in pack_blocks(blocks, profile["size"], profile["overlap"]):
            out.append({"text": chunk, "page": page_no})
    return out


# ---------------- Markdown ----------------
def _strip_front_matter(text: str) -> tuple:
    """去掉 YAML front matter，返回 (正文, front matter 文本)。"""
    match = _FRONT_MATTER_RE.match(text)
    if not match:
        return text, ""
    return text[match.end():], match.group(1)


def _markdown_blocks(body: str) -> list:
    """把章节正文拆成段落块，代码围栏与表格保持完整。"""
    blocks: list[str] = []
    buffer: list[str] = []
    fence = None

    def flush():
        if buffer:
            text = "\n".join(buffer).strip()
            if text:
                blocks.append(text)
            buffer.clear()

    for line in body.splitlines():
        stripped = line.strip()
        if fence:
            buffer.append(line)
            if stripped.startswith(fence):
                fence = None
                flush()
            continue
        match = _FENCE_RE.match(line)
        if match:
            flush()
            fence = match.group(1)[:3]
            buffer.append(line)
            continue
        if not stripped:
            flush()
            continue
        buffer.append(line)
    flush()
    return blocks


def _markdown_sections(text: str) -> list:
    """按 ATX 标题层级切分章节，返回 [{'path': 标题路径, 'body': 正文}]。"""
    sections: list[dict] = []
    stack: list[tuple] = []
    buffer: list[str] = []
    fence = None

    def flush():
        body = "\n".join(buffer).strip()
        buffer.clear()
        if not body:
            return
        path = " > ".join(title for _, title in stack)
        sections.append({"path": path, "body": body})

    for line in text.splitlines():
        stripped = line.strip()
        if fence:
            buffer.append(line)
            if stripped.startswith(fence):
                fence = None
            continue
        fence_match = _FENCE_RE.match(line)
        if fence_match:
            fence = fence_match.group(1)[:3]
            buffer.append(line)
            continue
        heading = _HEADING_RE.match(line)
        if heading:
            flush()
            level = len(heading.group(1))
            title = heading.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            continue
        buffer.append(line)
    flush()
    return sections


def extract_markdown_chunks(path: str) -> list:
    profile = _profile("md")
    size, overlap = profile["size"], profile["overlap"]
    text, front_matter = _strip_front_matter(load_txt(path))
    text = clean_text(text)
    if not text:
        return []

    sections = _markdown_sections(text)
    if not sections:
        return [{"text": chunk, "page": 0}
                for chunk in chunk_text(text, size, overlap, src_type="md")]

    # front matter 里的 title 在文档没有一级标题时作为标题路径
    fallback_title = ""
    if front_matter:
        match = re.search(r"^title:\s*[\"']?(.+?)[\"']?\s*$", front_matter, re.M)
        if match:
            fallback_title = match.group(1).strip()

    chunks: list[str] = []
    for section in sections:
        body = section["body"]
        path_prefix = section["path"] or fallback_title
        prefix = f"{path_prefix}\n\n" if path_prefix else ""
        full_text = f"{prefix}{body}".strip()
        if len(full_text) <= size:
            chunks.append(full_text)
            continue

        body_size = max(size - len(prefix), 300)
        for piece in pack_blocks(_markdown_blocks(body), body_size, overlap):
            chunks.append(f"{prefix}{piece}".strip() if prefix else piece)
    return [{"text": chunk, "page": 0} for chunk in chunks if chunk.strip()]


# ---------------- 图片 ----------------
def extract_image_chunks(path: str) -> list:
    description = describe_image(path).strip()
    if not description:
        raise ValueError("视觉模型未返回任何图片描述")
    description = description[:60000]
    if is_multimodal_embedding_model():
        # 多模态模型直接为图片生成向量，描述文本作为检索命中后的可读内容
        return [{"text": description, "page": 0}]
    profile = _profile("image")
    return [{"text": chunk, "page": 0}
            for chunk in chunk_text(description, profile["size"], profile["overlap"], "image")]


def extract_chunks(path: str, src_type: str) -> list:
    """从文件提取分块，返回 [{'text', 'page'}]。"""
    if src_type == "pdf":
        return extract_pdf_chunks(path)
    if src_type == "md":
        return extract_markdown_chunks(path)
    if src_type == "txt":
        profile = _profile("txt")
        return [{"text": chunk, "page": 0}
                for chunk in chunk_text(load_txt(path), profile["size"], profile["overlap"], "txt")]
    if src_type == "image":
        return extract_image_chunks(path)
    raise ValueError(f"未知类型 {src_type}")


def ingest_file(path: str, doc_name: str = None, table_name: str = COLLECTION_NAME) -> dict:
    """把单个文件解析、分块、向量化后写入 milvus-lite。返回入库摘要。"""
    src_type = detect_type(path)
    doc_name = doc_name or os.path.basename(path)
    ensure_kb_ready()

    chunks = extract_chunks(path, src_type)
    if not chunks:
        raise ValueError("文件中未提取到可用文本内容")

    vectors = None
    if src_type == "image" and is_multimodal_embedding_model():
        try:
            vectors = [get_image_embedding_from_path(path)]
        except Exception as exc:  # noqa: BLE001
            log.warning("图片向量化失败，回退为描述文本向量：%s", exc)
            vectors = None
    if vectors is None:
        vectors = get_embedding([c["text"] for c in chunks], dimensions=EMBED_DIM)
    if len(vectors) != len(chunks):
        raise ValueError("向量数量与分块数量不一致，请检查向量模型配置")

    rows = [
        {
            "text": c["text"],
            "vector": v,
            "doc_name": doc_name,
            "src_type": src_type,
            "chunk_idx": i,
            "page": c["page"],
        }
        for i, (c, v) in enumerate(zip(chunks, vectors))
    ]
    ensure_collection(table_name)
    # 同名文档重复上传时先清掉旧分块，避免向量库中残留重复内容
    delete_document(doc_name, table_name)
    insert_rows(rows, table_name)
    return {"doc_name": doc_name, "src_type": src_type, "chunks": len(rows)}


# ---------------- 向量模型变更后的索引重建 ----------------
_META_PATH = Path(DB_PATH).parent / "rag_meta.json"
_ready_lock = threading.RLock()
_ready_signature: str | None = None
_ready_failed_signature: str | None = None
_ready_error: str | None = None


def _read_meta() -> dict:
    try:
        with open(_META_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_meta(signature: str, rows: int) -> None:
    try:
        _META_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_META_PATH, "w", encoding="utf-8") as fh:
            json.dump({"signature": signature, "rows": rows}, fh, ensure_ascii=False, indent=2)
    except OSError as exc:
        log.warning("写入向量库元数据失败：%s", exc)


def _rebuild_index(rows: list, signature: str) -> None:
    """用当前向量模型重新计算全部文本向量，再原子替换旧 collection。"""
    valid = [row for row in rows if str(row.get("text") or "").strip()]
    if not valid:
        return

    log.info("向量模型已变更（%s），开始重建知识库索引：%d 个片段", signature, len(valid))
    texts = [str(row["text"]) for row in valid]
    vectors: list[list] = []
    batch = 160
    for start in range(0, len(texts), batch):
        vectors.extend(get_embedding(texts[start:start + batch], dimensions=EMBED_DIM))
        log.info("重建索引进度：%d/%d", min(start + batch, len(texts)), len(texts))
    if len(vectors) != len(valid):
        raise RuntimeError("重建索引失败：向量数量与片段数量不一致")

    temp_name = f"{COLLECTION_NAME}__rebuild"
    backup_name = f"{COLLECTION_NAME}__old"
    drop_collection(temp_name)
    drop_collection(backup_name)
    ensure_collection(temp_name)
    try:
        insert_rows([
            {
                "text": str(row["text"]),
                "vector": vector,
                "doc_name": str(row.get("doc_name") or "未知文档"),
                "src_type": str(row.get("src_type") or "txt"),
                "chunk_idx": int(row.get("chunk_idx") or 0),
                "page": int(row.get("page") or 0),
            }
            for row, vector in zip(valid, vectors)
        ], temp_name)
        rename_collection(COLLECTION_NAME, backup_name)
        try:
            rename_collection(temp_name, COLLECTION_NAME)
        except Exception:
            rename_collection(backup_name, COLLECTION_NAME)
            raise
        drop_collection(backup_name)
    except Exception:
        drop_collection(temp_name)
        raise
    log.info("知识库索引重建完成：%d 个片段", len(valid))
    _write_meta(signature, len(valid))


def ensure_kb_ready(force: bool = False) -> None:
    """确保 collection 使用当前向量模型；模型/维度变化时自动重建索引。"""
    global _ready_signature, _ready_failed_signature, _ready_error
    signature = embedding_signature()
    with _ready_lock:
        if not force and _ready_signature == signature:
            return
        if not force and _ready_failed_signature == signature and _ready_error:
            raise RuntimeError(f"知识库索引重建失败：{_ready_error}")
        if force:
            _ready_failed_signature = None
            _ready_error = None
        # 上次重建若在重命名阶段中断，优先恢复备份，避免误建空库
        backup_name = f"{COLLECTION_NAME}__old"
        if not has_collection(COLLECTION_NAME) and has_collection(backup_name):
            log.warning("检测到上次索引重建中断，正在恢复备份 collection")
            rename_collection(backup_name, COLLECTION_NAME)
        ensure_collection()
        if force:
            _ready_signature = None
        else:
            meta = _read_meta()
            if meta.get("signature") == signature:
                _ready_signature = signature
                return

        rows = query_all_rows()
        if not rows:
            _write_meta(signature, 0)
            _ready_signature = signature
            _ready_failed_signature = None
            _ready_error = None
            return
        try:
            _rebuild_index(rows, signature)
        except Exception as exc:  # noqa: BLE001
            _ready_error = str(exc)
            _ready_failed_signature = signature
            log.exception("知识库索引重建失败，暂时保留旧索引")
            raise
        _ready_signature = signature
        _ready_failed_signature = None
        _ready_error = None


def embedding_status() -> dict:
    """知识库向量配置状态，供健康检查 / 页面提示使用。"""
    signature = embedding_signature()
    meta = _read_meta()
    return {
        "signature": signature,
        "ready": _ready_signature == signature or meta.get("signature") == signature,
        "error": _ready_error,
    }
