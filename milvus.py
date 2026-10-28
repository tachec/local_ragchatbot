# -*- coding: utf-8 -*-
"""本地 milvus-lite 向量库封装。

相对 demo 的变化：
- 不再连接远程 Milvus（AWS_MS_URL/AWS_MS_TOKEN 弃用），改用本地文件库
  MilvusClient("./data/rag.db")，由 milvus-lite 纯 Python 引擎驱动；
- 文本字段长度 500 -> 65535，并增加来源元数据字段（doc_name/chunk_idx/page/src_type），
  用于回答时标注知识来源；
- 分析器类型用 milvus-lite 支持的 "jieba"（服务器版的 "chinese" 在 lite 中不存在）；
- 所有写操作加线程锁：milvus-lite 为单进程本地引擎，写需串行；
- 新增 list_documents / delete_document / collection_stats 供知识库管理页使用。
"""
import os
import threading

from dotenv import load_dotenv
from pymilvus import (
    AnnSearchRequest,
    DataType,
    Function,
    FunctionType,
    MilvusClient,
    RRFRanker,
)

load_dotenv(dotenv_path=".env", encoding="utf-8")

_DB_PATH = os.path.abspath(os.getenv("MILVUS_DB_PATH", "./data/rag.db"))
DB_PATH = _DB_PATH
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_kb")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))
# milvus-lite 支持 standard / jieba（无 "chinese" 类型，jieba 即中文分词）
TEXT_ANALYZER = os.getenv("TEXT_ANALYZER", "jieba")

os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)

_lock = threading.RLock()
ms_client = MilvusClient(uri=_DB_PATH)


class INITMS:
    """连接本地 milvus-lite，创建表。

    create_schema -> create_bm5_index -> create_index -> create_collection
    """

    def __init__(self) -> None:
        self.schema = None
        self.index_params = None
        self.functions = []

    def create_schema(self, fields: list):
        """创建schema"""
        self.schema = MilvusClient.create_schema(enable_dynamic_field=True)
        for field in fields:
            self.schema.add_field(**field)

    def create_index(self, indexs: list):
        """创建索引；创建 BM25 稀疏索引前须先调用 create_bm5_index"""
        self.index_params = ms_client.prepare_index_params()
        for index in indexs:
            self.index_params.add_index(**index)

    def create_bm5_index(self, name: str, bm5_input: str, bm5_output: str):
        """创建 BM25 稀疏函数 function"""
        bm25_function = Function(
            name=name,
            input_field_names=[bm5_input],
            output_field_names=[bm5_output],
            function_type=FunctionType.BM25,
        )
        self.schema.add_function(bm25_function)
        self.functions.append(bm25_function)

    def create_collection(self, table_name: str):
        """创建 collection"""
        with _lock:
            if self.index_params is not None:
                ms_client.create_collection(
                    collection_name=table_name,
                    schema=self.schema,
                    index_params=self.index_params,
                )
            else:
                ms_client.create_collection(collection_name=table_name, schema=self.schema)
            ms_client.load_collection(table_name)


def kb_fields() -> list:
    """知识库 collection 的字段定义（含来源元数据）。"""
    return [
        {"field_name": "id", "datatype": DataType.INT64, "is_primary": True, "auto_id": True},
        {"field_name": "vector", "datatype": DataType.FLOAT_VECTOR, "dim": EMBED_DIM},
        {
            "field_name": "text", "datatype": DataType.VARCHAR, "max_length": 65535,
            "enable_analyzer": True, "enable_match": True,
            "analyzer_params": {"type": TEXT_ANALYZER},
        },
        {"field_name": "sparse", "datatype": DataType.SPARSE_FLOAT_VECTOR},
        # ---- 来源元数据：用于答案中的来源标注与知识库管理 ----
        {"field_name": "doc_name", "datatype": DataType.VARCHAR, "max_length": 512},
        {"field_name": "src_type", "datatype": DataType.VARCHAR, "max_length": 16},  # pdf/txt/image
        {"field_name": "chunk_idx", "datatype": DataType.INT64},
        {"field_name": "page", "datatype": DataType.INT64},  # PDF 页码，无页码概念时为 0
    ]


def kb_indexes() -> list:
    return [
        {"field_name": "vector", "index_name": "dense_hnsw", "index_type": "HNSW", "metric_type": "COSINE"},
        {
            "field_name": "sparse", "index_name": "sparse_bm25",
            "index_type": "SPARSE_INVERTED_INDEX", "metric_type": "BM25",
            "params": {"bm25_k1": 1.6},
        },
    ]


def ensure_collection(table_name: str = COLLECTION_NAME) -> str:
    """知识库表不存在则创建，返回表名。"""
    with _lock:
        if not ms_client.has_collection(table_name):
            init = INITMS()
            init.create_schema(kb_fields())
            init.create_bm5_index("text_bm25_emb", "text", "sparse")
            init.create_index(kb_indexes())
            init.create_collection(table_name)
        else:
            ms_client.load_collection(table_name)
    return table_name


class SearchMilvusMixed:
    """稠密向量 + BM25 稀疏向量混合检索，RRF 融合排序。

    Args:
        table_name: collection 名称
        dense_field / sparse_field: 稠密/稀疏向量字段名
        dense_query: 单条稠密向量查询（List[float]）
        sparse_query: BM25 查询原始文本（str，由 BM25 函数端分析）
        source_limit: 两路召回候选数（应 >= limit）
        limit: 最终返回数量
        output_fields: 需要返回的字段列表
    """

    def __init__(
        self,
        table_name: str,
        dense_field: str,
        sparse_field: str,
        dense_query: list,
        sparse_query,
        source_limit: int = 8,
        limit: int = 5,
        output_fields: list = None,
        radius: float = 0.0,
    ) -> None:
        if source_limit < limit:
            raise ValueError(f"source_limit({source_limit}) 应 >= limit({limit})")
        self._table_name = table_name
        self._dense_field = dense_field
        self._sparse_field = sparse_field
        self._source_limit = source_limit
        self._limit = limit
        self._dense_query = dense_query
        self._sparse_query = sparse_query
        self._output_fields = output_fields or ["text", "doc_name", "chunk_idx", "page", "src_type"]
        self._radius = radius

    def search_rrf(self) -> list:
        search_cm = AnnSearchRequest(
            data=[self._dense_query],
            anns_field=self._dense_field,
            param={"metric_type": "COSINE", "radius": self._radius},
            limit=self._source_limit,
        )
        search_xs = AnnSearchRequest(
            data=[self._sparse_query],
            anns_field=self._sparse_field,
            param={"metric_type": "BM25"},
            limit=self._source_limit,
        )
        with _lock:
            res = ms_client.hybrid_search(
                collection_name=self._table_name,
                reqs=[search_cm, search_xs],
                ranker=RRFRanker(k=50),  # RRF 融合排序，也可用 WeightedRanker 加权
                limit=self._limit,
                output_fields=self._output_fields,
            )
        return res[0] if res else []


def insert_rows(data: list, table_name: str = COLLECTION_NAME) -> bool:
    """向知识库插入数据行（list[dict]）。"""
    with _lock:
        ms_client.insert(collection_name=table_name, data=data)
    return True


def load_collection(table_name: str = COLLECTION_NAME) -> None:
    with _lock:
        if ms_client.has_collection(table_name):
            ms_client.load_collection(table_name)


def has_collection(table_name: str = COLLECTION_NAME) -> bool:
    with _lock:
        return bool(ms_client.has_collection(table_name))


def drop_collection(table_name: str) -> None:
    with _lock:
        if ms_client.has_collection(table_name):
            ms_client.drop_collection(table_name)


def rename_collection(old_name: str, new_name: str) -> None:
    with _lock:
        ms_client.rename_collection(old_name, new_name)
        if ms_client.has_collection(new_name):
            ms_client.load_collection(new_name)


def query_all_rows(table_name: str = COLLECTION_NAME) -> list:
    """读取 collection 的全部文本行（用于向量模型变更后的重建索引）。"""
    with _lock:
        if not ms_client.has_collection(table_name):
            return []
        ms_client.load_collection(table_name)
        iterator = ms_client.query_iterator(
            collection_name=table_name,
            batch_size=1000,
            limit=-1,
            filter="id >= 0",
            output_fields=["text", "doc_name", "src_type", "chunk_idx", "page"],
        )
        rows = []
        try:
            while True:
                batch = iterator.next()
                if not batch:
                    break
                rows.extend(batch)
        finally:
            iterator.close()
    return rows


def delete_document(doc_name: str, table_name: str = COLLECTION_NAME) -> int:
    """按文档名删除其全部分块，返回删除行数。"""
    esc = doc_name.replace('"', '\\"')
    with _lock:
        res = ms_client.delete(collection_name=table_name, filter=f'doc_name == "{esc}"')
    if isinstance(res, dict):
        return int(res.get("delete_count", 0))
    return int(getattr(res, "delete_count", 0) or 0)


def collection_stats(table_name: str = COLLECTION_NAME) -> dict:
    """返回行数等统计信息（表不存在时 row_count=0）。"""
    with _lock:
        if not ms_client.has_collection(table_name):
            return {"row_count": 0}
        stats = ms_client.get_collection_stats(table_name)
    return stats


# 启动时确保知识库表就绪
ensure_collection()
