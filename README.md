# 知源 · 本地知识库问答助手

基于 **FastAPI + 原生 Web 前端 + milvus-lite + 大模型** 的本地知识库 RAG 应用。

界面参考主流大模型 Web 产品重做：左侧历史会话、中部对话流、右侧知识库抽屉；支持流式输出、来源卡片、明暗主题和移动端。输入框支持随消息上传文本或图片：文本会解析后交给模型，图片会先用视觉模型解析，再通过多模态向量直接检索知识库并参与回答。所有资料与会话都保存在本机。

\---

## 快速开始

### 1\. 环境要求

* Python 3.10–3.14（推荐 3.11+）
* 阿里百炼 OpenAI 兼容接口（LLM / embedding / rerank）
* 本地 milvus-lite（随依赖安装，无需启动服务）

### 2\. 安装依赖

```bash
pip install -r requirements.txt
```

如需启用 mem0 长期记忆，额外安装：

```bash
pip install mem0ai
```

### 3\. 配置 `.env`

编辑 `.env`，填入你的百炼工作区地址和密钥：

```env
# 百炼 OpenAI 兼容接口
ALI\_URL=https://your-workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
ALI\_TOKEN=sk-ws-your-token

\#dashscope重排序接口
ALI\_RERANK\_URL=https://your-workspace.cn-beijing.maas.aliyuncs.com/api/v1/

# 向量库与知识库
MILVUS\_DB\_PATH=./data/rag.db
COLLECTION\_NAME=rag\_kb
EMBED\_DIM=1024
TEXT\_ANALYZER=jieba

# 模型名（按工作区实际可用模型调整）
LLM\_MODEL=qwen3.8-flash
# 多模态向量模型：文本与图片共用同一语义空间
EMBED\_MODEL=qwen3-vl-embedding
# 如改用纯文本模型，会自动走旧版 OpenAI embedding 接口：
# EMBED\_MODEL=qwen3.7-text-embedding-flash
RERANK\_MODEL=qwen3.7-text-rerank
VISION\_MODEL=qwen3-vl-plus   # 图片描述 / OCR；留空则图片附件和图片入库不可用

RAG\_THRESHOLD=0.35           # rerank 相关度 ≥ 此值视为资料充足
MAX\_HISTORY\_ROUNDS=3         # 发给大模型的最近对话轮数
HISTORY\_DB=./data/history.db
USE\_MEM0=false               # 设为 true 并安装 mem0ai 启用长期记忆
```

> 更换 `EMBED\_MODEL` 或 `EMBED\_DIM` 后，启动时会自动用新模型重建已有知识库索引（旧文本会重新向量化，原文件无需重新上传）；首次启动可能需要几分钟。

### 4\. 启动

```bash
python app.py
```

默认地址：[http://127.0.0.1:7860](http://127.0.0.1:7860)

可通过环境变量 `APP\_HOST` / `APP\_PORT` 修改监听地址和端口。

### 5\. CLI 自测

```bash
python main.py "你的问题"
```

\---

## 界面亮点

|区域|说明|
|-|-|
|左侧会话栏|新建对话、搜索历史、按“今天 / 昨天 / 更早”分组、悬停删除|
|顶部状态栏|当前会话标题、知识库连接状态、主题切换、知识库入口|
|对话区|用户气泡 + 助手正文；支持 Markdown、代码块复制、引用角标、来源卡片|
|输入区|自适应高度、Enter 发送 / Shift+Enter 换行、停止生成、文本附件、图片搜库|
|知识库抽屉|拖拽 / 选择文件上传、实时解析状态、资料统计、按文档删除|
|来源详情|点击来源卡片查看文档名、页码、片段号、相关度与原文片段|
|明暗主题|跟随系统或手动切换，本地记忆|
|移动端|侧栏抽屉、全屏知识库、触控友好的按钮与滚动|

\---

## 文本附件与图片搜库

输入框左侧有两个附件按钮：

* **文本**：支持 `.txt` / `.md` / `.csv` / `.json` / `.log` / 代码文件等，自动识别 UTF-8 / GB18030，解析后随消息发送给模型；可只发附件，也可附带问题。为避免超出模型上下文，超长文本会自动截断。
* **图片**：支持 PNG / JPG / WEBP / BMP / GIF。图片先由 `VISION\_MODEL` 解析（描述 + OCR）；若使用多模态向量模型，还会直接用图片生成检索向量（以图搜库），图文同时存在时生成融合查询向量。解析文本同时用于：

  1. 检索本地知识库，命中结果会显示来源卡片；
  2. 作为附件上下文发送给对话模型，结合知识库资料一起回答。

附件只是随消息发送，不会写入向量知识库；如需把文件长期入库，请使用右侧“知识库”抽屉。附件解析结果会随会话历史保存，刷新后仍能看到附件名称和解析状态。

\---

## 知识库分块策略

不同格式使用不同的切分方式，避免把语义或结构切断：

|格式|切分方式|
|-|-|
|PDF|按页提取文本块，页内按段落打包（约 900 字，重叠 120 字），保留页码；自动过滤多页重复的页眉/页脚|
|Markdown|先按 `#` 标题层级切分章节，再按段落切块（约 700 字）；每块带上“一级标题 > 二级标题”路径，代码块和表格尽量保持完整|
|TXT / LOG|按空行段落 + 句子边界切块（约 700 字，重叠 100 字）|
|图片|视觉模型生成描述；`qwen3-vl-embedding` 下直接使用图片向量，描述作为命中后的可读内容|

同名文档重新上传时会先删除旧分块，再写入新分块，不会产生重复内容。

\---

## 功能概览

|功能|说明|
|-|-|
|🤖 RAG 问答|优先检索本地知识库；资料不足时谨慎使用通用知识并标注|
|📚 来源标注|回答中的 `【来源N】` 可点击，底部来源卡片展示文档、页码、片段与相关度|
|🖼️ 图片搜库|多模态向量模型下直接用图片向量检索；纯文本模型下回退为视觉描述检索|
|📎 文本附件|上传 txt / md / csv / json 等文本，解析后随消息发送给模型，可结合知识库回答|
|🌊 流式输出|SSE 逐字输出，附带“正在检索 / 正在生成”状态|
|💬 多轮对话|记住最近 3 轮上下文，历史保存到本地 SQLite|
|🆕 新会话|首次发送消息时才创建会话，避免产生空会话|
|📄 上传入库|支持 PDF / Markdown / TXT / LOG / 图片，按格式分块、向量化|
|🗑️ 文档管理|查看已入库文档，按文档删除；同名文档重新上传时自动替换旧分块|
|🧠 长期记忆|可选接入 mem0，跨会话沉淀用户知识|
|🧭 知识库整体分析|欢迎页三张卡片对全部资料做总结、提炼结论、对比差异，不再当作普通问题只召回零散片段|
|🔐 用户识别|首次访问下发长效 UUID Cookie，同一浏览器重开可恢复会话|
|📜 会话回溯|左侧历史会话可打开、继续追问、搜索和删除|
|🌗 主题|明暗主题切换，移动端自适应|

\---

## 项目结构

```text
py-mem/
├── app.py              # FastAPI 入口：页面路由 + SSE 聊天 + 会话/知识库 API
├── web/
│   ├── index.html      # 单页结构、SVG 图标
│   ├── styles.css      # 设计系统、明暗主题、响应式
│   └── app.js          # 会话、流式 SSE、Markdown、附件、图片搜库、主题
├── milvus.py           # 本地 milvus-lite 封装（建表/检索/插入/删除/重建索引）
├── llm.py              # 大模型调用（聊天/多模态向量/重排/视觉）
├── kb.py               # 知识库 ingestion（PDF/Markdown/TXT/图片解析 + 格式分块 + 索引重建）
├── storage.py          # SQLite 持久化（文档登记 + 会话历史 + mem0）
├── main.py             # CLI 版 RAG 自测
├── .env                # 配置文件
├── requirements.txt    # 依赖清单
├── data/               # 运行时数据（自动创建）
│   ├── rag.db          # milvus-lite 向量库
│   ├── rag\_meta.json   # 当前向量模型签名（用于自动重建索引）
│   └── history.db      # SQLite 历史记录
└── README.md
```

\---

## API 一览

|方法|路径|说明|
|-|-|-|
|GET|`/api/bootstrap`|用户、会话列表、最近会话消息、知识库统计|
|GET|`/api/sessions`|当前用户的会话列表|
|GET|`/api/sessions/{id}`|打开某个会话|
|DELETE|`/api/sessions/{id}`|删除某个会话|
|POST|`/api/chat`|SSE 流式聊天（`message`、可选 `session\_id` / `attachments` / `task`；`task` 为 `summarize` / `key\_points` / `compare` 时走整库分析）|
|GET|`/api/docs`|知识库文档列表与统计|
|POST|`/api/upload`|多文件上传并入库（multipart，字段名 `files`）|
|DELETE|`/api/docs?name=...`|删除指定文档|
|GET|`/api/health`|健康检查|

\---

## 用户识别与数据

* **首次访问**：中间件下发 `user\_id=UUID` 长效 Cookie（10 年、HttpOnly、SameSite=Lax）。
* **后续访问**：浏览器自动携带 Cookie，服务端按 `user\_id` 读取该用户的会话。
* **会话恢复**：打开页面时自动恢复最近一次有消息的会话；没有则展示欢迎页。
* **数据位置**：会话/文档登记在 `data/history.db`，向量在 `data/rag.db`，上传临时文件处理完即删除。消息附件（文本内容 / 图片解析结果）随消息保存在 `history.db`，但不写入向量库。
* **换浏览器/清除 Cookie**：会生成新的 `user\_id`，旧会话仍留在本机数据库中，但不会自动关联。

\---

## 注意事项

1. **单进程限制**：milvus-lite 是本地嵌入式引擎，同一数据目录只能被一个进程打开。运行 `app.py` 期间不要同时运行 `main.py`。
2. **图片功能**：图片附件需要配置 `VISION\_MODEL`（建议 `qwen3-vl-plus`）生成描述 / OCR；图片直接检索还需要多模态向量模型 `qwen3-vl-embedding`，若使用 `qwen3.7-text-embedding-flash` 会自动回退为描述文本检索。`qwen-image-\*` 属图像生成模型，不能用于图片解析。
3. **中文分词**：BM25 使用 `jieba` 分析器（`.env` 中 `TEXT\_ANALYZER=jieba`），首次运行会自动下载词表。
4. **上传限制**：知识库文件单个最大 50 MB；消息文本附件最大 2 MB，图片附件最大 8 MB，单条消息附件合计最大 16 MB。知识库支持 PDF / Markdown / TXT / LOG / PNG / JPG / JPEG / WEBP / BMP / GIF。
5. **mem0 可选**：默认关闭（`USE\_MEM0=false`）。启用后每轮对话结束异步沉淀长期记忆，需额外安装 `mem0ai`。
6. **旧数据兼容**：旧版 `data/history.db` 可直接沿用；旧回答末尾的“参考来源”页脚会被解析为新的来源卡片。更换向量模型后，旧知识库会自动重建索引。
7. **端口占用**：默认 `127.0.0.1:7860`，如被占用可设置 `APP\_PORT=7861` 后启动。

\---

## 版本信息

* Python 3.14 已验证
* pymilvus 3.0.2 / milvus-lite 3.2.2（BM25 + RRF 混合检索）
* FastAPI 0.162 / uvicorn 0.54
* 阿里百炼 qwen3 系列模型（qwen3.8-max / qwen3-vl-embedding / qwen3.7-text-rerank / qwen3-vl-plus）

