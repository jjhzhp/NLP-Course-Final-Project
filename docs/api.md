# API

## Health

```http
GET /api/health
```

## Upload Documents

```http
POST /api/documents/upload
Content-Type: multipart/form-data
```

Field:

```text
files: UploadFile[]
```

支持格式：`pdf` / `md` / `markdown` / `txt`，单个文件默认上限 100 MB（可通过 `MAX_UPLOAD_SIZE_MB` 调整）。
- 不支持的扩展名返回 `400`。
- 单个文件超过上限返回 `413`。

## List Documents

```http
GET /api/documents
```

## Delete Document

```http
DELETE /api/documents/{document_id}
```

## Search

```http
POST /api/search
Content-Type: application/json

{
  "query": "什么是条件随机场？",
  "top_k": 5,
  "profile": "qa"
}
```

返回结果包含 `vector_score`、`bm25_score`、`rerank_score` 和 `final_score`。开启 reranker 时，最终顺序优先由 `rerank_score` 决定。

`profile` 可选：

- `default`：基础混合检索。
- `qa`：偏精确问答检索。
- `summary`：扩展 query 后做广覆盖总结检索。

## Chat

```http
POST /api/chat
Content-Type: application/json

{
  "query": "什么是条件随机场？",
  "task_type": "qa",
  "use_pro_model": false,
  "top_k": 5
}
```

总结请求示例：

```http
POST /api/chat
Content-Type: application/json

{
  "query": "总结一下中文分词的主要方法",
  "task_type": "summary",
  "use_pro_model": false
}
```

自动路由出题请求示例：

```http
POST /api/chat
Content-Type: application/json

{
  "query": "围绕 HMM 和 CRF 的区别出三道题",
  "task_type": "auto",
  "use_pro_model": false
}
```

批改请求示例：

```http
POST /api/chat
Content-Type: application/json

{
  "query": "题目：CRF 和 HMM 有什么区别？\n我的答案：HMM 是生成式模型，CRF 是判别式模型。",
  "task_type": "grade"
}
```

选择题批改时，系统优先判断学生选择是否为题干选项中的最佳答案；如果课程资料还有更完整表述，会放在知识补充中，不直接影响选择题判分。

`task_type` 支持：

- `auto`
- `qa`
- `summary`
- `quiz`
- `grade`

当 `task_type="auto"` 时，后端会先调用 LLM Router Agent 判断任务类型、检索策略和改写后的检索 query。显式传入其他任务类型时，用户选择优先。LLM Router 失败时会回退到规则路由。

## Chat Stream (SSE)

```http
POST /api/chat/stream
Content-Type: application/json

{
  "query": "什么是条件随机场？",
  "task_type": "qa"
}
```

返回 `text/event-stream`，每条 `data:` 行是一个 JSON：

- `{"type":"meta","task_type":"qa","confidence":"high","sources":[...],"message":null}` 首先到达。
- `{"type":"delta","text":"..."}` 多次，逐 token 拼接成最终回答。
- `{"type":"done"}` 表示正常结束。
- `{"type":"error","message":"..."}` 表示中途失败。

请求体与 `/api/chat` 完全一致。
