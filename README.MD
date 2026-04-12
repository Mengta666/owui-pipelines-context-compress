# OpenWebUI Pipeline - Mini Agent Code Build

基于 Open WebUI Pipeline 的轻量 Agent 运行时，聚焦**长上下文治理**、**附件/长文件渐进式读取**、**会话状态可恢复**。  
适用于多轮代码分析、文档分析、带附件对话等场景。

---

## 1. 项目定位

本项目不是“自动写代码+自动执行”的自治框架（**可以轻松扩展到自动执行框架**），而是一个偏工程化的**只读分析型 Agent Runtime**：

- 面向多轮长会话，避免上下文爆炸
- 支持附件登记、文件按需读取，不把大文件全文塞进 prompt
- 对历史进行分层压缩（轻压缩 + 摘要压缩 + 硬限制裁剪）
- 被压缩内容外置到 transcript/artifact，并可在后续轮次恢复
- 让**不具备长文阅读**的模型**可以长文阅读**
- 具体框架优缺点、实现可查看 `./docs/project_description_doc.md`
- `./docs/` 目录中也有一些设计文档和实现细节说明（以流程图为主）

---

## 2. 目录结构

```text
pipelines/
├── docker-compose.pipelines.yaml
├── .env.pipelines.example
├── docs/
│   └── github_project_description_doc.md
├── pipelines/
│   ├── mini_agent_code_build.py
│   ├── agent_workspace/
│   └── agent-code-build/
│       └── agent_runtime/
│           ├── __init__.py
│           ├── common.py
│           ├── context_manager.py
│           ├── history_compactor.py
│           ├── message_types.py
│           ├── permission_manager.py
│           ├── provider_adapter.py
│           ├── runtime.py
│           ├── tool_registry.py
│           ├── workspace_store.py
│           └── tools/
│               ├── __init__.py
│               ├── list_attachments.py
│               ├── read_file_chunk.py
│               ├── read_file_range.py
│               ├── read_prompt_history.py
│               ├── read_transcript.py
│               └── search_in_file.py
├── scripts/
│   └── openwebui_chat_cache_gc.sh
└── function 旧版/
    ├── doc/
    └── filters/
        └── context_compaction_filter.py
```

---

## 3. 核心架构

主链路：

```text
Pipeline (mini_agent_code_build.py)
-> Runtime (agent_runtime/runtime.py)
-> ContextManager / HistoryCompactor
-> ProviderAdapter (OpenAI compatible API)
-> ToolRegistry + WorkspaceStore
```

### 模块职责

- `pipelines/mini_agent_code_build.py`
  - Open WebUI Pipeline 入口
  - 读取 valves（环境配置）
  - 请求类型分类（`chat` / `title` / `tags` / `follow_ups` / `citations`）
  - 缓存 chat 上下文并在 meta 请求时复用
  - 支持流式返回状态事件 + 文本

- `pipelines/agent-code-build/agent_runtime/runtime.py`
  - Agent 主循环（多步推理 + 工具调用）
  - 会话状态加载/保存
  - 附件注册与一致性校验
  - 请求级 token 预算控制
  - meta 请求上下文化处理
  - 结果持久化（state/summary/history）

- `pipelines/agent-code-build/agent_runtime/context_manager.py`
  - 组装 `system prompt` 与 `user context`
  - 控制压缩触发与重建后的输入消息结构

- `pipelines/agent-code-build/agent_runtime/history_compactor.py`
  - token 统计
  - 轻压缩（长消息外置 + 替代摘要）
  - 全量摘要压缩（中段历史转 transcript + summary）
  - 紧急硬限制裁剪

- `pipelines/agent-code-build/agent_runtime/workspace_store.py`
  - 会话目录与文件持久化
  - 附件清单、prompt 历史、snapshot/transcript/artifact 管理
  - 文本文件读取、按行范围读取、子串搜索
  - 上传路径映射与安全校验

- `pipelines/agent-code-build/agent_runtime/tools/*`
  - `list_attachments`
  - `search_in_file`
  - `read_file_chunk`
  - `read_file_range`
  - `read_prompt_history`
  - `read_transcript`

---

## 4. 会话持久化与数据落盘

默认工作区根目录（容器内）：

- `MINI_AGENT_WORKSPACE_ROOT=/app/pipelines/agent_workspace`

会话目录结构（每个 chat_id 一套）：

```text
agent_workspace/sessions/<chat_id>/
├── state.json             # 主状态（history、last_answer、预算信息等）
├── attachments.json       # 已注册附件清单
├── snapshot.json          # 压缩快照（summary、transcript_paths 等）
├── prompt_history.json    # 用户 prompt 历史（外置）
├── transcripts/           # 历史转储 .md
├── artifacts/             # 工具大结果外置 .md
├── attachments/           # copy 模式下的附件副本
└── scratch/               # 临时目录
```

---

## 5. 运行方式（Docker Compose）

> 说明：`docker-compose.pipelines.yaml` 依赖 `Dockerfile`（`build.dockerfile: Dockerfile`）。  
> 你需要去官方仓库拉取框架源代码：```git clone https://github.com/open-webui/pipelines.git```

### 5.1 准备环境变量

```powershell
Copy-Item ".\.env.pipelines.example" ".\.env.pipelines"
```

按需编辑 `.env.pipelines` 的关键项：

- `MINI_AGENT_MODEL_API_BASE_URL`：模型服务地址（OpenAI 兼容）
- `MINI_AGENT_MODEL_API_KEY`：模型 API Key（如需要）
- `MINI_AGENT_TARGET_MODEL`：主模型 ID（必填）
- `OPEN_WEBUI_DATA_VOLUME`：Open WebUI 数据卷名（默认 `open-webui_open-webui`）
- `OPENWEBUI_API_BASE_URL` / `OPENWEBUI_API_TOKEN`：供 GC 服务校验 chat 是否存在

### 5.2 启动

```powershell
docker compose --env-file ".\.env.pipelines" -f ".\docker-compose.pipelines.yaml" up -d --build
```

### 5.3 查看日志

```powershell
docker compose -f ".\docker-compose.pipelines.yaml" logs -f pipelines
docker compose -f ".\docker-compose.pipelines.yaml" logs -f pipelines-chat-cache-gc
```

---

## 6. 配置说明（Valves / 环境变量）

配置来源：`pipelines/mini_agent_code_build.py` + `.env.pipelines.example`。

### 6.1 模型接入

- `MINI_AGENT_MODEL_API_BASE_URL`：模型 API Base URL
- `MINI_AGENT_MODEL_API_KEY`：模型 API Key
- `MINI_AGENT_TARGET_MODEL`：主模型
- `MINI_AGENT_SUMMARY_MODEL`：摘要模型（可空，通常可复用主模型）
- `MINI_AGENT_REQUEST_TIMEOUT_SEC`：请求超时
- `MINI_AGENT_MODEL_TEMPERATURE` / `MINI_AGENT_SUMMARY_TEMPERATURE`
- `MINI_AGENT_MAX_OUTPUT_TOKENS` / `MINI_AGENT_SUMMARY_MAX_TOKENS`

### 6.2 上下文预算与压缩阈值

- `MINI_AGENT_MAX_CONTEXT_TOKENS`
- `MINI_AGENT_RESERVED_OUTPUT_TOKENS`
- `MINI_AGENT_AUTOCOMPACT_BUFFER_TOKENS`
- `MINI_AGENT_LIGHTCOMPACT_BUFFER_TOKENS`
- `MINI_AGENT_MAX_CONSECUTIVE_COMPACTION_FAILURES`
- `MINI_AGENT_KEEP_FIRST_MESSAGES`
- `MINI_AGENT_KEEP_LAST_MESSAGES`
- `MINI_AGENT_KEEP_LAST_USER_MESSAGES`
- `MINI_AGENT_EMERGENCY_TAIL_KEEP_MESSAGES`
- `MINI_AGENT_ENABLE_HISTORY_COMPACTION`

### 6.3 文件/工具预算

- `MINI_AGENT_FILE_EXCERPT_CHARS`
- `MINI_AGENT_TOOL_MAX_RESULTS`
- `MINI_AGENT_TOOL_MAX_FILE_CHARS`
- `MINI_AGENT_TOOL_MAX_TRANSCRIPT_CHARS`
- `MINI_AGENT_TOOL_RESULT_INLINE_BUDGET_CHARS`
- `MINI_AGENT_LIGHTCOMPACT_MAX_SUMMARY_LINES`
- `MINI_AGENT_LIGHTCOMPACT_LINE_CHARS`

### 6.4 工作区与上传路径安全

- `MINI_AGENT_WORKSPACE_ROOT`
- `MINI_AGENT_OPENWEBUI_UPLOADS_ROOT`
- `MINI_AGENT_PIPELINE_MOUNTED_UPLOADS_ROOT`
- `MINI_AGENT_ENFORCE_UPLOAD_ROOT`：是否限制附件路径必须在允许根目录内
- `MINI_AGENT_FILE_REGISTRATION_MODE`：
  - `reference`：引用原始路径（默认）
  - `copy`：复制到会话附件目录

---

## 7. 请求类型与执行逻辑

### Chat 请求

1. Pipeline 缓存上下文（含 files/messages）
2. Runtime 解析 chat_id，加载 state
3. 同步 prompt history
4. 收集并注册附件
5. 进入 agent loop（模型调用 + 工具调用）
6. 必要时压缩历史并外置长内容
7. 产出最终回答并保存 state

### Meta 请求（`title/tags/follow_ups/citations`）

- 从缓存 chat 上下文中提取可见历史
- 构造紧凑上下文 + 专项指令
- 走一次 stateless 或上下文化请求
- `citations` 场景可回写结果到会话历史

---

## 8. 工具能力（只读）

- `list_attachments`：列出已登记附件
- `search_in_file`：文件内搜索关键词
- `read_file_chunk`：从某行起按字符预算读块（适合大文件连续扫描）
- `read_file_range`：按起止行读取
- `read_prompt_history`：恢复外置 prompt 历史
- `read_transcript`：读取历史 transcript/artifact

> 设计原则：只读 + 可恢复，优先保证分析链路稳定性和可追踪性。

---

## 9. Chat Cache GC（会话缓存清理）

`pipelines-chat-cache-gc` 服务使用脚本：`scripts/openwebui_chat_cache_gc.sh`。  
功能：周期扫描会话目录，向 Open WebUI 校验 chat 是否仍存在，对“已不存在且足够旧”的会话执行删除（支持 dry-run）。

关键变量：

- `CHAT_CACHE_GC_INTERVAL_SEC`：扫描周期
- `CHAT_CACHE_GC_MIN_AGE_SEC`：最小保留年龄
- `CHAT_CACHE_GC_DRY_RUN`：仅打印，不删除
- `CHAT_CACHE_GC_RUN_ONCE`：只执行一次
- `OPENWEBUI_API_BASE_URL` / `OPENWEBUI_CHAT_CHECK_URL_TEMPLATE`
- `OPENWEBUI_API_TOKEN` + 认证头相关参数

**注**：你可以直接修改里面循环改成cron定时任务形式

---

## 10. 常见问题排查

### Q1: 服务启动但无可用模型

检查：

- `MINI_AGENT_TARGET_MODEL` 是否为空
- `MINI_AGENT_MODEL_API_BASE_URL` 是否可从容器访问
- `MINI_AGENT_MODEL_API_KEY` 是否需要且已配置

### Q2: 附件找不到或无法读取

检查：

- Open WebUI 数据卷是否正确挂载到 `open-webui-data`
- `MINI_AGENT_OPENWEBUI_UPLOADS_ROOT` 是否与 Open WebUI 实际 uploads 路径一致
- 如使用路径映射，确认 `MINI_AGENT_PIPELINE_MOUNTED_UPLOADS_ROOT`
- `MINI_AGENT_ENFORCE_UPLOAD_ROOT=true` 时，路径越界会被拒绝

### Q3: 回答提示“未提供附件”，但实际上传了

系统内有附件一致性重试逻辑；仍出现时建议：

- 查看 `pipelines` 日志中的 `attachments_registered`、`attachment_conflict_retry`
- 确认请求体 `files` 字段在 inlet/pipe 流程中是否存在

### Q4: 上下文过长导致表现不稳定

可调低压力参数：

- 降低 `MINI_AGENT_MAX_OUTPUT_TOKENS`
- 提高压缩积极性（调整 `*_BUFFER_TOKENS`）
- 降低工具结果内联预算 `MINI_AGENT_TOOL_RESULT_INLINE_BUDGET_CHARS`

### Q5: GC 误删风险

先启用：

- `CHAT_CACHE_GC_DRY_RUN=true`
- `CHAT_CACHE_GC_RUN_ONCE=true`

验证日志后再切换为真实删除。

---

## 11. 开发与调试建议

- 打开调试：`MINI_AGENT_DEBUG=true`
- 重点观察日志事件：
  - `pipeline:init / pipe_enter`
  - `runtime:pipe_start / agent_step_start / attachments_registered`
  - `runtime:request_budget_applied / tool_result_budget_applied`
  - `workspace:register_uploads / save_state / save_snapshot`
- 若要分析压缩效果，优先查看会话目录中的：
  - `snapshot.json`
  - `transcripts/*.md`
  - `artifacts/*.md`
  - `prompt_history.json`

---

## 12. 当前边界

- 当前工具集为只读，不包含写文件/打补丁/执行命令能力
- 摘要质量依赖模型能力
- transcript/artifact 也可能很长，恢复时仍需分段读取

---

## 13. 参考文档

- `docs/github_project_description_doc.md`
- `function 旧版/doc/*`（历史方案与参数说明留档）
- `function 旧版/filters/context_compaction_filter.py`（旧版实现）
- [官方源仓库](https://github.com/open-webui/pipelines#)
- `The newest Claude Code open source`



