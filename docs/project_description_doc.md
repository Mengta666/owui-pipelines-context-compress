# Mini Agent Code Build 项目说明文档

---

## 1. 项目定位

Mini Agent Code Build 是一个运行在 Open WebUI Pipeline 中的轻量级 Agent 运行时。项目的重点并不在“多工具堆叠”本身，而在于：在**多轮对话、带附件、长文件分析、上下文持续增长**的条件下，仍然能够稳定地完成代码分析与文档分析任务。

从能力边界看，它更接近一个：

- **长上下文代码/文档分析 Agent**
- **可恢复会话状态的工具型运行时**
- **带工作区与外部化存储的上下文治理系统**

而不是一个以代码写入、执行测试、自动修复为主的自治编程框架。

---

## 2. 项目目标

该项目主要解决以下几类问题：

### 2.1 多轮会话持续增长
普通聊天式上下文在多轮任务中会快速膨胀，最终超过模型窗口。项目通过分层压缩、摘要融合、外部 transcript / artifact 存储，解决历史不可持续增长的问题。

### 2.2 附件与长文件无法直接塞入 prompt
代码仓库、长文档、日志文件、转储文本等内容通常不适合直接内联到模型上下文中。项目通过工作区注册与按需读取工具，让模型能够在预算内渐进式读取文件内容。

### 2.3 压缩后不能“失忆”
被移出 prompt 的内容不能直接丢弃，否则多轮任务会出现上下文断层。项目将被压缩的内容写入 transcript / artifact，并提供恢复工具，保证上下文压缩是**可逆的**。

### 2.4 避免模型对未读内容进行脑补
针对附件和长文件，系统通过 system prompt、user context、file read progress、chunk meta 等多层约束，持续提醒模型：**未读完就不要假装已经完整分析**。

---

## 3. 整体架构概览

项目整体可以概括为如下链路：

```text
Pipeline -> Runtime -> ContextManager -> ProviderAdapter
         -> Tool Loop -> WorkspaceStore -> HistoryCompactor
```

每一层的职责都比较明确：

- **Pipeline**：接入 Open WebUI 生命周期，接收请求、分类请求、缓存上下文。
- **Runtime**：Agent 主调度器，负责会话状态加载、历史维护、工具循环、模型调用。
- **ContextManager**：负责把“系统规则 + 当前工作状态 + 可见历史”重建为下一轮模型输入。
- **ProviderAdapter**：适配 OpenAI 风格 chat/completions。
- **Tool Loop**：负责执行只读工具，扩展模型的文件读取能力与恢复能力。
- **WorkspaceStore**：负责会话文件落盘、附件注册、transcript / artifact / state 持久化。
- **HistoryCompactor**：负责长上下文治理，包括轻压缩、摘要融合与紧急裁剪。

---

## 4. 项目结构

```text
mini_agent_code_build.py                    # Open WebUI Pipeline 入口
agent-code-build/
└── agent_runtime/
    ├── common.py                           # 公共工具：日志、内容扁平化、digest 等
    ├── message_types.py                    # history item 标准化与去重
    ├── permission_manager.py               # 工具权限控制（当前仅放行只读工具）
    ├── provider_adapter.py                 # OpenAI 兼容 chat/completions 适配层
    ├── workspace_store.py                  # 会话目录、附件注册、转储、文件读取、prompt history 持久化
    ├── tool_registry.py                    # 工具 schema 注册与执行分发
    ├── context_manager.py                  # system prompt / user context 组装，压缩触发入口
    ├── history_compactor.py                # 轻压缩、摘要压缩、转储、硬限制裁剪的核心
    ├── runtime.py                          # Agent 主循环
    └── tools/
        ├── list_attachments.py             # 查看已注册附件
        ├── search_in_file.py               # 子串搜索
        ├── read_file_chunk.py              # 按预算自动连续读块
        ├── read_file_range.py              # 显式按行读范围
        ├── read_prompt_history.py          # 恢复被截断的 prompt 历史
        └── read_transcript.py              # 恢复被压缩移出的 transcript / artifact
```

从架构重要性上看，最核心的实现集中在以下模块：

- `runtime.py`
- `context_manager.py`
- `history_compactor.py`
- `workspace_store.py`

它们共同定义了项目最关键的能力：**长上下文的压缩、外置、恢复与重组**。

---

## 5. 核心设计思想

### 5.1 不是“全量 prompt 回放”，而是“工作态重建”
项目并不尝试把全部历史和全部文件内容塞进模型，而是只构建当前轮继续工作所需的最小上下文集合。系统更关注“当前该知道什么”，而不是“历史原文必须全部重现”。

### 5.2 不是“删历史”，而是“外置历史”
一旦历史或工具结果过长，系统不会直接丢弃，而是将原始内容写入 transcript / artifact，并在历史中留下摘要与恢复入口。

### 5.3 不是“一次性读完整文件”，而是“渐进式读取”
大文件通过 `read_file_chunk`、`read_file_range`、`search_in_file` 等工具按需读取，并将读取进度写回状态，以支持多轮连续分析。

### 5.4 不是“单一总结”，而是“分层压缩”
系统同时区分：

- prompt history 的截断与恢复
- 工具结果的预算外置
- 历史消息的轻压缩
- 中段历史的摘要压缩
- 极端场景下的预算硬裁剪

这意味着项目对不同类型的“长内容”采取不同治理策略，而不是统一粗暴压缩。

---

## 6. 核心模块说明

### 6.1 `mini_agent_code_build.py`
作为 Open WebUI Pipeline 入口，负责：

- 读取 Valves / 配置项
- 分类请求类型（chat / title / tags / follow_ups / citations 等）
- 缓存最近一次 chat 的上下文
- 调度 runtime
- 处理流式输出

它更多承担接入层职责，不直接参与具体上下文治理逻辑。

### 6.2 `runtime.py`
这是主执行器。其主要职责包括：

- 解析请求类型
- 加载 session state
- 同步 prompt history
- 注册附件
- 维护会话 history
- 驱动模型调用与工具调用循环
- 在回答结束后写回 state / summary / last answer

它的运行方式接近一个标准的 tool-using agent loop，但额外引入了附件一致性校验、工具结果外置、会话压缩治理等工程性能力。

### 6.3 `context_manager.py`
该模块负责把会话状态重组为模型可消费的输入消息，尤其是构造两条最关键的前置消息：

- `system prompt`
- `user context`

其中：

- `system prompt` 用于定义运行规则、工具使用方式、文件读取约束、恢复逻辑。
- `user context` 用于告诉模型当前会话工作区状态，例如附件、transcript、文件读取进度、prompt history 窗口、当前读块预算等。

这一层决定了：模型看到的不是原始状态，而是**被整理后的工作态视图**。

### 6.4 `history_compactor.py`
这是长上下文治理的核心模块，负责：

- token 统计
- 轻压缩（lightcompact）
- 全量摘要压缩（full compaction）
- transcript 写入
- continuation summary 融合
- 紧急预算裁剪

它的目标不是“尽快做摘要”，而是在不同压力阶段采取不同强度的压缩策略，尽量保留任务连续性与恢复能力。

### 6.5 `workspace_store.py`
该模块是系统的外部化存储层，负责维护：

- `state.json`
- `attachments.json`
- `snapshot.json`
- `prompt_history.json`
- `transcripts/*.md`
- `artifacts/*.md`

它使得压缩后的历史和工具输出能够在后续轮次中恢复，从而让“长上下文治理”具备工程闭环。

### 6.6 `tools/*`
当前工具集以只读能力为主，主要包括：

- 附件枚举：`list_attachments`
- 文件搜索：`search_in_file`
- 连续读块：`read_file_chunk`
- 范围读取：`read_file_range`
- 恢复 prompt 历史：`read_prompt_history`
- 恢复 transcript / artifact：`read_transcript`

这些工具共同构成了模型的“外部记忆访问接口”。

---

## 7. 运行机制

### 7.1 Chat 请求主流程

在 chat 请求下，项目的执行路径大致如下：

```text
Pipeline.inlet
-> 缓存 chat 上下文
-> Pipeline.pipe
-> Runtime.pipe
-> 加载 / 初始化 session state
-> 同步 prompt history
-> 注册附件
-> 追加当前轮用户消息到 history
-> prepare_turn
-> build_messages
-> 调模型
-> 如有 tool_calls 则执行工具并回写 history
-> 循环直到得到最终文本
-> 保存 state / summary / answer
```

该流程的关键点不在“调一次模型”，而在于：

- 每一轮前都会重新构造上下文
- 长历史可在进入模型前被压缩
- 长文件可在循环内渐进读取
- 工具结果会变成下一轮的工作状态输入

### 7.2 Meta 请求
对于 `title`、`tags`、`follow_ups` 等请求，系统不会进入完整的 agent loop，而是：

- 从之前缓存的 chat visible history 中取材料
- 构造一个更紧凑的上下文
- 追加专项 meta 指令
- 发起一次 stateless 调用
- 返回结构化结果

因此，meta 请求属于“基于聊天上下文的一次性派生生成”，而不是“继续推进会话”。

---

## 8. 长上下文治理机制

这是项目最核心的能力。

### 8.1 分层治理目标
系统并不把“上下文超长”视为单一问题，而是把它拆成几类子问题：

1. 历史消息整体太长
2. 单条消息太长
3. 工具结果太长
4. prompt history 太长
5. 文件本身太长

不同问题采用不同策略。

### 8.2 轻压缩（Lightcompact）
对于历史中的超长 assistant 文本或 tool result，系统会：

- 将完整原文写入 artifact 文件
- 在 history 中插入摘要替代内容
- 保留恢复指针

其特点是：

- 不破坏整体历史结构
- 不强制改写会话语义
- 把“长正文”转为“摘要 + 恢复入口”

这属于第一层温和压缩。

### 8.3 全量摘要压缩（Full Compaction）
如果整体上下文仍然超长，则系统会对中段历史做摘要压缩，过程包括：

- 对历史按逻辑组分组
- 保留早期锚点消息
- 保留最近工作集消息
- 保留近期用户约束
- 将中间 delta 历史写入 transcript
- 生成或合并 continuation summary
- 用 `memory_summary + early + recent` 重建 history

该机制的目标是：

> 把“完整中段历史”变成“可恢复摘要态”，同时维持当前轮继续工作的上下文完整性。

### 8.4 Prompt History 外置
除了普通 history，系统还单独维护 `prompt_history.json`，并只把尾部窗口内联到 `user context`。更早的 prompt 可以通过 `read_prompt_history` 恢复。

这样做的意义在于：

- 用户约束不完全依赖普通 history
- 即使旧 assistant 输出已经压缩，最近用户要求仍然能优先保留

### 8.5 请求级预算裁剪
在真正发起模型请求前，系统还会对单次请求做一次预算控制，包括：

- 保留 `system`
- 保留 `user context`
- 优先移除中间低优先级消息
- 必要时裁剪单条超长内容

这保证了每次请求都能在窗口约束内稳定发出。

---

## 9. 长文件处理机制

### 9.1 文件不会默认进入 prompt
附件和工作区文件默认不会全文内联到模型输入中。模型只会先看到：

- 文件清单
- 工作区路径
- 最近文件读取进度
- 工具使用规则

这能显著降低 prompt 膨胀风险。

### 9.2 `read_file_chunk`
`read_file_chunk` 是项目处理长文本的核心工具。它的作用是：

- 根据当前预算自动决定读取块大小
- 默认从连续未读区间的下一行继续
- 返回显式的进度信息，如：
  - 当前读取范围
  - 连续覆盖范围
  - 是否 EOF
  - 剩余行数
  - 下一段建议起点

这意味着文件分析是**带状态的顺序扫描**，而不是模型随意猜测阅读边界。

### 9.3 `read_file_range`
用于显式读取指定行范围，适合模型已经知道需要精确定位的场景。

### 9.4 `search_in_file`
用于做快速定位，减少对大文件的盲读。

### 9.5 文件读取进度参与下一轮上下文重建
系统会把文件读取状态写回 `user context`，使模型在下一轮中知道：

- 哪些段已经读过
- 哪些内容尚未覆盖
- 如果继续读，应从哪里开始

这使多轮文件分析变成一个可持续的状态化过程。

---

## 10. 上下文重组机制

### 10.1 输入不是“旧历史原文”，而是“状态与摘要”
下一轮模型实际收到的上下文，通常由以下几部分组成：

- `system prompt`
- `user context`
- 可见历史（包括 memory summary、近期消息、必要工具轨迹）
- 必要时的 retry instruction

其中真正关键的是：

- `system prompt`：定义行为约束
- `user context`：定义当前工作态

### 10.2 `user context` 的作用
`user context` 是项目的核心设计之一。它承载的信息通常包括：

- workspace 路径
- attachments 清单
- recent transcripts
- recent file read progress
- 当前 chunk-read budget
- prompt history excerpt
- 文件工具与恢复工具的使用规则

因此它不是普通 user message，而更像一张“当前轮工作目录页”。

### 10.3 按需恢复而非全量展开
如果模型需要更老的历史或更完整的文件内容，它不会自动拿到全文，而是需要主动调用：

- `read_transcript`
- `read_prompt_history`
- `read_file_chunk`
- `read_file_range`
- `search_in_file`

这说明系统的重组逻辑不是“回放原文”，而是“按需恢复局部细节”。

---

## 11. 适合解决的问题类型

该项目尤其适合以下任务类型：

- 大型代码仓库的结构梳理
- 长文档 / 规范 / 说明书分析
- 多轮迭代的设计讨论
- 带附件的代码审阅与问题定位
- 需要连续读文件、补充上下文、逐步收敛结论的任务

它特别适合“答案依赖外部文件且文件本身很长”的场景。

---

## 12. 当前边界与局限

### 12.1 偏重读取与分析，不支持完整写入闭环
当前工具集主要是只读工具，尚未提供：

- 写文件
- patch 文件
- 执行命令
- 跑测试
- 代码自动修改闭环

因此该项目更适合作为分析型 Agent 基座，而不是完整的自治开发系统。

### 12.2 恢复能力以文件级工具为主
虽然 transcript / artifact 可恢复，但当前恢复方式更偏“文件回读”，尚未引入更强的语义检索或片段级精准召回。

### 12.3 摘要质量依赖模型
full compaction 生成的 summary 本质上仍然依赖摘要模型质量。摘要模型越稳定，会话长期保持一致性的能力越强。

### 12.4 超长 transcript 仍可能再次需要分段读取
即使 transcript 可恢复，其内容本身也可能很长，因此恢复操作依然可能需要进一步分段处理。

---

## 13. 后续可能扩展方向

如果要把该项目进一步演化为更完整的工程 Agent，可以优先考虑以下方向：

### 13.1 增加写入类工具
例如：

- 写文件
- 应用补丁
- 批量修改代码
- 生成新文件

### 13.2 增加执行与验证能力
例如：

- 运行命令
- 跑测试
- 执行静态检查
- 读取命令输出并回写状态

### 13.3 增加更精细的检索层

由于语义召回问题，长文阅读下**结果可能不太准确**（待解决），可通过如下方式进一步增强1结果准确性。

例如：

- transcript 语义检索
- artifact 索引
- 文件 chunk embedding 检索
- prompt / tool trace 的语义回收

### 13.4 增加更强的状态诊断能力
例如：

- 当前上下文占用视图
- 已压缩历史可视化
- file read progress dashboard
- summary 漂移检测

## 结论

Mini Agent Code Build 的核心价值不在于“做了多少工具”，而在于它把一个工具型 Agent 在真实多轮任务中最难的部分落到了工程实现上：

- 会话状态如何持续
- 长上下文如何压缩
- 被压缩内容如何恢复
- 长文件如何渐进式读取
- 当前工作状态如何在下一轮被重组给模型

它的本质不是一个普通聊天代理，而是一个围绕**长上下文治理、外部化记忆、工作态重建、只读型文件分析**设计的 Agent 运行时。
