# 普通对话 Agent（ChatAgent）

1.0 起框架除了 ReAct 引擎，还提供一种**普通 Agent**：`ChatAgent`。定位是"单轮对话 + 可选会话记忆 + 基础工具调用"，不做 Reasoning + Acting 自主循环——至多执行 `maxToolRounds` 轮工具调用（默认 1 轮），随后一轮必然产出最终答案，**LLM 调用次数上限为 `maxToolRounds + 1`**。

## 1. 为什么需要两种 Agent

ReAct 引擎（`ReactAgent`）为"多轮推理 + 工具执行"设计，代价是循环控制、HITL、压缩等一整套机制。但大量场景（问答、客服、翻译、摘要）本质是"输入 - 回复"，即便需要工具也只是"查一次再回答"，使用 ReAct 引擎属于能力过载。

两类 Agent 统一实现 `com.agentx.ai.core.agent.Agent` 接口：

| 能力 | ReactAgent（ReAct） | ChatAgent（普通对话） |
|------|---------------------|----------------------|
| LLM 调用次数 | 多轮循环直到收敛 | ≤ `maxToolRounds` + 1（默认 2 次） |
| 基础工具调用（Function Calling） | ✅ | ✅（有界轮次，默认 1 轮） |
| 工具发现 / ToolSearch / MCP / 沙箱工具 | ✅ | ❌ |
| HITL（ask_user / 拦截审批） | ✅ | ❌ |
| 中断恢复（PauseState） | ✅ | ❌ |
| 上下文压缩（6 层策略链） | ✅ | ❌ |
| SubAgent 委派（作为父 Agent） | ✅ | ❌ |
| 沙箱隔离 | ✅ | ❌ |
| 会话记忆（agentx_session） | ✅ | ✅ |
| 调用边界（agentx_conversation） | ✅ | ✅ |
| Trace 审计（agentx_trace） | ✅ | ✅ |
| Hook 生命周期 | ✅ | ✅（BeforeCall / BeforeReasoning / AfterReasoning / AfterCall / Error） |
| 思考模型适配（think 标签 / reasoning_content） | ✅ | ✅ |
| 流式输出 / 停止 | ✅ | ✅ |
| 结构化输出（outputType） | ✅ | ✅ |

## 2. 执行链路

```text
用户请求
  ↓
LoopMessageBuilder：系统指令 + 会话历史 + 输出格式指令（共同基础设施，与 ReactAgent 相同）
  ↓
LLM 流式调用（携带工具规格，含透明重试）  ←────────────┐
  ↓                                                    │
思考内容 / 正文分片还原为 Thinking / Text 事件            │
  ↓                                                    │
有无 tool calls？                                        │
  ├─ 无（或工具轮次已用尽）→ 收敛最终答案                  │
  └─ 有且未超 maxToolRounds → 执行工具 → 回填结果 ────────┘
  ↓
终态落库（agentx_session 原始消息链 + working_messages、agentx_conversation 状态回写、agentx_trace 逐轮记录）
  ↓
Complete 事件（含 token 统计与 sessionId）
```

执行内核在 `agent/internal/ChatExecutor`，与 `AgentLoopExecutor`（ReAct 内核）共享 `LoopMessageBuilder` / `ThinkingModeProcessor` / `LlmInvoker` / `ToolCallExecutor` / `SessionPersister` / `HookManager` / `AgentTaskManager`，因此行为口径（落库时机、trace 记录方式、重试逻辑、事件类型）与 ReactAgent 保持一致。区别只在于循环是**有界**的：工具轮次由 `maxToolRounds` 限制，用尽后最后一轮不再携带工具规格。

## 3. 快速上手

```java
ChatAgent agent = ChatAgent.builder()
        .models(AgentModels.of(chatModel))          // 与 ReactAgent 相同的模型注入方式
        .dataSource(dataSource)                     // 传入后同一 conversationId 拥有会话记忆
        .instructions("你是一个简洁友好的中文助手")   // 系统指令
        .tools(AgentTool.fromAnnotated(new WeatherTools()))  // 可选：基础工具调用
        .maxToolRounds(1)                           // 工具轮次上限，默认 1
        .maxRetries(3)
        .build();

// 同步
String answer = agent.call("你好");

// 带会话参数调用（跨轮记忆）
agent.call("介绍一下北京", params);

// 同步 + 完整结果
AgentResult result = agent.callForResult("你好", params);

// 流式（文本流 / 完整事件流）
agent.stream("你好", params).subscribe(System.out::print);
agent.streamForResult("你好", params).doOnNext(TestConfig::printEvent).blockLast();

// 流式中途停止
agent.stopStream(conversationId);
```

Builder 选项一览：

| 选项 | 说明 | 默认 |
|------|------|------|
| `models(AgentModels)` | 推荐：一次注入同步 + 流式模型对 | — |
| `chatModel` / `streamingChatModel` | 兼容写法，等价 `AgentModels.resolve` | — |
| `instructions` | 系统指令 | 无 |
| `dataSource` | 落库（会话记忆 + Trace），按需初始化三张表 | 无 |
| `sessionMessageStore` / `conversationStore` | 自定义存储实现（不依赖 DataSource 时） | null |
| `enableSession` | 是否写 `agentx_session` | true |
| `enableTrace` | 是否写 `agentx_trace` | true |
| `hooks` | Hook 生命周期 | 无 |
| `tools` | 注册基础工具（Function Calling），支持可变参数或 `List<AgentTool>` | 无 |
| `maxToolRounds` | 允许的工具执行轮数；0 表示不下发工具 | 1 |
| `thinkingMode` | 思考输出格式（THINK_TAG / REASONING_CONTENT） | DISABLED |
| `maxRetries` | LLM 调用最大重试次数 | 3 |
| `taskManager` | 任务管理器（并发控制 / 停止） | 内存默认实现 |

## 4. 会话记忆与结构化输出

会话记忆口径与 ReactAgent 完全一致：

- 调用参数带 `conversationId` 且配置了 `dataSource` 时，本轮 `UserMessage` / `AiMessage` 终态批量写入 `agentx_session`（原始消息链 + `working_messages` 视图）。
- 下一轮加载历史：优先 `working_messages`，回退 `original_messages`。
- `RunnableParams.outputType(...)` 结构化输出同样生效（user 消息注入 JSON 格式指令，`JsonRepairUtil` 修复返回）。

## 5. 工具调用（有界轮次）

`ChatAgent` 通过 `tools(...)` 注册基础 Function Calling 工具：LLM 决定调用 → 框架执行 → 结果回填 → 继续本轮回答。与 ReAct 的区别是**轮次有界**：

| `maxToolRounds` | 行为 | 最多 LLM 调用 |
|-----------------|------|---------------|
| `1`（默认） | 执行 1 轮工具调用，随后一轮不带工具规格产出最终答案 | 2 |
| `N` &gt; 1 | 允许链式调用（如先检索再取详情），最多 N 轮工具 | N + 1 |
| `0` | 完全不向 LLM 暴露工具，等价纯对话 | 1 |

```java
ChatAgent agent = ChatAgent.builder()
        .models(models)
        .tools(new WeatherTool(), new TimeTool())   // 或 tools(List<AgentTool>)
        .maxToolRounds(2)                           // 允许链式调用（如先检索再取详情）
        .build();
```

行为要点：

- **工具消息入链**：工具调用 AiMessage 与 `ToolExecutionResultMessage` 会写入 `agentx_session`，后续轮次可续聊（会话记忆保持完整消息链）。
- **事件可见**：流式场景下，工具执行前后发 `ToolStart` / `ToolEnd` 事件，前端可展示"调用了什么工具、参数与结果"。
- **trace 逐轮记录**：每轮 LLM 调用各写一条 `agentx_trace`（工具调用轮记录 tool_calls，最终轮记录答案）。
- **最终答案只取最后一轮**：引擎在收到 `ToolStart` 时清空文本缓冲，`call()` 返回值即最终回答，不含工具轮的前导文本。

不覆盖的能力：HITL 工具拦截（`ask_user` / 审批暂停）、ToolSearch 延迟加载、MCP、沙箱内工具。需要这些时使用 `ReactAgent`。

示例：`langchain4j-agentx-samples` → `v1_1/ChatAgentToolTest`（真实 API + MySQL，验证工具消息入链与逐轮 trace）。

## 6. 作为 SubAgent 被委派

`ChatAgent` 可注册为父 Agent（`ReactAgent`）的子代理，由父 Agent 的 LLM 根据描述决定是否委派：

```java
ReactAgent parent = ReactAgent.builder()
        .models(models)
        .subAgent(() -> ChatAgent.builder()
                .name("translator")                 // 必填：生成 call_translator 工具名
                .description("中英互译助手")          // 必填：父 LLM 据此决定是否委派
                .models(subModels)
                .build())
        .build();
```

框架在委派前会自动把子代理切到 SubAgent 模式（`Agent#configureAsSubAgent`）：

- **关闭子代理自身会话持久化**：父 Agent 是唯一的 `agentx_session` 边界，子代理不写自己的会话
- **复用父 Agent 的 TraceStore**：trace 记录挂在父 Agent 的审计链上（父关闭 trace 时子代理也不记录）

因为 `ChatAgent` 作为子代理时只执行自己注册的基础工具，父 Agent 的委派合规校验（禁止 `call_*` 嵌套 SubAgent / `ask_user`）只需确保其工具清单不含这些；默认（不注册工具）天然通过。委派执行时透传父 Agent 的 `RunnableParams`，流式场景下子代理的事件会打上 `SubAgentSource` 标签转发到父事件流（前端可用 `source != null` 区分来源）。

## 7. 什么场景用什么

- 纯对话、问答、翻译、摘要 → `ChatAgent`（更轻）
- "查一次再回答"的浅层工具场景 → `ChatAgent` + `tools(...)`
- 需要多轮推理、MCP、HITL / 长上下文 / 子代理委派 / 沙箱 → `ReactAgent`
- 两类 Agent 可面向 `Agent` 接口混用：同一个业务入口按配置路由到不同实现
