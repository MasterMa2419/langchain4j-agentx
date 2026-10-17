# 普通对话 Agent（ChatAgent）

1.0 起框架除了 ReAct 引擎，还提供一种**普通 Agent**：`ChatAgent`。定位是"单轮对话 + 可选会话记忆"，不注册工具、不走 Reasoning + Acting 循环，**一次 LLM 调用直接产出最终答案**。

## 1. 为什么需要两种 Agent

ReAct 引擎（`ReactAgent`）为"多轮推理 + 工具执行"设计，代价是循环控制、HITL、压缩等一整套机制。但大量场景（问答、客服、翻译、摘要）本质是"输入 - 回复"，没有工具循环，使用 ReAct 引擎属于能力过载。

两类 Agent 统一实现 `com.agentx.ai.core.agent.Agent` 接口：

| 能力 | ReactAgent（ReAct） | ChatAgent（普通对话） |
|------|---------------------|----------------------|
| LLM 调用次数 | 多轮循环直到收敛 | 固定 1 次 |
| 工具调用 / ToolSearch / MCP | ✅ | ❌ 不注册工具规格 |
| HITL（ask_user / 拦截审批） | ✅ | ❌ |
| 中断恢复（PauseState） | ✅ | ❌ |
| 上下文压缩（6 层策略链） | ✅ | ❌ |
| SubAgent 委派 | ✅ | ❌ |
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
LLM 流式调用（单次，含透明重试）
  ↓
思考内容 / 正文分片还原为 Thinking / Text 事件
  ↓
终态落库（agentx_session 原始消息链 + working_messages、agentx_conversation 状态回写、agentx_trace 单轮记录）
  ↓
Complete 事件（含 token 统计与 sessionId）
```

执行内核在 `agent/internal/ChatExecutor`，与 `AgentLoopExecutor`（ReAct 内核）共享 `LoopMessageBuilder` / `ThinkingModeProcessor` / `LlmInvoker` / `SessionPersister` / `HookManager` / `AgentTaskManager`，因此行为口径（落库时机、trace 记录方式、重试逻辑、事件类型）与 ReactAgent 保持一致。

## 3. 快速上手

```java
ChatAgent agent = ChatAgent.builder()
        .models(AgentModels.of(chatModel))          // 与 ReactAgent 相同的模型注入方式
        .dataSource(dataSource)                     // 传入后同一 conversationId 拥有会话记忆
        .instructions("你是一个简洁友好的中文助手")   // 系统指令
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
| `thinkingMode` | 思考输出格式（THINK_TAG / REASONING_CONTENT） | DISABLED |
| `maxRetries` | LLM 调用最大重试次数 | 3 |
| `taskManager` | 任务管理器（并发控制 / 停止） | 内存默认实现 |

## 4. 会话记忆与结构化输出

会话记忆口径与 ReactAgent 完全一致：

- 调用参数带 `conversationId` 且配置了 `dataSource` 时，本轮 `UserMessage` / `AiMessage` 终态批量写入 `agentx_session`（原始消息链 + `working_messages` 视图）。
- 下一轮加载历史：优先 `working_messages`，回退 `original_messages`。
- `RunnableParams.outputType(...)` 结构化输出同样生效（user 消息注入 JSON 格式指令，`JsonRepairUtil` 修复返回）。

## 5. 作为 SubAgent 被委派

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

因为 `ChatAgent` 不持有工具，父 Agent 的委派合规校验（禁止嵌套 SubAgent / AskUser）天然通过。委派执行时透传父 Agent 的 `RunnableParams`，流式场景下子代理的事件会打上 `SubAgentSource` 标签转发到父事件流（前端可用 `source != null` 区分来源）。

## 6. 什么场景用什么

- 纯对话、问答、翻译、摘要 → `ChatAgent`（更轻）
- 需要工具 / MCP / HITL / 长上下文 / 子代理委派 → `ReactAgent`
- 两类 Agent 可面向 `Agent` 接口混用：同一个业务入口按配置路由到不同实现
