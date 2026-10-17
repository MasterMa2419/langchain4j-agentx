# AgentX

基于 LangChain4j 的智能体（Agent）开发框架，提供 ReAct 执行引擎、当前会话记忆、长期记忆、上下文压缩、工具调度、Human-in-the-Loop 等核心能力，帮助开发者快速构建可落地的 Java Agent。

> 当前版本：**1.0.0**
>
> JDK 21+ | Spring Boot 3.5.x | LangChain4j 1.19.0

## 模块结构

| 模块 | 说明 |
|------|------|
| `langchain4j-agentx-core` | Agent 执行引擎核心：ReAct 主循环、记忆、上下文压缩、工具调度、HITL、Hook、沙箱 |
| `langchain4j-agentx-rag` | AgentX 之上的 RAG Harness：解析 → 分块 → 存储 → 查询增强 → 检索 → 编排 |
| `langchain4j-agentx-samples` | 可直接运行的示例模块，覆盖各能力域 |

## LangChain4j AgentX 是什么

LangChain4j AgentX 是一款面向 Java 开发者的 AI Agent 开发框架。框架基于 LangChain4j 与 Reactor 构建，专注 Agent 执行引擎本身，不引入额外的 Graph 编排范式，尽量复用 LangChain4j 原生能力完成多轮推理、工具调用、会话持久化与执行控制。

### 设计理念

- 不造新范式，只做 Agent 引擎
- 不用 Graph，基于 Reactor 驱动多轮执行
- 把模型之外的能力统一收口到 Harness：工具调度、当前会话状态、上下文压缩、HITL、追踪审计、技能体系

## 能力总览

| 能力 | 说明 |
|------|------|
| ReAct Agent 引擎 | 基于 Reasoning + Acting 驱动多轮执行闭环，统一处理 LLM 调用、工具调用与终态收敛 |
| 普通 Chat Agent 引擎 | `ChatAgent` 普通对话：默认单轮一次 LLM 出答案；可选基础工具调用（`tools` + 有界 `maxToolRounds`，默认 1 轮）；复用会话记忆、流式输出、持久化与 Hook |
| 同步与流式输出 | `call` / `stream` 双模式，流式基于 Reactor Flux，支持流式中途停止 |
| 当前会话记忆 | `agentx_conversation` 记录每次调用边界；`agentx_session` 按 `original_messages`、`working_messages`、`offload_context` 三种状态键维护会话状态 |
| 长期记忆 | 从 `original_messages` 异步抽取跨会话知识，写入外部 EmbeddingStore；每次调用前按 `userId` 语义检索注入 system prompt |
| 上下文压缩 | 6 层渐进式压缩策略链 + 外层门禁 + `lastKeep` 保护区 + `context_reload` 工具回溯（详见下文与 [10-上下文压缩](docs/core/v1_0/10-上下文压缩.md)） |
| 结构化输出 | `RunnableParams.outputType(...)` 按单次调用启用 JSON 输出，不影响同一会话中的普通对话 |
| 工具调度与 MCP | 原生 Function Calling + MCP 协议，工具执行由框架统一接管；支持 ToolSearch 按需发现 |
| 运行时参数注入 | `RunnableParams` 动态覆盖工具参数，精准控制工具行为 |
| 任务管理与并发控制 | 会话级并发控制与中断机制，保障同一会话执行的有序性与可控性 |
| Human-in-the-Loop | `askUser(true)` 默认注册内置 `ask_user` 工具与拦截；`interceptTools(...)` 支持审批类工具与输入类工具两种语义 |
| 中断与恢复 | `agentx_pause_state` 持久化暂停快照，统一支持 `HITL_TOOL_REQUEST` 与 `USER_INTERRUPT` 两种暂停原因 |
| SubAgent 子代理 | ReactAgent 与 ChatAgent 均可注册为 `call_{name}` 工具被委派，子代理拥有独立 context window；父 Agent 是 session 持久化边界 |
| TraceAudit 追踪审计 | `agentx_trace` 记录每轮 LLM 请求、响应、思考内容与 token 消耗 |
| TodoWrite 任务追踪 | 结构化任务列表工具，支持流式 TodoProgress 事件 |
| Skills 技能体系 | 按需加载技能内容，减少大段提示词常驻上下文 |
| 思考模型适配 | 支持 `<think/>` 标签和 `reasoning_content` 两种思考输出格式，`DeepSeekV4ChatModel` 工厂原生回传 reasoning_content |
| 异常处理与重试 | 内置透明重试机制，统一异常处理（AgentException + AgentErrorCode） |
| Hook 生命周期机制 | 7 个 Hook 事件覆盖 Agent 全生命周期，支持 Before* 干预输入和 After* 观测结果，通过 AgentRuntimeContext 操控共享状态 |
| 沙箱隔离执行 | BashTool / FileSystemTools / GrepTool 在 Docker 容器或本地受限目录中执行；调用结束快照持久化，下次调用自动恢复；支持 CONVERSATION / USER 隔离级别与严格模式 fail-closed |

## RAG 模块

框架另提供与 core 并列的 `langchain4j-agentx-rag` 模块，做成一款「AgentX 之上的 RAG Harness」，把文档解析、分块、存储、查询增强、检索与编排这些 RAG 运行时能力框架化，让调用方以最小成本接入完整 RAG Pipeline。

| 能力域 | 说明 |
|--------|------|
| 文档解析 | Tika（简单文本）+ MinerU（多模态，表格/图片混排） |
| 文档分块 | 按长度 / 按符号 / 按标题（父子分块） |
| 存储 | PG / Redis 双向量库 + JDBC / Redis 父块库，按 chunk 角色路由写入 |
| 查询增强 | 问题压缩 / 改写 / HyDE / 多查询扩展 |
| 检索与重排 | 向量 / 父子召回 + 可选 rerank 精排 |
| 编排 | RagPipeline，支持串行 RAG 与 Agentic RAG（RAG 作为工具交给 ReactAgent） |

RAG 模块复用 LangChain4j 的 EmbeddingStore / EmbeddingModel / ChatModel 等原生 SPI，不重复造底层脚手架。功能说明与接入方式见 [docs/rag/README.md](docs/rag/README.md)。

## 快速开始

### 1. 从源码构建

```bash
git clone https://github.com/MasterMa2419/langchain4j-agentx.git
cd langchain4j-agentx
mvn clean install -DskipTests
```

### 2. 引入依赖

```xml
<dependency>
    <groupId>com.agentx.ai</groupId>
    <artifactId>langchain4j-agentx-core</artifactId>
    <version>1.0.0</version>
</dependency>
```

### 3. 构建一个最小可用 Agent

Agent 通过 `AgentModels` 一次注入模型：ReAct 主循环走**流式**（对外 token 级事件），上下文压缩 / 长期记忆 / ToolSearch 等旁路走**同步**。对接 OpenAI 兼容 API 时通常是两个实例，推荐一次构造后交给 `.models(...)`。

```java
import com.agentx.ai.core.agent.ReactAgent;
import com.agentx.ai.core.chatmodels.AgentModels;
import com.agentx.ai.core.context.ContextPolicy;
import com.agentx.ai.core.model.AgentResult;
import com.agentx.ai.core.model.RunnableParams;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.model.openai.OpenAiStreamingChatModel;

// 同步 + 流式一对模型（推荐：真实 token 级流式）
AgentModels models = AgentModels.of(
        OpenAiChatModel.builder()
                .baseUrl("https://dashscope.aliyuncs.com/compatible-mode/v1")
                .apiKey("your-api-key")
                .modelName("qwen-plus")
                .temperature(0.7)
                .build(),
        OpenAiStreamingChatModel.builder()
                .baseUrl("https://dashscope.aliyuncs.com/compatible-mode/v1")
                .apiKey("your-api-key")
                .modelName("qwen-plus")
                .temperature(0.7)
                .build());

DataSource dataSource = ...;

ReactAgent agent = ReactAgent.builder()
        .models(models)                                          // 统一模型入口
        .dataSource(dataSource)                                  // 传入后按需初始化会话与追踪表
        .stateStore(new JdbcPauseStateStore(dataSource))         // 暂停快照持久化
        .contextPolicy(ContextPolicy.defaults())                 // 开启 6 层上下文压缩
        .askUser(true)                                           // 启用 HITL（内置 ask_user）
        .maxRounds(10)
        .build();

RunnableParams params = RunnableParams.builder()
        .conversationId("conv_001")
        .userId("user_123")
        .build();

AgentResult result = agent.callForResult("帮我查一下北京天气，并给我一句穿衣建议", params);
```

**模型注入方式怎么选**

| 方式 | 用法 | 说明 |
|------|------|------|
| `AgentModels.of(sync, streaming)` | 上例 | **推荐**。真实流式 + 同步旁路，配置一次收口 |
| `AgentModels.of(chatModel)` | 单实例 | 实例同时实现 `ChatModel` + `StreamingChatModel` 时自动复用；否则包装为伪流式（整段一次吐出，无 token 级增量）并打 WARN |
| `AgentModels.ofStreaming(streamingChatModel)` | 仅流式 | 同步旁路包装为「收集整段流」，压缩 / 记忆等调用会阻塞至流结束 |
| `.chatModel(...)` / `.streamingChatModel(...)` | 兼容写法 | 等价于 `AgentModels.resolve(...)` 自动解析 |

DeepSeek 等思考模型可用工厂一次拿到 pair：

```java
AgentModels models = DeepSeekV4ChatModel.builder()
        .apiKey("sk-...")
        .modelName("deepseek-v4")
        .buildWithStreaming();

ReactAgent agent = ReactAgent.builder()
        .models(models)
        .thinkingMode(ThinkingMode.REASONING_CONTENT)
        .build();
```

### 单轮对话场景：ChatAgent（普通 Agent）

问答、客服、翻译、摘要等场景可使用 `ChatAgent`——默认一次 LLM 调用直接出答案，不走 ReAct 多轮迭代；需要"查一次再回答"时可注册基础工具：

```java
import com.agentx.ai.core.agent.ChatAgent;

ChatAgent agent = ChatAgent.builder()
        .models(models)                                   // 与 ReactAgent 相同的 AgentModels 注入方式
        .dataSource(dataSource)                           // 传入后同一 conversationId 拥有会话记忆
        .instructions("你是一个简洁友好的中文助手")          // 系统指令（system prompt）
        .tools(AgentTool.fromAnnotated(new WeatherTools()))  // 可选：基础工具调用
        .maxToolRounds(1)                                 // 工具轮次上限，默认 1（LLM 调用 ≤ N+1）
        .maxRetries(3)
        .build();

String answer = agent.call("你好");
agent.stream("你好", params).subscribe(System.out::print);   // token 级流式
```

`ChatAgent` 与 `ReactAgent` 统一实现 `Agent` 接口（`call` / `stream` / `callForResult` / `streamForResult`），两类 Agent 可面向接口编程、混用在同一业务栈中。`ChatAgent` 复用同套基础设施：会话记忆（`agentx_session`）、调用边界（`agentx_conversation`）、Trace（`agentx_trace`）、思考模型适配与 Hook 生命周期。工具支持限于**基础 Function Calling 且轮次有界**；HITL 中断恢复、ToolSearch / MCP、上下文压缩、沙箱、以及作为父 Agent 委派子代理属于 ReAct 引擎能力，需要时请用 `ReactAgent`（详见 [21-普通对话Agent](docs/core/v1_0/21-普通对话Agent.md)）。

`ChatAgent` 也可以反过来作为 **SubAgent** 被 `ReactAgent` 委派（注册为 `call_{name}` 工具）：

```java
ReactAgent parent = ReactAgent.builder()
        .models(models)
        .subAgent(() -> ChatAgent.builder()                     // 普通 Agent 也能被委派
                .name("translator")
                .description("中英互译助手")
                .models(subModels)
                .build())
        .build();
// 父 Agent 的 LLM 根据 description 决定是否调用 call_translator
```

### 4. 数据持久化与开关

传入 `DataSource` 后，框架会按需初始化会话表和追踪表。常用控制项如下：

| 参数 | 控制内容 | 默认值 | 说明 |
|------|----------|--------|------|
| `enableSession` | `agentx_session` 当前会话状态 | `true` | SubAgent 场景会被框架自动关闭 |
| `enableTrace` | `agentx_trace` 审计日志 | `true` | 父 Agent 关闭时，SubAgent 也不再记录 trace |

## 上下文压缩设计

上下文压缩用于让长会话仍能稳定执行，通过 `ContextPolicy` 开启，由 `ContextCompactionHook` 接入主循环。共 6 层策略（对应 `context/compress/strategy` 下的 6 个策略类）：

| 层级 | 策略类 | 处理区 | 是否调 LLM |
|------|--------|--------|------------|
| L1 | `HistoricalToolListStrategy` | 历史区 | 否 |
| L2 | `LargeMsgOffloadWithKeepStrategy` | 历史区 | 否 |
| L3 | `LargeMsgOffloadNoKeepStrategy` | 历史区 | 否 |
| L4 | `HistoricalRoundSummaryStrategy` | 历史区 | 是 |
| L5 | `CurrentRoundLargeMsgStrategy` | 当前任务区 | 是 |
| L6 | `CurrentRoundOverallStrategy` | 当前任务区 | 是 |

即 **L1-L3 不调 LLM，L4-L6 调 LLM**；**L1-L4 处理历史区，L5-L6 处理当前任务区**。压缩后的工作视图覆盖写入 `working_messages`，被移出的原文保存在 `offload_context`，模型可通过 `context_reload` 工具按 uuid 取回。

调用边界独立成 `agentx_conversation` 表：每次 `call` / `stream` 开局写一行，终态回写状态、token 用量与最终回答，调用方查询「某次调用发生了什么」无需聚合 `agentx_session`。

## 文档导航

核心专题文档放在 [docs/core/v1_0](docs/core/v1_0) 下：

| # | 功能 | 文档 |
|---|------|------|
| 1 | 同步与流式输出 | [01-同步与流式输出](docs/core/v1_0/01-同步与流式输出.md) |
| 2 | 工具与 MCP | [02-工具与MCP](docs/core/v1_0/02-工具与MCP.md) |
| 3 | 动态会话参数 | [03-动态会话参数](docs/core/v1_0/03-动态会话参数.md) |
| 4 | 任务管理与并发控制 | [04-任务管理与并发控制](docs/core/v1_0/04-任务管理与并发控制.md) |
| 5 | 分层记忆体系 | [05-分层记忆体系](docs/core/v1_0/05-分层记忆体系.md) |
| 6 | Human-in-the-Loop | [06-Human-in-the-Loop](docs/core/v1_0/06-Human-in-the-Loop.md) |
| 7 | Skills 技能体系 | [07-Skills技能体系](docs/core/v1_0/07-Skills技能体系.md) |
| 8 | 思考模型适配 | [08-思考模型适配](docs/core/v1_0/08-思考模型适配.md) |
| 9 | 异常处理与重试 | [09-异常处理与重试](docs/core/v1_0/09-异常处理与重试.md) |
| 10 | 上下文压缩 | [10-上下文压缩](docs/core/v1_0/10-上下文压缩.md) |
| 11 | ToolSearch 工具检索 | [11-ToolSearch工具检索](docs/core/v1_0/11-ToolSearch工具检索.md) |
| 12 | 结构化输出 | [12-结构化输出](docs/core/v1_0/12-结构化输出.md) |
| 13 | 综合示例 | [13-综合示例](docs/core/v1_0/13-综合示例.md) |
| 14 | DeepSeek-V4 兼容 | [14-DeepSeek-V4兼容](docs/core/v1_0/14-DeepSeek-V4兼容.md) |
| 15 | TodoWrite 任务追踪 | [15-TodoWrite任务追踪](docs/core/v1_0/15-TodoWrite任务追踪.md) |
| 16 | TraceAudit 追踪审计 | [16-TraceAudit追踪审计](docs/core/v1_0/16-TraceAudit追踪审计.md) |
| 17 | SubAgent 子代理 | [17-SubAgent子代理](docs/core/v1_0/17-SubAgent子代理.md) |
| 18 | 中断与恢复 | [18-中断与恢复](docs/core/v1_0/18-中断与恢复.md) |
| 19 | Hook 机制 | [19-Hook机制](docs/core/v1_0/19-Hook机制.md) |
| 20 | 沙箱隔离执行 | [20-沙箱隔离执行](docs/core/v1_0/20-沙箱隔离执行.md) |
| 21 | 普通对话 Agent | [21-普通对话Agent](docs/core/v1_0/21-普通对话Agent.md) |

推荐阅读顺序：05 → 10 → 12 → 18 → 17 → 19 → 20。

RAG 模块文档见 [docs/rag/README.md](docs/rag/README.md)。

## 示例索引

`langchain4j-agentx-samples` 模块下的样例可以直接验证当前行为：

| 示例类 | 主要验证点 |
|--------|------------|
| `ToolCallSessionTest` | 工具调用后 `agentx_conversation` / `agentx_session` 的基础落库行为 |
| `MemoryTest` | 当前会话记忆不跨会话；长期记忆跨会话抽取、注入 |
| `MultiTurnConversationTest` | 同一 `conversationId` 跨多轮复用会话状态 |
| `CompressionLayerTest` | 6 层压缩策略、`working_messages` 覆盖写与 `offload_context` 留痕 |
| `StructuredOutputSessionTest` | 同一会话中先普通回答、再切换结构化输出 |
| `InterruptResumeSessionTest` | `USER_INTERRUPT` / `HITL_TOOL_REQUEST` 暂停恢复与会话落库 |
| `SubAgentSessionTest` | SubAgent 流式委派时，只有父 Agent 写 `agentx_session` |
| `HookTest` | 7 种 Hook 事件的注册、参数改写、流式注入、异常捕获 |
| `SandboxTest` / `DockerSandboxTest` | Local 与 Docker 两种沙箱模式的工具执行、快照恢复、隔离级别 |
| `ChatAgentSessionTest` | 普通 ChatAgent 单轮对话、跨轮会话记忆、与 ReactAgent 同表的落库行为 |
| `ChatAgentToolTest` | 普通 ChatAgent 基础工具调用：有界工具轮次、工具消息入链与逐轮 trace |

## 规划中

- Plan & Execute 架构
- Agent Teams
- 可观测性体系
