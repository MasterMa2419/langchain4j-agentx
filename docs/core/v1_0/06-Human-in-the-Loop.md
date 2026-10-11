# Human-in-the-Loop

框架通过 `ReactAgent.Builder.interceptTools(...)` 声明「拦截工具名单」。名单中的工具被 LLM 调用时不会直接执行，而是暂停 Agent 执行并把待处理信息返回给调用方；调用方处理后通过 `resume` / `resumeStream` 恢复执行。

## 适用场景

- **用户问答**：Agent 需要向用户提问获取信息（如偏好、需求）
- **操作审批**：敏感操作需要人工确认后执行（如写文件、发送邮件）
- **多步确认**：复杂任务中分步骤确认执行计划

## 核心流程

```
callForResult / streamForResult
    │
    ▼
Agent 执行 → 命中拦截工具 → 暂停 → 返回 Paused
    │                                    │
    │                              调用方处理
    │                              （获取用户输入）
    │                                    │
    │                                    ▼
    │                              resume / resumeStream
    │                                    │
    ▼                                    ▼
继续执行 ← ← ← ← ← ← ← ← ← ← ← ← ←
    │
    ▼
Completed / 再次 Paused（循环）
```

## 拦截工具的两类语义

`interceptTools` 名单中的工具按恢复行为分为两类：

| 工具类型 | 判定方式 | 用户输入含义 | 恢复行为 |
|---------|---------|------------|---------|
| **输入类工具** | 工具名为 `ask_user` | 用户回答即工具结果 | 直接将用户输入作为工具结果返回给模型 |
| **审批类工具** | 名单中除 `ask_user` 之外的其它工具 | 确认或拒绝 | 确认→框架实际执行工具并返回真实结果；拒绝→不执行，返回拒绝信息 |

### 确认与拒绝

- **确认关键词**：`ok`、`yes`、`y`、`好`、`好的`、`确认`、`同意`、`是`、`是的`、`approve`、`confirm`（不区分大小写）
- **拒绝**：输入非确认关键词时，工具不会执行，框架返回「`{工具名}` 工具被用户拒绝执行：`{用户输入}`」给模型，模型据此调整方案

## PauseState — 暂停状态

Agent 暂停时返回的 `PauseState` 包含完整的执行上下文（`AgentResult.Paused.state()`）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `messages` | `List<ChatMessage>` | 暂停当下的消息快照 |
| `currentRound` | `int` | 当前执行轮次 |
| `pendingToolCalls` | `List<PendingToolCall>` | 待处理的工具调用列表 |
| `params` | `RunnableParams` | 原始调用参数 |
| `query` | `String` | 本次执行的用户输入 |
| `sessionId` | `long` | 本次执行所属 session |
| `totalPromptTokens` / `totalCompletionTokens` | `long` | 已累计 token |
| `reason` | `PauseReason` | 暂停原因（`HITL_TOOL_REQUEST` / `USER_INTERRUPT`） |
| `safePoint` | `SafePoint` | 安全点（`INIT` / `LLM_STREAMING` / `TOOL_EXECUTION`） |
| `interruptMessage` | `String` | 中断/暂停附带的说明信息 |
| `interruptedAt` | `long` | 中断触发时间戳（epoch millis） |

### PendingToolCall — 待处理工具调用

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | `String` | 工具调用 ID（用于 resume 时匹配结果） |
| `name` | `String` | 工具名称 |
| `arguments` | `String` | 工具参数（JSON 格式） |

## 快速模式：askUser(true)

`askUser(true)` 是「只想让 Agent 能主动提问」时的最简配置。启用后框架自动：

- 注册内置 `AskUserTool`（LLM 可调用 `ask_user` 工具）
- 将 `ask_user` 加入拦截名单，命中即暂停等待外部回答

```java
ReactAgent agent = ReactAgent.builder()
        .chatModel(chatModel)
        .askUser(true)                    // 等价于 .tools(AskUserTool.create()).interceptTools("ask_user")
        .stateStore(stateStore)           // 可选：持久化 PauseState，支持断点重连
        .build();
```

`ask_user` 工具支持两个参数：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `question` | `String` | 是 | 要问用户的问题 |
| `options` | `List<String>` | 否 | 供用户选择的选项列表，开放性问题不需要提供 |

LLM 会根据问题类型自动判断是否需要提供选项：选择性问题传 `options`（如方案选择、偏好筛选），开放性问题不传（如预算、需求描述）。

### AskUserTool 两种模式

| 模式 | 创建方式 | 适用场景 | 行为 |
|------|---------|---------|------|
| HITL 模式 | `AskUserTool.create()` | Web 服务、需要异步交互 | 非阻塞暂停，配合 `interceptTools` |
| CLI 模式 | `AskUserTool.createBlocking()` | 命令行 / 桌面应用 | 直接阻塞等待控制台输入 |

## 手动模式：interceptTools

需要同时拦截输入类工具和审批类工具时，直接声明拦截名单：

```java
Scanner scanner = new Scanner(System.in);

ReactAgent agent = ReactAgent.builder()
        .chatModel(chatModel)
        .tools(AskUserTool.create())            // 注册提问工具
        .tools(FileSystemTools.create())        // 注册文件工具
        .interceptTools("ask_user", "write_file")  // ask_user 为输入类；write_file 为审批类
        .build();

System.out.print("请输入你的问题: ");
String query = scanner.nextLine();

AgentResult result = agent.callForResult(query, RunnableParams.empty());

while (result instanceof AgentResult.Paused p) {
    Map<String, String> answers = new LinkedHashMap<>();
    for (PendingToolCall ptc : p.state().getPendingToolCalls()) {
        System.out.println("[" + ptc.name() + "] " + ptc.arguments());

        if ("ask_user".equals(ptc.name())) {
            System.out.print("你的回答: ");      // 用户回答即工具结果
        } else {
            System.out.print("确认执行 " + ptc.name() + "（ok/拒绝）: ");  // 审批类
        }
        answers.put(ptc.id(), scanner.nextLine());
    }

    // 恢复执行：
    // - ask_user → 用户回答作为工具结果
    // - write_file + "ok" → 实际执行写文件，返回真实结果
    // - write_file + "拒绝" → 不执行，返回拒绝信息给模型
    result = agent.resume(p.state(), answers);
}

if (result instanceof AgentResult.Completed c) {
    System.out.println("A: " + c.answer());
}
```

## 自定义输入类工具

当内置 `AskUserTool` 的参数或提示词无法满足业务需求时，可以自定义输入类工具，并把它加入拦截名单：

```java
// 1. 定义自定义工具（自定义入参结构和提示词）
public class CustomAskTool {

    @Tool(name = "custom_ask", value = """
            向用户提问并提供选项列表供选择。
            在需要用户从多个方案中选择时使用此工具。
            """)
    public String customAsk(
            @P(value = "要问用户的问题") String question,
            @P(value = "供用户选择的选项列表") List<String> options) {
        return "此工具需要配合 HITL 拦截使用";
    }
}

// 2. 注册工具 + 加入拦截名单
ReactAgent agent = ReactAgent.builder()
        .chatModel(chatModel)
        .tools(FileSystemTools.create())
        .tools(AgentTool.fromAnnotated(new CustomAskTool()))   // 通过 tools(List<AgentTool>) 注册
        .interceptTools("custom_ask", "write_file")
        .maxRounds(20)
        .build();
```

> 注意：框架只把工具名恰好为 `ask_user` 的拦截工具视为「输入类」。自定义输入类工具（如 `custom_ask`）名字不为 `ask_user`，会被当作审批类工具处理（用户需用 `ok` / `confirm` 等关键词确认后，框架才会执行其方法体）。

## 流式暂停恢复

流式路径使用 `streamForResult` + `resumeStream` 实现相同的暂停恢复模式：

```java
Scanner scanner = new Scanner(System.in);
System.out.print("请输入你的问题: ");
String query = scanner.nextLine();

// 首次流式调用
PauseState pauseState = collectStreamEvents(agent.streamForResult(query, RunnableParams.empty()));

// 循环处理暂停
while (pauseState != null) {
    System.out.println("--- Agent 暂停 ---");
    Map<String, String> answers = new LinkedHashMap<>();

    for (PendingToolCall ptc : pauseState.getPendingToolCalls()) {
        System.out.println("[" + ptc.name() + "] " + ptc.arguments());
        if ("ask_user".equals(ptc.name())) {
            System.out.print("你的回答: ");
        } else {
            System.out.print("确认执行 " + ptc.name() + "（ok/拒绝）: ");
        }
        answers.put(ptc.id(), scanner.nextLine());
    }

    // 恢复执行
    pauseState = collectStreamEvents(agent.resumeStream(pauseState, answers));
}

private static PauseState collectStreamEvents(Flux<AgentStreamEvent> flux) {
    PauseState[] holder = new PauseState[1];
    flux.doOnNext(event -> {
        if (event instanceof AgentStreamEvent.Paused p) {
            holder[0] = p.state();
        } else {
            printEvent(event);   // 处理其它事件（Text、ToolStart、ToolEnd 等）
        }
    }).blockLast();
    return holder[0];
}
```

## 与用户主动中断的区别

`PauseState.reason` 区分两种暂停来源：

| 场景 | reason | safePoint | pendingToolCalls |
|------|--------|-----------|------------------|
| HITL 工具拦截 | `HITL_TOOL_REQUEST` | `TOOL_EXECUTION` | 有（待处理工具） |
| 用户主动中断 | `USER_INTERRUPT` | `INIT` / `LLM_STREAMING` / `TOOL_EXECUTION` | 视中断阶段而定 |

> 用户主动中断的恢复路径详见 [18-中断与恢复](18-中断与恢复.md)。

## 最佳实践

1. **区分工具类型**：只想让 Agent 提问用 `askUser(true)`；需要审批副作用工具时把它们加入 `interceptTools`
2. **合理设置 maxRounds**：HITL 场景通常需要更多轮次（多轮提问 + 多轮确认）
3. **处理拒绝场景**：用户拒绝后模型会自行调整方案，确保拒绝信息有意义
4. **Web 场景用 HITL 模式**：不要用 `createBlocking()`，它会阻塞线程
5. **需要跨进程恢复时配置 stateStore**：`PauseState` 可序列化持久化，配合 `stateStore` 支持断点重连

## 相关类

| 类 | 包路径 | 说明 |
|----|--------|------|
| `ReactAgent.Builder.interceptTools` | `com.agentx.ai.core.agent` | HITL 拦截工具名单入口 |
| `PauseState` | `com.agentx.ai.core.model` | 暂停状态（完整执行上下文） |
| `PendingToolCall` | `com.agentx.ai.core.model` | 待处理工具调用 |
| `PauseReason` | `com.agentx.ai.core.interrupt` | 暂停原因枚举 |
| `SafePoint` | `com.agentx.ai.core.interrupt` | 安全点枚举 |
| `AskUserTool` | `com.agentx.ai.core.tools` | 内置用户提问工具 |
