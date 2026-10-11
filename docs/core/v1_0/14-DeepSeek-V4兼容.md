# DeepSeek-V4 兼容说明

## 背景

DeepSeek V4 等思考模型通过独立的 `reasoning_content` 字段返回推理过程。DeepSeek API 有一个严格要求：**上一轮返回的 `reasoning_content` 必须原样回传给下一轮 API**，否则返回 HTTP 400 错误：

```
The `reasoning_content` in the thinking mode must be passed back to the API.
```

因此「思考模型 + 工具调用」的多轮组合场景，必须保证历史思考内容随请求回传。

## 框架的做法

LangChain4j 的 OpenAI 客户端原生支持这一需求：

- 通过 `returnThinking(true)` 解析 `reasoning_content`，写入 `AiMessage.thinking()`
- 通过 `sendThinking(true)` 在多轮请求中把历史思考内容回传给服务端

框架提供 `DeepSeekV4ChatModel` 工厂，把这两项配置固化下来，底层仍走 DeepSeek 的 OpenAI 兼容接口：

| 构建方法 | 返回 | 说明 |
|---------|------|------|
| `build()` | `ChatModel` | 同步模型 |
| `buildStreaming()` | `StreamingChatModel` | 流式模型 |
| `buildWithStreaming()` | `AgentModels` | 一次性拿到同步 + 流式，可直接 `.models(...)` 注入 ReactAgent |

默认值：`baseUrl = https://api.deepseek.com/v1`，`modelName = deepseek-v4`，`returnThinking = true`，`sendThinking = true`。

## 使用方式

### 1. 构造 ChatModel

使用 `com.agentx.ai.core.chatmodels.DeepSeekV4ChatModel`：

```java
import com.agentx.ai.core.chatmodels.DeepSeekV4ChatModel;
import dev.langchain4j.model.chat.ChatModel;

ChatModel chatModel = DeepSeekV4ChatModel.builder()
        .apiKey("sk-...")          // 或 .baseUrl("...") 覆盖默认地址
        .modelName("deepseek-v4")
        .temperature(0.7)
        .build();
```

需要流式时一次性构建：

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

### 2. 构建 Agent

指定 `ThinkingMode.REASONING_CONTENT`：

```java
ReactAgent agent = ReactAgent.builder()
        .chatModel(chatModel)
        .thinkingMode(ThinkingMode.REASONING_CONTENT)
        .build();

// 同步调用
String answer = agent.call("帮我分析一下 Java 和 Go 的优劣势");

// 流式调用
agent.stream("用三句话介绍 LangChain4j")
        .doOnNext(chunk -> System.out.print(chunk))
        .blockLast();
```

## 示例代码

完整示例见 `TestConfig.createDeepSeekV4ChatModel()` 与 `ThinkingModeTest`，演示了 DeepSeek V4 + 工具调用 + 流式输出的组合：

```java
ChatModel chatModel = TestConfig.createDeepSeekV4ChatModel();

ReactAgent agent = ReactAgent.builder()
        .chatModel(chatModel)
        .tools(BashTool.create(), FileSystemTools.create(), GrepTool.create())
        .thinkingMode(ThinkingMode.REASONING_CONTENT)
        .maxRounds(50)
        .build();

TestConfig.collectStreamEvents(agent.streamForResult("帮我扫描桌面有哪些文件", RunnableParams.empty()));
```

## 注意事项

- **使用 `DeepSeekV4ChatModel`（或任何开启 `returnThinking(true)` + `sendThinking(true)` 的 OpenAI 兼容模型）**：否则多轮工具调用时会因 `reasoning_content` 未回传而报错
- **必须设置 `ThinkingMode.REASONING_CONTENT`**：否则框架不会把独立通道的思考内容分发为 `Thinking` 事件
- **依赖**：`DeepSeekV4ChatModel` 基于 `langchain4j-open-ai`（core 模块已包含），不需要引入 DeepSeek 专用依赖
