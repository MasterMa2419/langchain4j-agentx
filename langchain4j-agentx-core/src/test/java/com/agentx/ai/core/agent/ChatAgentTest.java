package com.agentx.ai.core.agent;

import com.agentx.ai.core.chatmodels.AgentModels;
import com.agentx.ai.core.hook.AgentHook;
import com.agentx.ai.core.hook.HookEvent;
import com.agentx.ai.core.model.AgentResult;
import com.agentx.ai.core.model.AgentStreamEvent;
import com.agentx.ai.core.model.AgentTool;
import com.agentx.ai.core.model.RunnableParams;
import com.agentx.ai.core.memory.store.SessionMessageStore;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import dev.langchain4j.service.tool.ToolExecutor;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.stream.Collectors;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * {@link ChatAgent} 普通单轮对话 Agent 行为测试。
 * <p>
 * 使用内存假模型（无需真实 LLM API），验证：
 * <ul>
 *   <li>单轮收敛：一次 LLM 调用直接产出答案，没有 ReAct 多轮迭代</li>
 *   <li>同步 / 流式 / callForResult / streamForResult 四种入口返回一致</li>
 *   <li>会话记忆：同一 conversationId 跨调用加载历史（内存会话存储）</li>
 *   <li>Hook 生命周期：Before/After/Reasoning/AfterCall 事件按序触发</li>
 *   <li>基础工具调用：有界工具轮次（默认 1，可配），工具执行后回填并收敛最终答案</li>
 *   <li>{@link Agent} 统一接口多态可用</li>
 * </ul>
 */
class ChatAgentTest {

    /** 可编程假模型：同时实现同步 + 流式两个接口，按轮次脚本返回 */
    static final class ScriptedModel implements ChatModel, StreamingChatModel {

        final List<ChatRequest> requests = new CopyOnWriteArrayList<>();

        @Override
        public java.util.Set<dev.langchain4j.model.chat.Capability> supportedCapabilities() {
            return java.util.Set.of();
        }

        @Override
        public dev.langchain4j.model.chat.request.ChatRequestParameters defaultRequestParameters() {
            return ChatModel.super.defaultRequestParameters();
        }

        @Override
        public java.util.List<dev.langchain4j.model.chat.listener.ChatModelListener> listeners() {
            return ChatModel.super.listeners();
        }

        @Override
        public dev.langchain4j.model.ModelProvider provider() {
            return ChatModel.super.provider();
        }

        @Override
        public ChatResponse doChat(ChatRequest request) {            requests.add(request);
            return ChatResponse.builder()
                    .aiMessage(AiMessage.from(answer()))
                    .build();
        }

        @Override
        public void doChat(ChatRequest request, StreamingChatResponseHandler handler) {
            requests.add(request);
            String answer = answer();
            int mid = answer.length() / 2;
            handler.onPartialResponse(answer.substring(0, mid));
            handler.onPartialResponse(answer.substring(mid));
            handler.onCompleteResponse(ChatResponse.builder()
                    .aiMessage(AiMessage.from(answer))
                    .build());
        }

        /** 有历史时说明进入了第二轮，返回不同内容便于断言 */
        private String answer() {
            return hasHistory() ? "第二轮回答" : "单轮回答";
        }

        private boolean hasHistory() {
            return requests.get(requests.size() - 1).messages().stream()
                    .anyMatch(m -> m instanceof AiMessage);
        }
    }

    /**
     * 工具场景假模型：当本轮携带工具规格且已执行工具轮数未达 {@code toolCallRounds} 时返回 tool call，
     * 否则返回最终文本。由此可验证"有界工具轮次"语义。
     */
    static final class ToolScriptedModel implements ChatModel, StreamingChatModel {

        final List<ChatRequest> requests = new CopyOnWriteArrayList<>();
        private final String toolName;
        private final int toolCallRounds;

        ToolScriptedModel(String toolName, int toolCallRounds) {
            this.toolName = toolName;
            this.toolCallRounds = toolCallRounds;
        }

        @Override
        public java.util.Set<dev.langchain4j.model.chat.Capability> supportedCapabilities() {
            return java.util.Set.of();
        }

        @Override
        public dev.langchain4j.model.chat.request.ChatRequestParameters defaultRequestParameters() {
            return ChatModel.super.defaultRequestParameters();
        }

        @Override
        public java.util.List<dev.langchain4j.model.chat.listener.ChatModelListener> listeners() {
            return ChatModel.super.listeners();
        }

        @Override
        public dev.langchain4j.model.ModelProvider provider() {
            return ChatModel.super.provider();
        }

        private AiMessage reply(ChatRequest request) {
            boolean hasToolSpec = request.toolSpecifications() != null
                    && !request.toolSpecifications().isEmpty();
            long toolResults = request.messages().stream()
                    .filter(m -> m instanceof ToolExecutionResultMessage).count();
            if (hasToolSpec && toolResults < toolCallRounds) {
                return AiMessage.builder()
                        .toolExecutionRequests(List.of(ToolExecutionRequest.builder()
                                .id("tc-" + toolResults)
                                .name(toolName)
                                .arguments("{\"q\":\"第" + toolResults + "次\"}")
                                .build()))
                        .build();
            }
            return AiMessage.from("最终回答-" + toolResults);
        }

        @Override
        public ChatResponse doChat(ChatRequest request) {
            requests.add(request);
            return ChatResponse.builder().aiMessage(reply(request)).build();
        }

        @Override
        public void doChat(ChatRequest request, StreamingChatResponseHandler handler) {
            requests.add(request);
            AiMessage aiMessage = reply(request);
            if (!aiMessage.hasToolExecutionRequests()) {
                String text = aiMessage.text();
                int mid = text.length() / 2;
                handler.onPartialResponse(text.substring(0, mid));
                handler.onPartialResponse(text.substring(mid));
            }
            handler.onCompleteResponse(ChatResponse.builder().aiMessage(aiMessage).build());
        }
    }

    /** 记录调用参数的测试工具 */
    static final class RecordingTool {

        final List<String> args = new CopyOnWriteArrayList<>();

        AgentTool asTool(String name) {
            ToolSpecification spec = ToolSpecification.builder()
                    .name(name)
                    .description("测试工具")
                    .build();
            ToolExecutor executor = (request, memoryId) -> {
                args.add(request.arguments());
                return "tool-result-" + args.size();
            };
            return new AgentTool(spec, executor);
        }
    }

    /** 录制 Hook 事件类型的记录器 */
    static final class RecordingHook implements AgentHook {
        final List<String> events = new CopyOnWriteArrayList<>();

        @Override
        public HookEvent onEvent(HookEvent event) {
            events.add(event.getClass().getSimpleName());
            return event;
        }
    }

    private static ChatAgent buildAgent(ScriptedModel model, SessionMessageStore store) {
        return ChatAgent.builder()
                .models(AgentModels.of(model))
                .sessionMessageStore(store)
                .build();
    }

    @Test
    void call_singleLlmCall_returnsAnswer() {
        ScriptedModel model = new ScriptedModel();
        ChatAgent agent = buildAgent(model, null);

        String answer = agent.call("你好");

        assertThat(answer).isEqualTo("单轮回答");
        // 单轮收敛：一次调用只产生一次 LLM 请求（ReAct 引擎会因工具循环产生多次）
        assertThat(model.requests).hasSize(1);
    }

    @Test
    void callForResult_returnsCompleted() {
        ScriptedModel model = new ScriptedModel();
        ChatAgent agent = buildAgent(model, null);

        AgentResult result = agent.callForResult("你好", RunnableParams.builder().build());

        assertThat(result).isInstanceOf(AgentResult.Completed.class);
        assertThat(result.answer()).isEqualTo("单轮回答");
        assertThat(model.requests).hasSize(1);
    }

    @Test
    void stream_emitsTextTokensThenComplete() {
        ScriptedModel model = new ScriptedModel();
        ChatAgent agent = buildAgent(model, null);

        List<AgentStreamEvent> events = agent
                .streamForResult("你好", RunnableParams.empty())
                .collectList()
                .block();

        assertThat(events).isNotNull();
        StringBuilder text = new StringBuilder();
        for (AgentStreamEvent event : events) {
            if (event instanceof AgentStreamEvent.Text t) {
                text.append(t.content());
            }
        }
        assertThat(text).hasSizeGreaterThanOrEqualTo(1);
        // 最后一件事是 Complete（token 统计与 sessionId 由引擎回填）
        assertThat(events.get(events.size() - 1)).isInstanceOf(AgentStreamEvent.Complete.class);
    }

    @Test
    void stream_textFlux_joinedEqualsAnswer() {
        ScriptedModel model = new ScriptedModel();
        ChatAgent agent = buildAgent(model, null);

        String joined = agent.stream("你好", RunnableParams.empty())
                .collect(Collectors.joining()).block();

        assertThat(joined).isEqualTo("单轮回答");
    }

    @Test
    void sessionMemory_sameConversationId_loadsHistoryOnSecondCall() {
        ScriptedModel model = new ScriptedModel();
        InMemorySessionStore store = new InMemorySessionStore();
        ChatAgent agent = buildAgent(model, store);

        RunnableParams params = RunnableParams.builder().conversationId("conv_1").build();

        String first = agent.call("你好", params);
        assertThat(first).isEqualTo("单轮回答");

        String second = agent.call("继续", params);
        assertThat(second).isEqualTo("第二轮回答");
        // 第二次请求应包含第一轮的 AiMessage 历史（会话记忆生效）
        List<ChatMessage> secondRequestMessages = model.requests.get(1).messages();
        assertThat(secondRequestMessages.stream().anyMatch(m -> m instanceof UserMessage
                && ((UserMessage) m).singleText().contains("你好"))).isTrue();
        assertThat(secondRequestMessages.stream().anyMatch(m -> m instanceof AiMessage)).isTrue();
    }

    @Test
    void sessionMemory_differentConversationIds_isolated() {
        ScriptedModel model = new ScriptedModel();
        InMemorySessionStore store = new InMemorySessionStore();
        ChatAgent agent = buildAgent(model, store);

        agent.call("你好", RunnableParams.builder().conversationId("conv_a").build());
        // 新会话无历史，仍返回首轮答案
        String answer = agent.call("你好",
                RunnableParams.builder().conversationId("conv_b").build());

        assertThat(answer).isEqualTo("单轮回答");
    }

    @Test
    void hooks_fireInLifecycleOrder() {
        ScriptedModel model = new ScriptedModel();
        RecordingHook hook = new RecordingHook();
        ChatAgent agent = ChatAgent.builder()
                .models(AgentModels.of(model))
                .hooks(List.of(hook))
                .build();

        agent.streamForResult("你好", RunnableParams.empty()).blockLast();

        assertThat(hook.events).containsExactly(
                "BeforeCallEvent",
                "BeforeReasoningEvent",
                "AfterReasoningEvent",
                "AfterCallEvent");
    }

    @Test
    void agentInterface_polymorphicCallOnBothTypes() {
        ScriptedModel chatModel = new ScriptedModel();
        ChatAgent chatAgent = buildAgent(chatModel, null);
        Agent asInterface = chatAgent;

        assertThat(asInterface.call("hi")).isEqualTo("单轮回答");
        assertThat(asInterface.stream("hi", RunnableParams.empty()).blockLast()).isNotNull();

        // ReactAgent 同样实现 Agent 接口（回归保护：默认 maxRounds 下无工具单轮收敛）
        ReactAgent reactAgent = ReactAgent.builder()
                .models(AgentModels.of(new ScriptedModel()))
                .build();
        assertThat(((Agent) reactAgent).call("hi")).isEqualTo("单轮回答");
        assertThat(((Agent) reactAgent).stream("hi", RunnableParams.empty()).blockLast()).isNotNull();
    }

    @Test
    void build_withoutModels_throws() {
        assertThatThrownBy(() -> ChatAgent.builder().build())
                .isInstanceOf(Exception.class);
    }

    // ==================== 工具调用（有界轮次） ====================

    @Test
    void tools_executeOneRoundThenFinalAnswer() {
        RecordingTool tool = new RecordingTool();
        // 模型只请求 1 轮工具调用
        ToolScriptedModel model = new ToolScriptedModel("lookup", 1);
        ChatAgent agent = ChatAgent.builder()
                .models(AgentModels.of(model))
                .tools(tool.asTool("lookup"))
                .build();

        String answer = agent.call("查一下");

        assertThat(answer).isEqualTo("最终回答-1");
        // 两轮 LLM：工具调用轮 + 最终答案轮
        assertThat(model.requests).hasSize(2);
        // 首轮暴露工具规格，最终轮不再暴露（模型必然收敛）
        assertThat(model.requests.get(0).toolSpecifications()).hasSize(1);
        assertThat(model.requests.get(1).toolSpecifications()).isEmpty();
        // 工具真实执行一次，参数透传
        assertThat(tool.args).containsExactly("{\"q\":\"第0次\"}");
        // 工具结果已回填进下一轮消息
        assertThat(model.requests.get(1).messages())
                .anyMatch(m -> m instanceof ToolExecutionResultMessage);
    }

    @Test
    void defaultMaxToolRounds_capsAtOneEvenIfModelKeepsCalling() {
        RecordingTool tool = new RecordingTool();
        // 模型在每轮都请求工具调用（只要有工具规格）
        ToolScriptedModel model = new ToolScriptedModel("lookup", Integer.MAX_VALUE);
        ChatAgent agent = ChatAgent.builder()
                .models(AgentModels.of(model))
                .tools(tool.asTool("lookup"))
                .build();

        String answer = agent.call("查一下");

        assertThat(answer).isEqualTo("最终回答-1");
        // 默认 maxToolRounds=1 → 上限 2 次 LLM 调用，工具只执行一次
        assertThat(model.requests).hasSize(2);
        assertThat(tool.args).hasSize(1);
        assertThat(model.requests.get(1).toolSpecifications()).isEmpty();
    }

    @Test
    void maxToolRounds_allowsChainedToolCalls() {
        RecordingTool tool = new RecordingTool();
        ToolScriptedModel model = new ToolScriptedModel("lookup", Integer.MAX_VALUE);
        ChatAgent agent = ChatAgent.builder()
                .models(AgentModels.of(model))
                .tools(tool.asTool("lookup"))
                .maxToolRounds(3)
                .build();

        String answer = agent.call("链式查一下");

        assertThat(answer).isEqualTo("最终回答-3");
        // 3 轮工具 + 1 轮最终回答 = 4 次 LLM 调用
        assertThat(model.requests).hasSize(4);
        assertThat(tool.args).hasSize(3);
        assertThat(model.requests.get(3).toolSpecifications()).isEmpty();
    }

    @Test
    void maxToolRoundsZero_doesNotExposeTools() {
        RecordingTool tool = new RecordingTool();
        ToolScriptedModel model = new ToolScriptedModel("lookup", Integer.MAX_VALUE);
        ChatAgent agent = ChatAgent.builder()
                .models(AgentModels.of(model))
                .tools(tool.asTool("lookup"))
                .maxToolRounds(0)
                .build();

        String answer = agent.call("查一下");

        assertThat(answer).isEqualTo("最终回答-0");
        // 工具完全不下发：单轮收敛，工具未执行
        assertThat(model.requests).hasSize(1);
        assertThat(model.requests.get(0).toolSpecifications()).isEmpty();
        assertThat(tool.args).isEmpty();
    }

    @Test
    void getTools_returnsConfiguredTools() {
        ChatAgent agent = ChatAgent.builder()
                .models(AgentModels.of(new ScriptedModel()))
                .tools(new RecordingTool().asTool("lookup"), new RecordingTool().asTool("search"))
                .build();

        assertThat(agent.getTools())
                .extracting(t -> t.getToolDefinition().name())
                .containsExactly("lookup", "search");
        // 默认工具轮次为 1
        assertThat(agent.getMaxToolRounds()).isEqualTo(1);
    }

    @Test
    void tools_streamForResult_emitsToolStartAndToolEnd() {
        RecordingTool tool = new RecordingTool();
        ToolScriptedModel model = new ToolScriptedModel("lookup", 1);
        ChatAgent agent = ChatAgent.builder()
                .models(AgentModels.of(model))
                .tools(tool.asTool("lookup"))
                .build();

        List<AgentStreamEvent> events = agent
                .streamForResult("查一下", RunnableParams.empty())
                .collectList()
                .block();

        assertThat(events).isNotNull();
        assertThat(events).anyMatch(e -> e instanceof AgentStreamEvent.ToolStart t
                && "lookup".equals(t.toolName()));
        assertThat(events).anyMatch(e -> e instanceof AgentStreamEvent.ToolEnd t
                && "lookup".equals(t.toolName()));
        assertThat(events.get(events.size() - 1)).isInstanceOf(AgentStreamEvent.Complete.class);
    }

    @Test
    void tools_persistToolMessagesInSession() {
        RecordingTool tool = new RecordingTool();
        ToolScriptedModel model = new ToolScriptedModel("lookup", 1);
        InMemorySessionStore store = new InMemorySessionStore();
        ChatAgent agent = ChatAgent.builder()
                .models(AgentModels.of(model))
                .tools(tool.asTool("lookup"))
                .sessionMessageStore(store)
                .build();

        agent.call("查一下", RunnableParams.builder().conversationId("conv_tool").build());

        // 会话中完整保留 工具调用 AiMessage + 工具结果，供后续多轮续聊
        assertThat(store.allMessages()).anyMatch(m -> m instanceof AiMessage ai
                && ai.hasToolExecutionRequests());
        assertThat(store.allMessages()).anyMatch(m -> m instanceof ToolExecutionResultMessage);
    }
}
