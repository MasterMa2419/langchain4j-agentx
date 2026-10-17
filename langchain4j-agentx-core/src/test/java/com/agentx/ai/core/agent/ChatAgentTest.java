package com.agentx.ai.core.agent;

import com.agentx.ai.core.chatmodels.AgentModels;
import com.agentx.ai.core.hook.AgentHook;
import com.agentx.ai.core.hook.HookEvent;
import com.agentx.ai.core.model.AgentResult;
import com.agentx.ai.core.model.AgentStreamEvent;
import com.agentx.ai.core.model.RunnableParams;
import com.agentx.ai.core.memory.store.SessionMessageStore;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
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
}
