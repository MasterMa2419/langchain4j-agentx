package com.agentx.ai.core.agent;

import com.agentx.ai.core.chatmodels.AgentModels;
import com.agentx.ai.core.model.AgentStreamEvent;
import com.agentx.ai.core.model.RunnableParams;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * SubAgent 委派测试：验证普通 Agent（{@link ChatAgent}）与 {@link ReactAgent}
 * 都能被注册为子代理并由父 Agent 委派。
 * <p>
 * 使用内存假模型，无需真实 LLM API。验证点：
 * <ul>
 *   <li>父 Agent 注册出 {@code call_{name}} 工具</li>
 *   <li>父 Agent 委派后，子 Agent 收到透传消息并单轮收敛</li>
 *   <li>SubAgent 模式：子代理会话持久化被关闭（父 Agent 是唯一的会话边界）</li>
 *   <li>流式父 Agent 会把子代理事件打上 source 标签转发</li>
 *   <li>ReactAgent 子代理委派行为不回归</li>
 * </ul>
 */
class SubAgentDelegationTest {

    /** 父模型：首轮发起工具调用，拿到工具结果后给出最终答案 */
    static final class DelegatingParentModel implements ChatModel, StreamingChatModel {

        final List<ChatRequest> requests = new CopyOnWriteArrayList<>();
        private final String toolName;

        DelegatingParentModel(String toolName) {
            this.toolName = toolName;
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

        @Override
        public ChatResponse doChat(ChatRequest request) {
            requests.add(request);
            return ChatResponse.builder().aiMessage(AiMessage.from("sync-parent")).build();
        }

        @Override
        public void doChat(ChatRequest request, StreamingChatResponseHandler handler) {
            requests.add(request);
            boolean hasToolResult = request.messages().stream()
                    .anyMatch(m -> m instanceof ToolExecutionResultMessage);
            if (!hasToolResult) {
                handler.onCompleteResponse(ChatResponse.builder()
                        .aiMessage(AiMessage.builder()
                                .toolExecutionRequests(List.of(ToolExecutionRequest.builder()
                                        .id("tc-" + toolName)
                                        .name(toolName)
                                        .arguments("{\"message\":\"把 hello 翻译成中文\"}")
                                        .build()))
                                .build())
                        .build());
            } else {
                handler.onPartialResponse("delegated-ok");
                handler.onCompleteResponse(ChatResponse.builder()
                        .aiMessage(AiMessage.from("delegated-ok"))
                        .build());
            }
        }
    }

    /** 子模型：记录收到的请求，流式返回固定答案 */
    static final class RecordingChildModel implements ChatModel, StreamingChatModel {

        final List<ChatRequest> requests = new CopyOnWriteArrayList<>();
        private final String answer;

        RecordingChildModel(String answer) {
            this.answer = answer;
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

        @Override
        public ChatResponse doChat(ChatRequest request) {
            requests.add(request);
            return ChatResponse.builder().aiMessage(AiMessage.from(answer)).build();
        }

        @Override
        public void doChat(ChatRequest request, StreamingChatResponseHandler handler) {
            requests.add(request);
            handler.onPartialResponse(answer.substring(0, Math.max(1, answer.length() / 2)));
            handler.onPartialResponse(answer.substring(Math.max(1, answer.length() / 2)));
            handler.onCompleteResponse(ChatResponse.builder()
                    .aiMessage(AiMessage.from(answer))
                    .build());
        }
    }

    @Test
    void reactParent_delegatesToChatSubAgent() {
        RecordingChildModel childModel = new RecordingChildModel("翻译结果：你好");
        InMemorySessionStore childStore = new InMemorySessionStore();

        ReactAgent parent = ReactAgent.builder()
                .models(AgentModels.of(new DelegatingParentModel("call_translator")))
                .subAgent(() -> ChatAgent.builder()
                        .name("translator")
                        .description("中英互译助手")
                        .models(AgentModels.of(childModel))
                        .sessionMessageStore(childStore)
                        .build())
                .maxRounds(5)
                .build();

        // 子代理被注册为 call_{name} 工具
        assertThat(parent.getTools())
                .extracting(t -> t.getToolDefinition().name())
                .containsExactly("call_translator");

        String answer = parent.call("把 hello 翻译成中文",
                RunnableParams.builder().conversationId("conv_parent").build());

        // 注：父 Agent call() 会汇总转发到父流的子代理文本（ReAct 既有语义），故用 contains 断言
        assertThat(answer).contains("delegated-ok");

        // 子 Agent 被真实委派：单轮请求，且收到父 Agent 透传的消息
        assertThat(childModel.requests).hasSize(1);
        assertThat(childModel.requests.get(0).messages())
                .anyMatch(m -> m instanceof UserMessage u && u.singleText().contains("把 hello 翻译成中文"));

        // SubAgent 模式：子代理关闭会话持久化，父 Agent 才是持久化边界
        assertThat(childStore.isEmpty()).isTrue();

        // 父模型经历两轮：工具调用轮 + 最终答案轮
        assertThat(((DelegatingParentModel) parent.getChatModel()).requests).hasSize(2);
    }

    @Test
    void chatAgent_directCall_writesSession_forContrast() {
        RecordingChildModel childModel = new RecordingChildModel("回答");
        InMemorySessionStore store = new InMemorySessionStore();

        ChatAgent agent = ChatAgent.builder()
                .name("translator")
                .models(AgentModels.of(childModel))
                .sessionMessageStore(store)
                .build();

        agent.call("你好", RunnableParams.builder().conversationId("conv_x").build());

        // 非 SubAgent 模式下，普通 Agent 正常写自己的会话（对照上例的 empty）
        assertThat(store.size()).isGreaterThan(0);
    }

    @Test
    void streamParent_forwardsSubAgentEventsWithSource() {
        RecordingChildModel childModel = new RecordingChildModel("翻译结果：你好");

        ReactAgent parent = ReactAgent.builder()
                .models(AgentModels.of(new DelegatingParentModel("call_translator")))
                .subAgent(() -> ChatAgent.builder()
                        .name("translator")
                        .description("中英互译助手")
                        .models(AgentModels.of(childModel))
                        .build())
                .build();

        List<AgentStreamEvent> events = parent
                .streamForResult("把 hello 翻译成中文",
                        RunnableParams.builder().conversationId("conv_stream").build())
                .collectList()
                .block();

        assertThat(events).isNotNull();
        // 子代理的输出事件带 source 标签转发到父流
        assertThat(events)
                .anyMatch(e -> e instanceof AgentStreamEvent.Text t
                        && t.source() != null
                        && "translator".equals(t.source().agentName()));
        // 父流正常收敛
        assertThat(events).anyMatch(e -> e instanceof AgentStreamEvent.Complete);
    }

    @Test
    void reactParent_delegatesToReactSubAgent_regression() {
        RecordingChildModel childModel = new RecordingChildModel("子代理回答");

        ReactAgent parent = ReactAgent.builder()
                .models(AgentModels.of(new DelegatingParentModel("call_worker")))
                .subAgent(() -> ReactAgent.builder()
                        .name("worker")
                        .description("通用子代理")
                        .models(AgentModels.of(childModel))
                        .maxRounds(3)
                        .build())
                .build();

        String answer = parent.call("干点活",
                RunnableParams.builder().conversationId("conv_react").build());

        assertThat(answer).contains("delegated-ok");
        assertThat(childModel.requests).hasSize(1);
    }

    @Test
    void subAgent_withoutName_rejectedAtBuild() {
        org.assertj.core.api.Assertions.assertThatThrownBy(() -> ReactAgent.builder()
                        .models(AgentModels.of(new RecordingChildModel("x")))
                        .subAgent(() -> ChatAgent.builder()
                                .models(AgentModels.of(new RecordingChildModel("y")))
                                .build())
                        .build())
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("name");
    }
}
