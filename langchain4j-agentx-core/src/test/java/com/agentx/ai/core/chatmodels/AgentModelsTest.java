package com.agentx.ai.core.chatmodels;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import org.junit.jupiter.api.Test;

import java.util.concurrent.atomic.AtomicReference;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * {@link AgentModels} 解析与适配器行为测试。
 */
class AgentModelsTest {

    /** 同时实现两个接口的模型 */
    static final class DualModel implements ChatModel, StreamingChatModel {

        @Override
        public ChatResponse doChat(ChatRequest request) {
            return ChatResponse.builder()
                    .aiMessage(AiMessage.builder().text("dual-sync").thinking("t").build())
                    .build();
        }

        @Override
        public void doChat(ChatRequest request, StreamingChatResponseHandler handler) {
            handler.onPartialResponse("dual-stream");
            handler.onCompleteResponse(doChat(request));
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
    }

    /** 仅同步 */
    static final class SyncOnlyModel implements ChatModel {

        @Override
        public ChatResponse doChat(ChatRequest request) {
            return ChatResponse.builder()
                    .aiMessage(AiMessage.builder().text("sync-only").thinking("think").build())
                    .build();
        }
    }

    /** 仅流式 */
    static final class StreamOnlyModel implements StreamingChatModel {

        @Override
        public void doChat(ChatRequest request, StreamingChatResponseHandler handler) {
            handler.onPartialResponse("hello ");
            handler.onPartialResponse("world");
            handler.onCompleteResponse(ChatResponse.builder()
                    .aiMessage(AiMessage.from("hello world"))
                    .build());
        }
    }

    private static ChatRequest emptyRequest() {
        return ChatRequest.builder()
                .messages(dev.langchain4j.data.message.UserMessage.from("hi"))
                .build();
    }

    @Test
    void of_chatModelImplementingBoth_reusesSameInstance() {
        DualModel dual = new DualModel();

        AgentModels models = AgentModels.of(dual);

        assertThat(models.chatModel()).isSameAs(dual);
        assertThat(models.streamingChatModel()).isSameAs(dual);
        assertThat(models.isStreamingAdapted()).isFalse();
        assertThat(models.isChatAdapted()).isFalse();
    }

    @Test
    void of_explicitPair_keepsBoth() {
        SyncOnlyModel sync = new SyncOnlyModel();
        StreamOnlyModel streaming = new StreamOnlyModel();

        AgentModels models = AgentModels.of(sync, streaming);

        assertThat(models.chatModel()).isSameAs(sync);
        assertThat(models.streamingChatModel()).isSameAs(streaming);
        assertThat(models.isStreamingAdapted()).isFalse();
        assertThat(models.isChatAdapted()).isFalse();
    }

    @Test
    void of_syncOnly_wrapsPseudoStreaming() {
        AgentModels models = AgentModels.of(new SyncOnlyModel());

        assertThat(models.isStreamingAdapted()).isTrue();

        AtomicReference<String> partial = new AtomicReference<>();
        AtomicReference<ChatResponse> complete = new AtomicReference<>();
        models.streamingChatModel().chat(emptyRequest(), new StreamingChatResponseHandler() {
            @Override
            public void onPartialResponse(String token) {
                partial.set(token);
            }

            @Override
            public void onCompleteResponse(ChatResponse response) {
                complete.set(response);
            }

            @Override
            public void onError(Throwable error) {
                throw new AssertionError(error);
            }
        });

        assertThat(partial.get()).isEqualTo("sync-only");
        assertThat(complete.get().aiMessage().text()).isEqualTo("sync-only");
        assertThat(complete.get().aiMessage().thinking()).isEqualTo("think");
    }

    @Test
    void of_streamingOnly_wrapsBlockingChat() {
        AgentModels models = AgentModels.ofStreaming(new StreamOnlyModel());

        assertThat(models.isChatAdapted()).isTrue();

        ChatResponse response = models.chatModel().chat(emptyRequest());

        assertThat(response.aiMessage().text()).isEqualTo("hello world");
    }

    @Test
    void resolve_prefersExplicitPair() {
        SyncOnlyModel sync = new SyncOnlyModel();
        StreamOnlyModel streaming = new StreamOnlyModel();

        AgentModels models = AgentModels.resolve(sync, streaming);

        assertThat(models.chatModel()).isSameAs(sync);
        assertThat(models.streamingChatModel()).isSameAs(streaming);
    }

    @Test
    void resolve_singleSide_adapts() {
        assertThat(AgentModels.resolve(new SyncOnlyModel(), null).isStreamingAdapted()).isTrue();
        assertThat(AgentModels.resolve(null, new StreamOnlyModel()).isChatAdapted()).isTrue();
    }

    @Test
    void resolve_bothNull_throws() {
        assertThatThrownBy(() -> AgentModels.resolve(null, null))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void of_nullArgs_throw() {
        assertThatThrownBy(() -> AgentModels.of((ChatModel) null))
                .isInstanceOf(NullPointerException.class);
        assertThatThrownBy(() -> AgentModels.ofStreaming(null))
                .isInstanceOf(NullPointerException.class);
        assertThatThrownBy(() -> AgentModels.of(null, new StreamOnlyModel()))
                .isInstanceOf(NullPointerException.class);
    }
}
