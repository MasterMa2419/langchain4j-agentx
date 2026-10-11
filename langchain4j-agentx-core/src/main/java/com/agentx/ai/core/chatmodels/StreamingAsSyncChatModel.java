package com.agentx.ai.core.chatmodels;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

/**
 * 把 {@link StreamingChatModel} 包装为同步 {@link ChatModel}。
 * <p>阻塞收集整段流：优先采用 {@code onCompleteResponse} 的完整 {@link ChatResponse}；
 * 若服务端未回完整响应，则用 partial 聚合出 AiMessage。
 *
 * @author bigchui
 */
final class StreamingAsSyncChatModel implements ChatModel {

    private static final long DEFAULT_TIMEOUT_MINUTES = 5;

    private final StreamingChatModel delegate;

    StreamingAsSyncChatModel(StreamingChatModel delegate) {
        this.delegate = delegate;
    }

    @Override
    public ChatResponse doChat(ChatRequest request) {
        CompletableFuture<ChatResponse> future = new CompletableFuture<>();
        StringBuilder text = new StringBuilder();
        StringBuilder thinking = new StringBuilder();

        try {
            delegate.chat(request, new StreamingChatResponseHandler() {

                @Override
                public void onPartialResponse(String token) {
                    if (token != null) {
                        text.append(token);
                    }
                }

                @Override
                public void onPartialThinking(dev.langchain4j.model.chat.response.PartialThinking partialThinking) {
                    if (partialThinking != null && partialThinking.text() != null) {
                        thinking.append(partialThinking.text());
                    }
                }

                @Override
                public void onCompleteResponse(ChatResponse complete) {
                    if (complete != null && complete.aiMessage() != null) {
                        future.complete(complete);
                    } else {
                        future.complete(fallbackResponse(text, thinking));
                    }
                }

                @Override
                public void onError(Throwable error) {
                    future.completeExceptionally(error);
                }
            });
        } catch (Throwable t) {
            future.completeExceptionally(t);
        }

        try {
            return future.get(DEFAULT_TIMEOUT_MINUTES, TimeUnit.MINUTES);
        } catch (TimeoutException e) {
            future.cancel(true);
            throw new IllegalStateException("StreamingChatModel 同步适配器等待超时（"
                    + DEFAULT_TIMEOUT_MINUTES + " 分钟）", e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("StreamingChatModel 同步适配器等待被中断", e);
        } catch (ExecutionException | CompletionException e) {
            Throwable cause = e.getCause() != null ? e.getCause() : e;
            if (cause instanceof RuntimeException runtime) {
                throw runtime;
            }
            throw new IllegalStateException("StreamingChatModel 同步适配器调用失败: " + cause.getMessage(), cause);
        }
    }

    private static ChatResponse fallbackResponse(StringBuilder text, StringBuilder thinking) {
        AiMessage.Builder builder = AiMessage.builder().text(text.toString());
        if (thinking.length() > 0) {
            builder.thinking(thinking.toString());
        }
        return ChatResponse.builder().aiMessage(builder.build()).build();
    }
}
