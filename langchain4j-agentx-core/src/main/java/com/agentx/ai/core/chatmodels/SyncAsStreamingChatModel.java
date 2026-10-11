package com.agentx.ai.core.chatmodels;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.PartialThinking;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;

/**
 * 把同步 {@link ChatModel} 包装为 {@link StreamingChatModel}。
 * <p>无 token 级增量：阻塞拿到完整响应后，将 thinking / text 整段作为一次 partial 发出，
 * 再回调 {@code onCompleteResponse}。
 *
 * @author bigchui
 */
final class SyncAsStreamingChatModel implements StreamingChatModel {

    private final ChatModel delegate;

    SyncAsStreamingChatModel(ChatModel delegate) {
        this.delegate = delegate;
    }

    @Override
    public void doChat(ChatRequest request, StreamingChatResponseHandler handler) {
        ChatResponse response;
        try {
            response = delegate.chat(request);
        } catch (Throwable t) {
            handler.onError(t);
            return;
        }
        if (response != null && response.aiMessage() != null) {
            AiMessage ai = response.aiMessage();
            String thinking = ai.thinking();
            if (thinking != null && !thinking.isEmpty()) {
                handler.onPartialThinking(new PartialThinking(thinking));
            }
            String text = ai.text();
            if (text != null && !text.isEmpty()) {
                handler.onPartialResponse(text);
            }
        }
        handler.onCompleteResponse(response);
    }
}
