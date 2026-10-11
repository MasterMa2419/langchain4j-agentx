package com.agentx.ai.core.memory.util;

import com.agentx.ai.core.memory.store.SessionMessageStore;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.ChatMessageType;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * 会话历史选择器。
 *
 * 从 agentx_session 的 original_messages 中提取最近几轮 QA，
 * 过滤掉工具调用等中间消息，只保留真实对话。
 *
 * @author bigchui
 */
public class ChatHistorySelector {

    private static final String STATE_KEY_ORIGINAL = "original_messages";

    private final SessionMessageStore sessionMessageStore;

    public ChatHistorySelector(SessionMessageStore sessionMessageStore) {
        this.sessionMessageStore = Objects.requireNonNull(sessionMessageStore, "sessionMessageStore");
    }

    /**
     * 提取指定会话最近 limit 条 QA 消息（仅用户提问与助手回答，过滤工具调用）。
     */
    public List<ChatMessage> recentQa(String conversationId, int limit) {
        if (conversationId == null || conversationId.isBlank() || limit <= 0) {
            return List.of();
        }
        List<ChatMessage> original = sessionMessageStore.getMessages(conversationId, STATE_KEY_ORIGINAL);

        List<ChatMessage> qa = new ArrayList<>();
        for (ChatMessage message : original) {
            ChatMessageType type = message.type();
            if (type == ChatMessageType.USER || type == ChatMessageType.AI) {
                qa.add(message);
            }
        }
        if (qa.size() <= limit) {
            return qa;
        }
        return qa.subList(qa.size() - limit, qa.size());
    }
}
