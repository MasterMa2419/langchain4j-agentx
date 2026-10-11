package com.agentx.ai.core.context.compress;

import dev.langchain4j.data.message.ChatMessage;

import java.util.List;

/**
 * 空实现 OffloadStore：不持久化任何原文。
 *
 * @author bigchui
 */
public class NoOpOffloadStore implements OffloadStore {

    @Override
    public String offload(String conversationId, long sessionId, List<ChatMessage> messages) {
        return null;
    }

    @Override
    public List<ChatMessage> load(String conversationId, String uuid) {
        return List.of();
    }
}
