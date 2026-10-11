package com.agentx.ai.core.context.compress;

import com.agentx.ai.core.stage.ThinkTagParser;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.ChatMessageType;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

/**
 * LLM 摘要调用封装（L4/L5/L6 共享）。
 * 统一处理 think 标签剥离与失败回退。
 *
 * @author bigchui
 */
public class LlmSummarizer {

    private static final Logger log = LoggerFactory.getLogger(LlmSummarizer.class);

    private final ChatModel chatModel;

    public LlmSummarizer(ChatModel chatModel) {
        this.chatModel = chatModel;
    }

    /**
     * 调用 LLM 生成摘要。
     *
     * @param systemPrompt 系统提示词（定义摘要风格与约束）
     * @param userPrompt   用户提示词（含待压缩原文）
     * @return 摘要文本；失败返回 null
     */
    public String summarize(String systemPrompt, String userPrompt) {
        try {
            ChatResponse response = chatModel.chat(ChatRequest.builder()
                    .messages(List.of(
                            SystemMessage.from(systemPrompt),
                            UserMessage.from(userPrompt)
                    ))
                    .build());
            String summary = response.aiMessage().text();
            summary = ThinkTagParser.stripThinkTags(summary);
            log.debug("[LlmSummarizer] summary generated: inputChars={}, outputChars={}",
                    userPrompt != null ? userPrompt.length() : 0,
                    summary != null ? summary.length() : 0);
            return summary;
        } catch (Exception e) {
            log.warn("[LlmSummarizer] LLM call failed: {}", e.getMessage());
            return null;
        }
    }

    /**
     * 将消息列表序列化为 LLM 可读的对话文本。
     */
    public static String buildConversationText(List<ChatMessage> messages) {
        StringBuilder sb = new StringBuilder();
        for (ChatMessage msg : messages) {
            sb.append("[").append(msg.type()).append("] ");
            sb.append(extractText(msg));
            sb.append("\n\n");
        }
        return sb.toString();
    }

    private static String extractText(ChatMessage msg) {
        if (msg instanceof AiMessage am) {
            StringBuilder sb = new StringBuilder();
            if (am.text() != null) {
                sb.append(am.text());
            }
            if (am.toolExecutionRequests() != null) {
                for (var tc : am.toolExecutionRequests()) {
                    sb.append("\n[ToolCall: ").append(tc.name())
                      .append(" args=").append(tc.arguments()).append("]");
                }
            }
            return sb.toString();
        }
        if (msg instanceof ToolExecutionResultMessage trm) {
            return trm.text() != null ? trm.text() : "";
        }
        if (msg instanceof UserMessage um) {
            return um.singleText();
        }
        if (msg instanceof SystemMessage sm) {
            return sm.text();
        }
        return String.valueOf(msg);
    }
}
