package com.agentx.ai.core.utils;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.ChatMessageType;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * LangChain4j {@link ChatMessage} 与 OpenAI 兼容 JSON 之间的双向序列化器。
 *
 * <p>正向 {@link #toJson(List)} 输出 OpenAI Chat Completions 风格的消息列表 JSON，
 * 用于 trace 审计与快照存储。反向 {@link #fromJson(String)} 还原 ChatMessage 列表，
 * 用于从中断快照恢复 Agent 执行。
 *
 * <h2>正向：ChatMessage → JSON</h2>
 * <pre>
 * AiMessage(text, toolExecutionRequests=[...], thinking=...)  →  { "role":"assistant", "content":"...", "tool_calls":[...], "reasoning_content":"..." }
 * UserMessage(text)                                           →  { "role":"user", "content":"..." }
 * SystemMessage(text)                                         →  { "role":"system", "content":"..." }
 * ToolExecutionResultMessage                                  →  { "role":"tool", "tool_call_id":"...", "name":"...", "content":"..." }
 * </pre>
 *
 * <h2>反向：JSON → ChatMessage</h2>
 * <pre>
 * { "role":"assistant", "tool_calls":[...] }  →  AiMessage(text, toolExecutionRequests=[...], thinking=...)
 * { "role":"user" }                            →  UserMessage(content)
 * { "role":"system" }                          →  SystemMessage(content)
 * { "role":"tool" }                            →  ToolExecutionResultMessage（每条一项，不合并）
 * </pre>
 *
 * <p>JSON 格式与 1.x 保持一致（OpenAI 风格），存量数据库数据无需迁移。
 *
 * @author bigchui
 */
public final class MessageJsonSerializer {

    private static final Logger log = LoggerFactory.getLogger(MessageJsonSerializer.class);

    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    private MessageJsonSerializer() {
    }

    // ======================== Forward: ChatMessage → JSON ========================

    /**
     * 将消息列表序列化为 OpenAI 兼容的 JSON 字符串。
     */
    public static String toJson(List<ChatMessage> messages) {
        if (messages == null || messages.isEmpty()) {
            return "[]";
        }
        try {
            List<Map<String, Object>> messageMaps = new ArrayList<>(messages.size());
            for (ChatMessage message : messages) {
                messageMaps.add(toMessageMap(message));
            }
            return OBJECT_MAPPER.writeValueAsString(messageMaps);
        } catch (Exception e) {
            log.warn("[MessageJsonSerializer] Failed to serialize messages: {}", e.getMessage());
            return "[]";
        }
    }

    /**
     * 将消息列表转换为 List&lt;Map&gt; 中间格式。
     * 供 JdbcPauseStateStore 等需要嵌套进更大 JSON 结构的场景使用。
     */
    public static List<Map<String, Object>> toMaps(List<ChatMessage> messages) {
        if (messages == null || messages.isEmpty()) {
            return List.of();
        }
        List<Map<String, Object>> result = new ArrayList<>(messages.size());
        for (ChatMessage message : messages) {
            result.add(toMessageMap(message));
        }
        return result;
    }

    /**
     * 将 List&lt;Map&gt; 中间格式反序列化为 ChatMessage 列表。
     * 供 JdbcPauseStateStore 等从嵌套 JSON 提取 messages 的场景使用。
     */
    public static List<ChatMessage> fromMaps(List<Map<String, Object>> raw) {
        if (raw == null || raw.isEmpty()) {
            return new ArrayList<>();
        }
        return parseMessageList(raw);
    }

    /**
     * 将单条 ChatMessage 转换为 JSON message map。
     */
    public static Map<String, Object> toMessageMap(ChatMessage message) {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("role", roleOf(message));

        if (message instanceof AiMessage aiMessage) {
            map.put("content", aiMessage.text());
            List<ToolExecutionRequest> toolCalls = aiMessage.toolExecutionRequests();
            if (toolCalls != null && !toolCalls.isEmpty()) {
                map.put("tool_calls", toToolCallMaps(toolCalls));
            }
            String thinking = aiMessage.thinking();
            if (thinking != null && !thinking.isEmpty()) {
                map.put("reasoning_content", thinking);
            }
        } else if (message instanceof ToolExecutionResultMessage toolMessage) {
            map.put("tool_call_id", toolMessage.id());
            map.put("name", toolMessage.toolName());
            map.put("content", toolMessage.text());
        } else if (message instanceof UserMessage userMessage) {
            map.put("content", userMessage.singleText());
        } else if (message instanceof SystemMessage systemMessage) {
            map.put("content", systemMessage.text());
        } else {
            map.put("content", String.valueOf(message));
        }
        return map;
    }

    private static List<Map<String, Object>> toToolCallMaps(List<ToolExecutionRequest> toolCalls) {
        List<Map<String, Object>> result = new ArrayList<>(toolCalls.size());
        for (ToolExecutionRequest tc : toolCalls) {
            Map<String, Object> tcMap = new LinkedHashMap<>();
            tcMap.put("id", tc.id());
            tcMap.put("type", "function");

            Map<String, Object> fnMap = new LinkedHashMap<>();
            fnMap.put("name", tc.name());
            fnMap.put("arguments", tc.arguments());
            tcMap.put("function", fnMap);

            result.add(tcMap);
        }
        return result;
    }

    private static String roleOf(ChatMessage message) {
        ChatMessageType type = message.type();
        if (type == null) {
            return "user";
        }
        return switch (type) {
            case SYSTEM -> "system";
            case USER -> "user";
            case AI -> "assistant";
            case TOOL_EXECUTION_RESULT -> "tool";
            case CUSTOM -> "user";
        };
    }

    // ======================== Reverse: JSON → ChatMessage ========================

    /**
     * 将 OpenAI 兼容 JSON 反序列化为 LangChain4j ChatMessage 列表。
     *
     * <p>与 1.x 不同：每条 {@code {role:"tool"}} 项对应一个独立的
     * {@link ToolExecutionResultMessage}，不再合并（LC4j 协议中工具结果就是单条消息）。
     */
    public static List<ChatMessage> fromJson(String json) {
        if (json == null || json.isBlank() || "[]".equals(json.trim())) {
            return new ArrayList<>();
        }
        try {
            List<Map<String, Object>> raw = OBJECT_MAPPER.readValue(json, new TypeReference<>() {});
            return parseMessageList(raw);
        } catch (Exception e) {
            log.warn("[MessageJsonSerializer] Failed to parse messages JSON: {}", e.getMessage());
            return new ArrayList<>();
        }
    }

    private static List<ChatMessage> parseMessageList(List<Map<String, Object>> raw) {
        List<ChatMessage> result = new ArrayList<>(raw.size());
        for (Map<String, Object> item : raw) {
            String role = asString(item.get("role"));
            result.add(parseSingleMessage(item, role));
        }
        return result;
    }

    private static ChatMessage parseSingleMessage(Map<String, Object> item, String role) {
        String content = asString(item.get("content"));

        if ("assistant".equalsIgnoreCase(role)) {
            List<ToolExecutionRequest> toolCalls = parseToolCalls(item.get("tool_calls"));
            String thinking = asString(item.get("reasoning_content"));
            AiMessage.Builder builder = AiMessage.builder()
                    .text(content != null ? content : "");
            if (thinking != null && !thinking.isEmpty()) {
                builder.thinking(thinking);
            }
            if (toolCalls != null && !toolCalls.isEmpty()) {
                builder.toolExecutionRequests(toolCalls);
            }
            return builder.build();
        }
        if ("system".equalsIgnoreCase(role)) {
            return SystemMessage.from(content != null ? content : "");
        }
        if ("tool".equalsIgnoreCase(role)) {
            return ToolExecutionResultMessage.from(
                    asString(item.get("tool_call_id")),
                    asString(item.get("name")),
                    content != null ? content : "");
        }
        // 默认按 user 处理（含 role="user" 及无法识别的 role）
        return UserMessage.from(content != null ? content : "");
    }

    private static List<ToolExecutionRequest> parseToolCalls(Object value) {
        if (!(value instanceof List<?> list) || list.isEmpty()) {
            return null;
        }
        List<ToolExecutionRequest> result = new ArrayList<>(list.size());
        for (Object item : list) {
            if (!(item instanceof Map<?, ?> map)) {
                continue;
            }
            String id = asString(map.get("id"));
            String name = null;
            String arguments = null;
            Object fn = map.get("function");
            if (fn instanceof Map<?, ?> fnMap) {
                name = asString(fnMap.get("name"));
                arguments = asString(fnMap.get("arguments"));
            }
            if (arguments == null) {
                arguments = "{}";
            }
            if (name == null) {
                // 没有函数名的 tool_call 无法重建，跳过
                log.warn("[MessageJsonSerializer] Skipping tool_call without function name: id={}", id);
                continue;
            }
            result.add(ToolExecutionRequest.builder()
                    .id(Objects.requireNonNullElse(id, ""))
                    .name(name)
                    .arguments(arguments)
                    .build());
        }
        return result.isEmpty() ? null : result;
    }

    private static String asString(Object value) {
        if (value == null) {
            return null;
        }
        return value.toString();
    }
}
