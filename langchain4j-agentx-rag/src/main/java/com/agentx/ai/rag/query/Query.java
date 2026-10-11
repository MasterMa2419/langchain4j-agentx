package com.agentx.ai.rag.query;

import dev.langchain4j.data.message.ChatMessage;

import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * 检索查询对象。
 *
 * <p>对应 1.x 的 Spring AI Query 类型，承载原始问题、
 * 对话历史与业务上下文，贯穿查询转换、扩展与检索全链路。
 *
 * @author bigchui
 */
public record Query(String text, List<ChatMessage> history, Map<String, Object> context) {

    public Query {
        Objects.requireNonNull(text, "text");
        history = history == null ? List.of() : List.copyOf(history);
        context = context == null ? Map.of() : Map.copyOf(context);
    }

    public static Builder builder() {
        return new Builder();
    }

    public static Query from(String text) {
        return new Query(text, List.of(), Map.of());
    }

    public static final class Builder {

        private String text;
        private List<ChatMessage> history;
        private Map<String, Object> context;

        private Builder() {
        }

        public Builder text(String text) {
            this.text = text;
            return this;
        }

        public Builder history(List<ChatMessage> history) {
            this.history = history;
            return this;
        }

        public Builder context(Map<String, Object> context) {
            this.context = context;
            return this;
        }

        public Query build() {
            Objects.requireNonNull(text, "text must not be null");
            return new Query(text, history, context);
        }
    }
}
