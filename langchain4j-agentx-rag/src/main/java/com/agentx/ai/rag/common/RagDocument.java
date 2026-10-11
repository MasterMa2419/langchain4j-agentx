package com.agentx.ai.rag.common;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * RAG 文档载体。
 *
 * <p>对应 1.x 的 Spring AI {@code Document}。比 LangChain4j 的 {@code TextSegment}
 * 多出 id 与 score 语义：id 承载 chunkId 主键（父块回查、按文档删除依赖它），
 * score 承载检索相关性分数，metadata 保留完整 Map 结构（支持 List/Boolean 等类型），
 * 以维持 {@link MetadataKeys} 全部契约字段。
 *
 * @author bigchui
 */
public final class RagDocument {

    private final String id;
    private final String text;
    private final Map<String, Object> metadata;
    private final Double score;

    private RagDocument(Builder builder) {
        this.id = builder.id;
        this.text = builder.text;
        this.metadata = builder.metadata == null ? new LinkedHashMap<>() : new LinkedHashMap<>(builder.metadata);
        this.score = builder.score;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static RagDocument from(String text) {
        return builder().text(text).build();
    }

    public static RagDocument from(String text, Map<String, Object> metadata) {
        return builder().text(text).metadata(metadata).build();
    }

    public String getId() {
        return id;
    }

    public String getText() {
        return text;
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }

    public Double getScore() {
        return score;
    }

    @Override
    public String toString() {
        return "RagDocument{id=" + id + ", textLen=" + (text == null ? 0 : text.length()) + '}';
    }

    public static final class Builder {

        private String id;
        private String text;
        private Map<String, Object> metadata;
        private Double score;

        private Builder() {
        }

        public Builder id(String id) {
            this.id = id;
            return this;
        }

        public Builder text(String text) {
            this.text = text;
            return this;
        }

        public Builder metadata(Map<String, Object> metadata) {
            this.metadata = metadata;
            return this;
        }

        public Builder score(Double score) {
            this.score = score;
            return this;
        }

        public RagDocument build() {
            Objects.requireNonNull(text, "text must not be null");
            return new RagDocument(this);
        }
    }
}
