package com.agentx.ai.core.memory;

import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingStore;

import java.util.Objects;

/**
 * 长期记忆配置。
 *
 * <p>封装由调用方预先构造好的 {@link EmbeddingStore}、{@link EmbeddingModel} 与检索/合并参数，
 * 传入 {@link com.agentx.ai.core.agent.ReactAgent.Builder#longTermMemory(LongTermMemoryConfig)}。
 * 传入即启用长期记忆；不调用则不启用。
 *
 * <p>EmbeddingStore 的实现（PgVector / Redis / InMemory 等）、表名、维度、距离类型都由调用方决定。
 * LC4j 的 EmbeddingStore 不负责向量化，因此还需传入 {@link EmbeddingModel}。
 * 框架只负责用它们做语义检索、抽取与合并。
 *
 * <p>{@code extractPrompt} / {@code mergePrompt} 允许调用方自定义长期记忆的抽取与合并提示词，
 * 不传则使用框架默认值。
 *
 * @author bigchui
 */
public class LongTermMemoryConfig {

    public static final int DEFAULT_TOP_K = 5;
    public static final double DEFAULT_SIMILARITY_THRESHOLD = 0.5;
    public static final int DEFAULT_DEDUP_TOP_K = 3;
    public static final double DEFAULT_DEDUP_THRESHOLD = 0.5;
    public static final int DEFAULT_MAX_CANDIDATES_PER_CALL = 3;

    private final EmbeddingStore<TextSegment> embeddingStore;
    private final EmbeddingModel embeddingModel;
    private final int topK;
    private final double similarityThreshold;
    private final int dedupTopK;
    private final double dedupThreshold;
    private final int maxCandidatesPerCall;
    private final String extractPrompt;
    private final String mergePrompt;

    private LongTermMemoryConfig(Builder b) {
        this.embeddingStore = b.embeddingStore;
        this.embeddingModel = b.embeddingModel;
        this.topK = b.topK;
        this.similarityThreshold = b.similarityThreshold;
        this.dedupTopK = b.dedupTopK;
        this.dedupThreshold = b.dedupThreshold;
        this.maxCandidatesPerCall = b.maxCandidatesPerCall;
        this.extractPrompt = b.extractPrompt;
        this.mergePrompt = b.mergePrompt;
    }

    public static Builder builder() {
        return new Builder();
    }

    public EmbeddingStore<TextSegment> getEmbeddingStore() {
        return embeddingStore;
    }

    public EmbeddingModel getEmbeddingModel() {
        return embeddingModel;
    }

    public int getTopK() {
        return topK;
    }

    public double getSimilarityThreshold() {
        return similarityThreshold;
    }

    public int getDedupTopK() {
        return dedupTopK;
    }

    public double getDedupThreshold() {
        return dedupThreshold;
    }

    public int getMaxCandidatesPerCall() {
        return maxCandidatesPerCall;
    }

    public String getExtractPrompt() {
        return extractPrompt;
    }

    public String getMergePrompt() {
        return mergePrompt;
    }

    public static class Builder {
        private EmbeddingStore<TextSegment> embeddingStore;
        private EmbeddingModel embeddingModel;
        private int topK = DEFAULT_TOP_K;
        private double similarityThreshold = DEFAULT_SIMILARITY_THRESHOLD;
        private int dedupTopK = DEFAULT_DEDUP_TOP_K;
        private double dedupThreshold = DEFAULT_DEDUP_THRESHOLD;
        private int maxCandidatesPerCall = DEFAULT_MAX_CANDIDATES_PER_CALL;
        private String extractPrompt;
        private String mergePrompt;

        /** 设置向量存储（必填） */
        public Builder embeddingStore(EmbeddingStore<TextSegment> store) {
            this.embeddingStore = store;
            return this;
        }

        /** 设置向量模型（必填） */
        public Builder embeddingModel(EmbeddingModel model) {
            this.embeddingModel = model;
            return this;
        }

        /** 兼容 1.x 命名：等价于 embeddingStore(store) */
        public Builder vectorStore(EmbeddingStore<TextSegment> store) {
            return embeddingStore(store);
        }

        public Builder topK(int k) {
            this.topK = k;
            return this;
        }

        public Builder similarityThreshold(double t) {
            this.similarityThreshold = t;
            return this;
        }

        public Builder dedupTopK(int k) {
            this.dedupTopK = k;
            return this;
        }

        public Builder dedupThreshold(double t) {
            this.dedupThreshold = t;
            return this;
        }

        public Builder maxCandidatesPerCall(int n) {
            this.maxCandidatesPerCall = n;
            return this;
        }

        public Builder extractPrompt(String prompt) {
            this.extractPrompt = prompt;
            return this;
        }

        public Builder mergePrompt(String prompt) {
            this.mergePrompt = prompt;
            return this;
        }

        public LongTermMemoryConfig build() {
            Objects.requireNonNull(embeddingStore, "embeddingStore must not be null");
            Objects.requireNonNull(embeddingModel, "embeddingModel must not be null");
            return new LongTermMemoryConfig(this);
        }
    }
}
