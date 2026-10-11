package com.agentx.ai.rag.retrieve;

import com.agentx.ai.rag.common.MetadataKeys;
import com.agentx.ai.rag.common.RagDocument;
import com.agentx.ai.rag.query.Query;
import com.agentx.ai.rag.store.RagVectorStores;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.filter.Filter;

import java.util.List;
import java.util.Objects;

/**
 * 向量检索器。
 *
 * <p>对应 1.x 的 Spring AI {@code VectorStoreDocumentRetriever}：把 query 文本
 * 向量化后在 {@link EmbeddingStore} 中做 KNN 检索，结果转回 {@link RagDocument}
 * 并携带相关性分数。
 *
 * @author bigchui
 */
public final class VectorDocumentRetriever implements DocumentRetriever {

    private static final int DEFAULT_TOP_K = 5;
    private static final double DEFAULT_MIN_SCORE = 0.0;

    private final EmbeddingStore<TextSegment> vectorStore;
    private final EmbeddingModel embeddingModel;
    private final int topK;
    private final double minScore;
    private final Filter filter;

    private VectorDocumentRetriever(Builder builder) {
        this.vectorStore = builder.vectorStore;
        this.embeddingModel = builder.embeddingModel;
        this.topK = builder.topK;
        this.minScore = builder.minScore;
        this.filter = builder.filter;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public List<RagDocument> retrieve(Query query) {
        Objects.requireNonNull(query, "query");
        var embedding = embeddingModel.embed(query.text()).content();
        EmbeddingSearchRequest request = EmbeddingSearchRequest.builder()
                .queryEmbedding(embedding)
                .maxResults(topK)
                .minScore(minScore)
                .filter(filter)
                .build();
        EmbeddingSearchResult<TextSegment> result = vectorStore.search(request);
        return result.matches().stream()
                .map(RagVectorStores::toRagDocument)
                .toList();
    }

    public static final class Builder {

        private EmbeddingStore<TextSegment> vectorStore;
        private EmbeddingModel embeddingModel;
        private int topK = DEFAULT_TOP_K;
        private double minScore = DEFAULT_MIN_SCORE;
        private Filter filter;

        public Builder vectorStore(EmbeddingStore<TextSegment> vectorStore) {
            this.vectorStore = vectorStore;
            return this;
        }

        public Builder embeddingModel(EmbeddingModel embeddingModel) {
            this.embeddingModel = embeddingModel;
            return this;
        }

        public Builder topK(int topK) {
            this.topK = topK;
            return this;
        }

        public Builder minScore(double minScore) {
            this.minScore = minScore;
            return this;
        }

        public Builder filter(Filter filter) {
            this.filter = filter;
            return this;
        }

        public VectorDocumentRetriever build() {
            Objects.requireNonNull(vectorStore, "vectorStore 为必填项");
            Objects.requireNonNull(embeddingModel, "embeddingModel 为必填项");
            return new VectorDocumentRetriever(this);
        }
    }
}
