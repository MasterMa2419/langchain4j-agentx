package com.agentx.ai.rag.retrieve;

import com.agentx.ai.rag.common.RagDocument;
import com.agentx.ai.rag.exception.RagErrorCode;
import com.agentx.ai.rag.exception.RagException;
import com.agentx.ai.rag.query.Query;
import com.agentx.ai.rag.store.DocumentStore;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.filter.Filter;

import java.util.List;
import java.util.Objects;

/**
 * 增强检索器。
 *
 * 统一封装向量检索与父子召回：enableParentChild 开启时命中子块回查父块，
 * 关闭时退化为纯向量检索。调用方无需在 {@link VectorDocumentRetriever} 与
 * {@link ParentChildDocumentRetriever} 之间自行选择。
 *
 * @author bigchui
 */
public final class EnhancedRetriever implements DocumentRetriever {

    private final DocumentRetriever delegate;

    private EnhancedRetriever(Builder builder) {
        VectorDocumentRetriever.Builder vectorBuilder = VectorDocumentRetriever.builder()
                .vectorStore(builder.vectorStore)
                .embeddingModel(builder.embeddingModel)
                .topK(builder.topK);
        if (builder.minScore != null) {
            vectorBuilder.minScore(builder.minScore);
        }
        if (builder.filter != null) {
            vectorBuilder.filter(builder.filter);
        }

        VectorDocumentRetriever vectorRetriever = vectorBuilder.build();
        if (builder.enableParentChild) {
            this.delegate = new ParentChildDocumentRetriever(vectorRetriever, builder.documentStore);
        } else {
            this.delegate = vectorRetriever;
        }
    }

    @Override
    public List<RagDocument> retrieve(Query query) {
        return delegate.retrieve(query);
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {

        private EmbeddingStore<TextSegment> vectorStore;
        private EmbeddingModel embeddingModel;
        private DocumentStore documentStore;
        private int topK = 5;
        private Double minScore;
        private Filter filter;
        private boolean enableParentChild;

        public Builder vectorStore(EmbeddingStore<TextSegment> vectorStore) {
            this.vectorStore = vectorStore;
            return this;
        }

        public Builder embeddingModel(EmbeddingModel embeddingModel) {
            this.embeddingModel = embeddingModel;
            return this;
        }

        public Builder documentStore(DocumentStore documentStore) {
            this.documentStore = documentStore;
            return this;
        }

        public Builder topK(int topK) {
            this.topK = topK;
            return this;
        }

        public Builder minScore(Double minScore) {
            this.minScore = minScore;
            return this;
        }

        public Builder filter(Filter filter) {
            this.filter = filter;
            return this;
        }

        /**
         * 开启父子召回：命中子块后回查父块完整小节。需同时配置 documentStore。
         */
        public Builder enableParentChild(boolean enableParentChild) {
            this.enableParentChild = enableParentChild;
            return this;
        }

        public EnhancedRetriever build() {
            Objects.requireNonNull(vectorStore, "vectorStore 为必填项");
            Objects.requireNonNull(embeddingModel, "embeddingModel 为必填项");
            if (topK <= 0) {
                throw new RagException(RagErrorCode.RETRIEVER_CONFIG_INVALID,
                        "topK 必须大于 0: " + topK);
            }
            if (enableParentChild && documentStore == null) {
                throw new RagException(RagErrorCode.RETRIEVER_CONFIG_INVALID,
                        "开启父子召回时 documentStore 为必填项");
            }
            return new EnhancedRetriever(this);
        }
    }
}
