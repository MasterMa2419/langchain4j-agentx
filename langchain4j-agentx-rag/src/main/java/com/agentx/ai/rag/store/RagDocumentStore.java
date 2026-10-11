package com.agentx.ai.rag.store;

import com.agentx.ai.rag.common.MetadataKeys;
import com.agentx.ai.rag.common.RagDocument;
import com.agentx.ai.rag.exception.RagErrorCode;
import com.agentx.ai.rag.exception.RagException;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingStore;
import lombok.extern.slf4j.Slf4j;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * RAG 文档写入路由门面。
 *
 * 按 skipEmbedding 把 chunk 分流到向量库、DocumentStore，
 * 调用方只负责在 Builder 里传数据源，未配置的源自动跳过。
 *
 * @author bigchui
 */
@Slf4j
public final class RagDocumentStore {

    /**
     * 单次 embedding 调用最大文档数，默认按 dashscope text-embedding-v3 上限。
     */
    private static final int DEFAULT_EMBEDDING_BATCH_SIZE = 10;

    private static final DateTimeFormatter INDEX_TIME_FORMATTER =
            DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    private final EmbeddingStore<TextSegment> vectorStore;
    private final EmbeddingModel embeddingModel;
    private final DocumentStore documentStore;
    private final int embeddingBatchSize;

    private RagDocumentStore(Builder builder) {
        this.vectorStore = builder.vectorStore;
        this.embeddingModel = builder.embeddingModel;
        this.documentStore = builder.documentStore;
        this.embeddingBatchSize = builder.embeddingBatchSize;
    }

    /**
     * 构建索引：先按 documentId 清掉旧数据，再重新写入。
     * 普通子块进向量库，skipEmbedding=1 的父块只进 DocumentStore。
     */
    public void index(String documentId, List<RagDocument> documents) {
        index(documentId, documents, null);
    }

    /**
     * 构建索引，并给所有块统一注入调用方自定义 metadata（如 docType）。
     * 自定义 metadata 会随块透传到各数据源，检索时可用于过滤。
     */
    public void index(String documentId, List<RagDocument> documents, Map<String, Object> extraMetadata) {
        if (documentId == null || documentId.isBlank()) {
            throw new RagException(RagErrorCode.STORE_CONFIG_INVALID, "documentId 不能为空");
        }
        if (documents == null || documents.isEmpty()) {
            return;
        }

        deleteByDocumentId(documentId);

        String createdAt = LocalDateTime.now().withNano(0).format(INDEX_TIME_FORMATTER);
        List<RagDocument> embedChunks = new ArrayList<>();
        List<RagDocument> parentChunks = new ArrayList<>();
        for (RagDocument doc : documents) {
            if (doc == null) {
                continue;
            }
            RagDocument normalized = withIndexMetadata(doc, documentId, createdAt, extraMetadata);
            if (isSkipEmbedding(normalized)) {
                parentChunks.add(normalized);
            } else {
                embedChunks.add(normalized);
            }
        }

        if (!embedChunks.isEmpty()) {
            for (List<RagDocument> batch : partition(embedChunks, embeddingBatchSize)) {
                List<String> ids = batch.stream().map(RagDocument::getId).toList();
                List<TextSegment> segments = batch.stream().map(RagVectorStores::toTextSegment).toList();
                List<dev.langchain4j.data.embedding.Embedding> embeddings =
                        embeddingModel.embedAll(segments).content();
                vectorStore.addAll(ids, embeddings, segments);
            }
        }

        if (!parentChunks.isEmpty()) {
            if (documentStore != null) {
                documentStore.save(parentChunks);
            } else {
                log.warn("[RagDocumentStore] 检测到 {} 个 skipEmbedding=1 的父块，但未配置 DocumentStore，父块未存储，父子回查不可用",
                        parentChunks.size());
            }
        }
    }

    /**
     * 按 documentId 删除该文档下的所有数据。
     */
    public void deleteByDocumentId(String documentId) {
        if (documentId == null || documentId.isBlank()) {
            return;
        }
        RagVectorStores.deleteByDocumentId(vectorStore, documentId);
        if (documentStore != null) {
            documentStore.deleteByDocumentId(documentId);
        }
    }

    private static RagDocument withIndexMetadata(RagDocument doc, String documentId, String createdAt,
                                                 Map<String, Object> extraMetadata) {
        Map<String, Object> metadata = new LinkedHashMap<>(doc.getMetadata());
        metadata.put(MetadataKeys.DOCUMENT_ID, documentId);
        metadata.put(MetadataKeys.CREATED_AT, createdAt);
        if (extraMetadata != null) {
            metadata.putAll(extraMetadata);
        }
        return RagDocument.builder()
                .id(doc.getId())
                .text(doc.getText())
                .metadata(metadata)
                .score(doc.getScore())
                .build();
    }

    private static boolean isSkipEmbedding(RagDocument doc) {
        Object skip = doc.getMetadata().get(MetadataKeys.SKIP_EMBEDDING);
        return Integer.valueOf(1).equals(skip);
    }

    private static List<List<RagDocument>> partition(List<RagDocument> documents, int batchSize) {
        List<List<RagDocument>> batches = new ArrayList<>();
        for (int i = 0; i < documents.size(); i += batchSize) {
            batches.add(documents.subList(i, Math.min(i + batchSize, documents.size())));
        }
        return batches;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {

        private EmbeddingStore<TextSegment> vectorStore;
        private EmbeddingModel embeddingModel;
        private DocumentStore documentStore;
        private int embeddingBatchSize = DEFAULT_EMBEDDING_BATCH_SIZE;

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

        /**
         * 单次 embedding 调用最大文档数，需匹配模型限制。
         */
        public Builder embeddingBatchSize(int embeddingBatchSize) {
            this.embeddingBatchSize = embeddingBatchSize;
            return this;
        }

        public RagDocumentStore build() {
            if (vectorStore == null) {
                throw new RagException(RagErrorCode.STORE_CONFIG_INVALID, "vectorStore 为必填项");
            }
            if (embeddingModel == null) {
                throw new RagException(RagErrorCode.STORE_CONFIG_INVALID, "embeddingModel 为必填项");
            }
            if (embeddingBatchSize <= 0) {
                throw new RagException(RagErrorCode.STORE_CONFIG_INVALID,
                        "embeddingBatchSize 必须大于 0: " + embeddingBatchSize);
            }
            return new RagDocumentStore(this);
        }
    }
}
