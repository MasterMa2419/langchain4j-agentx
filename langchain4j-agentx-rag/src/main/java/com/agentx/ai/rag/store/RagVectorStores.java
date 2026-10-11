package com.agentx.ai.rag.store;

import com.agentx.ai.rag.common.MetadataKeys;
import com.agentx.ai.rag.common.RagDocument;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.filter.Filter;
import dev.langchain4j.store.embedding.filter.comparison.IsEqualTo;
import redis.clients.jedis.JedisPooled;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 向量存储工厂与适配工具。
 *
 * <p>对应 1.x 的 Spring AI RedisVectorStore 工厂。LC4j 的 {@link EmbeddingStore}
 * 只接收 {@link TextSegment}，本类额外提供 RagDocument ↔ TextSegment 的转换，
 * 以及按 metadata 过滤删除的能力（基于内存过滤回读 ID 后逐条删除，
 * 语义对应 Spring AI 的 {@code vectorStore.delete(Filter)}）。
 *
 * <p>Redis 实现默认注册全部契约字段，仅注册的 key 会随检索结果返回 metadata；
 * 调用方自定义 metadata 如需回读，通过 extraMetadataKeys 追加注册。
 *
 * @author bigchui
 */
public final class RagVectorStores {

    private static final List<String> CONTRACT_METADATA_KEYS = List.of(
            // 父子检索链路
            MetadataKeys.DOCUMENT_ID,
            MetadataKeys.PARENT_CHUNK_ID,
            MetadataKeys.CHUNK_ROLE,
            MetadataKeys.CHUNK_GROUP_ID,
            // 溯源
            MetadataKeys.CHUNK_ID,
            MetadataKeys.FILE_NAME,
            MetadataKeys.CONTENT_TYPE,
            MetadataKeys.CREATED_AT,
            // 标题结构
            MetadataKeys.HEADING,
            MetadataKeys.HEADING_PATH,
            // 多模态标记
            MetadataKeys.IMG_URLS,
            MetadataKeys.OVERSIZED_IMG_REF,
            MetadataKeys.OVERSIZED_TABLE_REF,
            MetadataKeys.CONTAINS_IMG,
            MetadataKeys.CONTAINS_TABLE,
            // 数值字段
            MetadataKeys.CHUNK_INDEX,
            MetadataKeys.CHUNK_TOTAL,
            MetadataKeys.SKIP_EMBEDDING,
            MetadataKeys.HEADING_LEVEL,
            MetadataKeys.TABLE_COUNT
    );

    private RagVectorStores() {
    }

    /**
     * 默认构建 Redis 向量存储：契约字段全量注册，索引名 agentx-rag-vector，前缀 rag:。
     */
    public static RedisEmbeddingStore createRedisStore(JedisPooled jedis, int dimension) {
        return createRedisStore(jedis, dimension, RedisEmbeddingStore.DEFAULT_INDEX_NAME,
                RedisEmbeddingStore.DEFAULT_PREFIX);
    }

    /**
     * 自定义索引名与前缀，契约字段仍然全量注册。
     */
    public static RedisEmbeddingStore createRedisStore(JedisPooled jedis, int dimension,
                                                       String indexName, String prefix) {
        return createRedisStore(jedis, dimension, indexName, prefix, List.of());
    }

    /**
     * 完整构建：契约字段 + 调用方追加的自定义 metadata key。
     */
    public static RedisEmbeddingStore createRedisStore(JedisPooled jedis, int dimension,
                                                       String indexName, String prefix,
                                                       Collection<String> extraMetadataKeys) {
        List<String> allKeys = new ArrayList<>(CONTRACT_METADATA_KEYS);
        if (extraMetadataKeys != null) {
            for (String key : extraMetadataKeys) {
                if (key != null && !key.isBlank() && !allKeys.contains(key)) {
                    allKeys.add(key);
                }
            }
        }
        return RedisEmbeddingStore.builder()
                .client(jedis)
                .indexName(indexName)
                .prefix(prefix)
                .dimension(dimension)
                .metadataKeys(allKeys)
                .build();
    }

    /**
     * 把 RagDocument 转成 TextSegment：id 写入 chunkId metadata，score 写入独立 key。
     */
    public static TextSegment toTextSegment(RagDocument document) {
        Metadata metadata = new Metadata();
        for (Map.Entry<String, Object> entry : document.getMetadata().entrySet()) {
            metadata.put(entry.getKey(), String.valueOf(asScalar(entry.getValue())));
        }
        if (document.getId() != null) {
            metadata.put(MetadataKeys.CHUNK_ID, document.getId());
        }
        if (document.getScore() != null) {
            metadata.put(SCORE_KEY, document.getScore());
        }
        return TextSegment.from(document.getText(), metadata);
    }

    /**
     * 把 TextSegment 转回 RagDocument：优先以 chunkId metadata 恢复 id。
     */
    public static RagDocument toRagDocument(TextSegment segment) {
        Map<String, Object> metadata = new LinkedHashMap<>(segment.metadata().toMap());
        String id = metadata.remove(MetadataKeys.CHUNK_ID) == null ? null
                : segment.metadata().getString(MetadataKeys.CHUNK_ID);
        Double score = null;
        if (metadata.containsKey(SCORE_KEY)) {
            Object raw = metadata.remove(SCORE_KEY);
            score = raw instanceof Number number ? number.doubleValue() : null;
        }
        return RagDocument.builder()
                .id(id)
                .text(segment.text())
                .metadata(metadata)
                .score(score)
                .build();
    }

    /**
     * 向量检索结果转 RagDocument，附带相关性分数。
     */
    public static RagDocument toRagDocument(dev.langchain4j.store.embedding.EmbeddingMatch<TextSegment> match) {
        RagDocument document = toRagDocument(match.embedded());
        if (document.getScore() == null && match.score() != null) {
            document.getMetadata().put(SCORE_KEY, match.score());
        }
        return document;
    }

    /**
     * 按 documentId 删除向量数据。
     *
     * <p>pgvector 等原生实现支持 {@code removeAll(Filter)}；自研 Redis 实现已内置
     * SCAN + 内存过滤的 {@code removeAll(Filter)}。为兼容任意实现，本方法先走
     * 原生过滤删除，不支持时（UnsupportedFeatureException）退回逐 ID 删除。
     */
    public static void deleteByDocumentId(EmbeddingStore<TextSegment> store, String documentId) {
        if (documentId == null || documentId.isBlank()) {
            return;
        }
        Filter filter = new IsEqualTo(MetadataKeys.DOCUMENT_ID, documentId);
        try {
            store.removeAll(filter);
        } catch (dev.langchain4j.exception.UnsupportedFeatureException ignored) {
            store.remove(documentId);
        }
    }

    private static final String SCORE_KEY = "score";

    private static Object asScalar(Object value) {
        if (value == null || value instanceof String || value instanceof Number || value instanceof Boolean) {
            return value;
        }
        return value.toString();
    }
}
