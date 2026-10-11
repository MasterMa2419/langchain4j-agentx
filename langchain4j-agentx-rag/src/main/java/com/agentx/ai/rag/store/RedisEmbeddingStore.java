package com.agentx.ai.rag.store;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.EmbeddingStore;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import dev.langchain4j.store.embedding.filter.Filter;
import redis.clients.jedis.JedisPooled;
import redis.clients.jedis.Pipeline;
import redis.clients.jedis.json.Path2;
import redis.clients.jedis.params.ScanParams;
import redis.clients.jedis.resps.ScanResult;
import redis.clients.jedis.search.Document;
import redis.clients.jedis.search.FTCreateParams;
import redis.clients.jedis.search.IndexDataType;
import redis.clients.jedis.search.Query;
import redis.clients.jedis.search.SearchResult;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

import static dev.langchain4j.internal.Utils.isNullOrEmpty;
import static redis.clients.jedis.search.RediSearchUtil.ToByteArray;

/**
 * 基于 Redis Stack（RediSearch + RedisJSON）的向量存储，适配 LangChain4j 1.19.0
 * 的 {@link EmbeddingStore#search(EmbeddingSearchRequest)} 接口。
 *
 * <p>设计参考 langchain4j-redis 1.0.0-alpha1（该版本实现的是旧 {@code findRelevant}
 * 接口，与 1.19.0 二进制不兼容），采用同样的 JSON 存储 + KNN 查询 + 余弦距离方案。
 * 与 1.x Spring AI RedisVectorStore 的索引互不兼容（存储结构不同），需重建索引。
 *
 * <p>metadata 以 JSON 平铺字段存储，仅注册进 schema 的 key 会随检索结果返回；
 * 数值/布尔/集合类型 metadata 序列化为字符串存储，读取时一律返回 String。
 *
 * @author bigchui
 */
public class RedisEmbeddingStore implements EmbeddingStore<TextSegment> {

    private static final Logger log = LoggerFactory.getLogger(RedisEmbeddingStore.class);
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    public static final String DEFAULT_INDEX_NAME = "agentx-rag-vector";
    public static final String DEFAULT_PREFIX = "rag:";

    private static final String SCORE_FIELD_NAME = "vector_score";

    private final JedisPooled client;
    private final String indexName;
    private final String prefix;
    private final String vectorFieldName;
    private final String scalarFieldName;
    private final Collection<String> metadataKeys;

    public RedisEmbeddingStore(JedisPooled client, String indexName, String prefix,
                               Integer dimension, Collection<String> metadataKeys) {
        this(client, indexName, prefix, "vector", "text", dimension, metadataKeys);
    }

    public RedisEmbeddingStore(JedisPooled client, String indexName, String prefix,
                               String vectorFieldName, String scalarFieldName,
                               Integer dimension, Collection<String> metadataKeys) {
        if (client == null) {
            throw new IllegalArgumentException("client must not be null");
        }
        if (dimension == null || dimension <= 0) {
            throw new IllegalArgumentException("dimension must be positive");
        }
        this.client = client;
        this.indexName = indexName == null || indexName.isBlank() ? DEFAULT_INDEX_NAME : indexName;
        this.prefix = prefix == null || prefix.isBlank() ? DEFAULT_PREFIX : prefix;
        this.vectorFieldName = vectorFieldName == null || vectorFieldName.isBlank() ? "vector" : vectorFieldName;
        this.scalarFieldName = scalarFieldName == null || scalarFieldName.isBlank() ? "text" : scalarFieldName;
        this.metadataKeys = metadataKeys == null ? List.of() : List.copyOf(metadataKeys);

        if (!isIndexExist(this.indexName)) {
            createIndex(this.indexName, dimension);
        }
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public String add(Embedding embedding) {
        String id = java.util.UUID.randomUUID().toString();
        add(id, embedding);
        return id;
    }

    @Override
    public void add(String id, Embedding embedding) {
        addInternal(id, embedding, null);
    }

    @Override
    public String add(Embedding embedding, TextSegment textSegment) {
        String id = java.util.UUID.randomUUID().toString();
        addInternal(id, embedding, textSegment);
        return id;
    }

    @Override
    public List<String> addAll(List<Embedding> embeddings) {
        List<String> ids = embeddings.stream()
                .map(ignored -> java.util.UUID.randomUUID().toString())
                .toList();
        addAll(ids, embeddings, null);
        return ids;
    }

    @Override
    public void addAll(List<String> ids, List<Embedding> embeddings, List<TextSegment> embedded) {
        if (isNullOrEmpty(ids) || isNullOrEmpty(embeddings)) {
            log.info("do not add empty embeddings to redis");
            return;
        }
        if (ids.size() != embeddings.size()) {
            throw new IllegalArgumentException("ids size is not equal to embeddings size");
        }
        if (embedded != null && embeddings.size() != embedded.size()) {
            throw new IllegalArgumentException("embeddings size is not equal to embedded size");
        }

        List<Object> responses;
        try (Pipeline pipeline = client.pipelined()) {
            int size = ids.size();
            for (int i = 0; i < size; i++) {
                String id = ids.get(i);
                Embedding embedding = embeddings.get(i);
                TextSegment textSegment = embedded == null ? null : embedded.get(i);
                Map<String, Object> fields = new HashMap<>();
                fields.put(vectorFieldName, embedding.vector());
                if (textSegment != null) {
                    fields.put(scalarFieldName, textSegment.text());
                    for (Map.Entry<String, Object> entry : textSegment.metadata().toMap().entrySet()) {
                        fields.put(entry.getKey(), asScalar(entry.getValue()));
                    }
                }
                String key = prefix + id;
                pipeline.jsonSetWithEscape(key, Path2.of("$"), fields);
            }
            responses = pipeline.syncAndReturnAll();
        }
        Optional<Object> errResponse = responses.stream().filter(response -> !"OK".equals(response)).findAny();
        if (errResponse.isPresent()) {
            log.error("add embedding failed, msg={}", errResponse.get());
            throw new IllegalStateException("add embedding failed, msg=" + errResponse.get());
        }
    }

    @Override
    public void removeAll(Filter filter) {
        if (filter == null) {
            removeAll();
            return;
        }
        List<String> keysToDelete = new ArrayList<>();
        String cursor = "0";
        do {
            ScanResult<String> scan = client.scan(cursor,
                    new ScanParams().match(prefix + "*").count(1000));
            cursor = scan.getCursor();
            for (String key : scan.getResult()) {
                try {
                    String json = client.jsonGetAsPlainString(key, new redis.clients.jedis.json.Path("$"));
                    if (json == null) {
                        continue;
                    }
                    Map<String, Object> fields = OBJECT_MAPPER.readValue(json, Map.class);
                    if (filter.test(fields)) {
                        keysToDelete.add(key);
                    }
                } catch (Exception e) {
                    throw new IllegalStateException("failed to read key for filter deletion: " + key, e);
                }
            }
        } while (!"0".equals(cursor));
        if (!keysToDelete.isEmpty()) {
            client.del(keysToDelete.toArray(new String[0]));
        }
    }

    @Override
    public void removeAll() {
        List<String> keysToDelete = new ArrayList<>();
        String cursor = "0";
        do {
            ScanResult<String> scan = client.scan(cursor,
                    new ScanParams().match(prefix + "*").count(1000));
            cursor = scan.getCursor();
            keysToDelete.addAll(scan.getResult());
        } while (!"0".equals(cursor));
        if (!keysToDelete.isEmpty()) {
            client.del(keysToDelete.toArray(new String[0]));
        }
    }

    @Override
    public EmbeddingSearchResult<TextSegment> search(EmbeddingSearchRequest request) {
        // Using KNN query on @vector field
        String queryTemplate = "*=>[ KNN %d @%s $BLOB AS %s ]";
        List<String> returnFields = new ArrayList<>(metadataKeys);
        returnFields.addAll(List.of(vectorFieldName, scalarFieldName, SCORE_FIELD_NAME));
        Query query = new Query(String.format(queryTemplate, request.maxResults(), vectorFieldName, SCORE_FIELD_NAME))
                .addParam("BLOB", ToByteArray(request.queryEmbedding().vector()))
                .returnFields(returnFields.toArray(new String[0]))
                .setSortBy(SCORE_FIELD_NAME, true)
                .dialect(2);
        if (request.filter() != null) {
            log.warn("RedisEmbeddingStore 暂不支持 metadata 过滤，filter 已忽略");
        }

        SearchResult result = client.ftSearch(indexName, query);
        List<EmbeddingMatch<TextSegment>> matches = toEmbeddingMatch(result.getDocuments(), request.minScore());
        return new EmbeddingSearchResult<>(matches);
    }

    private List<EmbeddingMatch<TextSegment>> toEmbeddingMatch(List<Document> documents, double minScore) {
        if (documents == null || documents.isEmpty()) {
            return new ArrayList<>();
        }
        return documents.stream()
                .map(document -> {
                    // RediSearch KNN 返回的是余弦距离，转换为 1 完全相似、0 完全不相关的分数
                    double score = (2 - Double.parseDouble(document.getString(SCORE_FIELD_NAME))) / 2;
                    String id = document.getId().substring(prefix.length());
                    String text = document.hasProperty(scalarFieldName)
                            ? document.getString(scalarFieldName)
                            : null;
                    TextSegment embedded = null;
                    if (text != null) {
                        Map<String, String> metadata = metadataKeys.stream()
                                .filter(document::hasProperty)
                                .collect(java.util.stream.Collectors.toMap(
                                        metadataKey -> metadataKey, document::getString));
                        embedded = new TextSegment(text, new Metadata(metadata));
                    }
                    Embedding embedding;
                    try {
                        float[] vectors = OBJECT_MAPPER.readValue(
                                document.getString(vectorFieldName), float[].class);
                        embedding = new Embedding(vectors);
                    } catch (JsonProcessingException e) {
                        throw new IllegalStateException("failed to parse embedding", e);
                    }
                    return new EmbeddingMatch<>(score, id, embedding, embedded);
                })
                .filter(match -> match.score() >= minScore)
                .toList();
    }

    private static Object asScalar(Object value) {
        if (value == null || value instanceof String || value instanceof Number || value instanceof Boolean) {
            return value;
        }
        return value.toString();
    }

    private void createIndex(String indexName, int dimension) {
        List<redis.clients.jedis.search.schemafields.SchemaField> fields = new ArrayList<>();
        fields.add(redis.clients.jedis.search.schemafields.TextField.of(scalarFieldName));
        fields.add(redis.clients.jedis.search.schemafields.VectorField.builder()
                .fieldName(vectorFieldName)
                .algorithm(redis.clients.jedis.search.schemafields.VectorField.VectorAlgorithm.FLAT)
                .addAttribute("TYPE", "FLOAT32")
                .addAttribute("DIM", dimension)
                .addAttribute("DISTANCE_METRIC", "COSINE")
                .build());
        String res = client.ftCreate(indexName, FTCreateParams.createParams()
                .on(IndexDataType.JSON)
                .addPrefix(prefix), fields);
        if (!"OK".equals(res)) {
            log.error("create index error, msg={}", res);
            throw new IllegalStateException("create index error, msg=" + res);
        }
    }

    private boolean isIndexExist(String indexName) {
        Set<String> indexes = client.ftList();
        return indexes.contains(indexName);
    }

    private void addInternal(String id, Embedding embedding, TextSegment embedded) {
        addAll(List.of(id), List.of(embedding), embedded == null ? null : List.of(embedded));
    }

    public static final class Builder {

        private JedisPooled client;
        private String indexName = DEFAULT_INDEX_NAME;
        private String prefix = DEFAULT_PREFIX;
        private String vectorFieldName = "vector";
        private String scalarFieldName = "text";
        private Integer dimension;
        private Collection<String> metadataKeys = new ArrayList<>();

        private Builder() {
        }

        public Builder client(JedisPooled client) {
            this.client = client;
            return this;
        }

        public Builder indexName(String indexName) {
            this.indexName = indexName;
            return this;
        }

        public Builder prefix(String prefix) {
            this.prefix = prefix;
            return this;
        }

        public Builder vectorFieldName(String vectorFieldName) {
            this.vectorFieldName = vectorFieldName;
            return this;
        }

        public Builder scalarFieldName(String scalarFieldName) {
            this.scalarFieldName = scalarFieldName;
            return this;
        }

        public Builder dimension(Integer dimension) {
            this.dimension = dimension;
            return this;
        }

        public Builder metadataKeys(Collection<String> metadataKeys) {
            this.metadataKeys = metadataKeys;
            return this;
        }

        public RedisEmbeddingStore build() {
            return new RedisEmbeddingStore(client, indexName, prefix,
                    vectorFieldName, scalarFieldName, dimension, metadataKeys);
        }
    }
}
