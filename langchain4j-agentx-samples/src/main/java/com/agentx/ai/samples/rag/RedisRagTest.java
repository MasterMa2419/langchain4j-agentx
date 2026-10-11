package com.agentx.ai.samples.rag;

import com.agentx.ai.rag.common.ChunkRole;
import com.agentx.ai.rag.common.DocumentIdGenerator;
import com.agentx.ai.rag.common.MetadataKeys;
import com.agentx.ai.rag.common.RagDocument;
import com.agentx.ai.rag.parser.DocumentParser;
import com.agentx.ai.rag.parser.tika.TikaDocumentParser;
import com.agentx.ai.rag.reader.LocalDocumentReader;
import com.agentx.ai.rag.reader.RawDocument;
import com.agentx.ai.rag.splitter.HeadingSplitter;
import com.agentx.ai.rag.store.RagDocumentStore;
import com.agentx.ai.rag.store.RedisDocumentStore;
import com.agentx.ai.rag.store.RedisEmbeddingStore;
import com.agentx.ai.rag.store.RagVectorStores;
import com.agentx.ai.samples.TestConfig;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.EmbeddingStore;
import org.springframework.data.redis.connection.lettuce.LettuceConnectionFactory;
import org.springframework.data.redis.core.StringRedisTemplate;
import redis.clients.jedis.DefaultJedisClientConfig;
import redis.clients.jedis.HostAndPort;
import redis.clients.jedis.JedisPooled;

import java.io.FileDescriptor;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

/**
 * RAG 存储门面 Redis 全链路测试。
 *
 * 向量库用 RedisEmbeddingStore，DocumentStore 用 RedisDocumentStore，
 * 验证父块落 Redis、子块向量化检索的完整链路。
 *
 * @author bigchui
 */
public class RedisRagTest {

    private static final Path DEFAULT_SAMPLE_FILE = Path.of(
            "C:\\Users\\Lenovo\\Desktop\\notes\\new\\tc\\25-执行验证工具-台词.md");

    private static final String REDIS_HOST = "192.168.113.52";
    private static final int REDIS_PORT = 6399;

    public static void main(String[] args) throws Exception {
        System.setOut(new PrintStream(new FileOutputStream(FileDescriptor.out), true, StandardCharsets.UTF_8));

        Path sampleFile = args.length > 0 ? Path.of(args[0]) : DEFAULT_SAMPLE_FILE;
        TestConfig.printTestHeader("RAG Redis Test");

        RawDocument rawDocument = new LocalDocumentReader(sampleFile).read();
        String documentId = DocumentIdGenerator.getDocId(rawDocument.content());
        List<RagDocument> chunks = parseAndChunk(rawDocument);
        List<RagDocument> parents = filterByRole(chunks, ChunkRole.PARENT);

        DefaultJedisClientConfig jedisConfig = DefaultJedisClientConfig.builder()
                .connectionTimeoutMillis(10000)
                .socketTimeoutMillis(10000)
                .build();
        JedisPooled jedis = new JedisPooled(new HostAndPort(REDIS_HOST, REDIS_PORT), jedisConfig);
        EmbeddingStore<TextSegment> vectorStore = RagVectorStores.createRedisStore(jedis, 1024,
                RedisEmbeddingStore.DEFAULT_INDEX_NAME, RedisEmbeddingStore.DEFAULT_PREFIX);

        LettuceConnectionFactory connectionFactory = new LettuceConnectionFactory(REDIS_HOST, REDIS_PORT);
        connectionFactory.afterPropertiesSet();
        RedisDocumentStore documentStore = new RedisDocumentStore(new StringRedisTemplate(connectionFactory));

        EmbeddingModel embeddingModel = TestConfig.createEmbeddingModel();
        RagDocumentStore store = RagDocumentStore.builder()
                .vectorStore(vectorStore)
                .embeddingModel(embeddingModel)
                .documentStore(documentStore)
                .build();
        store.index(documentId, chunks);
        System.out.printf("documentId=%s，已写入 %d 个块（父块 %d 个）%n",
                documentId, chunks.size(), parents.size());

        for (RagDocument parent : parents) {
            RagDocument loaded = documentStore.get(parent.getId());
            requireState(loaded != null, "父块必须落库: " + parent.getId());
            requireState(Objects.equals(loaded.getText(), parent.getText()), "父块原文必须一致");
        }
        System.out.printf("父块回查校验通过：%d 个父块均落 Redis DocumentStore%n", parents.size());

        var embedding = embeddingModel.embed("验证工具").content();
        EmbeddingSearchResult<TextSegment> result = vectorStore.search(EmbeddingSearchRequest.builder()
                .queryEmbedding(embedding)
                .maxResults(5)
                .minScore(0.3)
                .build());
        requireState(!result.matches().isEmpty(), "子块必须能向量检索到");
        System.out.printf("向量检索命中 %d 个结果%n", result.matches().size());

        System.out.println("\n校验通过：Redis 全链路（向量库 + DocumentStore）正常");
    }

    private static List<RagDocument> parseAndChunk(RawDocument rawDocument) throws Exception {
        DocumentParser parser = new TikaDocumentParser();
        List<RagDocument> sourceDocuments = parser.parseToDocuments(rawDocument);
        return HeadingSplitter.builder()
                .maxHeadingLevel(2).chunkSize(300).overlap(50)
                .enableParentChild(true).build()
                .split(sourceDocuments);
    }

    private static List<RagDocument> filterByRole(List<RagDocument> chunks, ChunkRole role) {
        return chunks.stream()
                .filter(c -> role.code().equals(c.getMetadata().get(MetadataKeys.CHUNK_ROLE)))
                .toList();
    }

    private static void requireState(boolean condition, String message) {
        if (!condition) {
            throw new IllegalStateException(message);
        }
    }
}
