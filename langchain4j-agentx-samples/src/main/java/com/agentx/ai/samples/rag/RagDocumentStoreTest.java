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
import com.agentx.ai.rag.store.JdbcDocumentStore;
import com.agentx.ai.rag.store.RagDocumentStore;
import com.agentx.ai.samples.TestConfig;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingStore;

import javax.sql.DataSource;
import java.io.FileDescriptor;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

/**
 * RAG 存储门面测试。
 *
 * 修改 testNumber 切换测试场景：
 * 1: 完整路由 — 父块落 DocumentStore，子块进向量库
 * 2: 未配 DocumentStore — 父块跳过（warn），子块照常写入
 * 3: 按 documentId 删除 — 父块与子块一并清除
 *
 * @author bigchui
 */
public class RagDocumentStoreTest {

    static int testNumber = 3;

    private static final Path DEFAULT_SAMPLE_FILE = Path.of(
            "C:\\Users\\Lenovo\\Desktop\\notes\\new\\tc\\25-执行验证工具-台词.md");

    private static final String RAG_VECTOR_TABLE = "agentx_rag_vector";

    public static void main(String[] args) throws Exception {
        System.setOut(new PrintStream(new FileOutputStream(FileDescriptor.out), true, StandardCharsets.UTF_8));

        Path sampleFile = args.length > 0 ? Path.of(args[0]) : DEFAULT_SAMPLE_FILE;
        TestConfig.printTestHeader("RAG Document Store Test #" + testNumber);

        switch (testNumber) {
            case 1 -> testIndexWithDocumentStore(sampleFile);
            case 2 -> testIndexWithoutDocumentStore(sampleFile);
            case 3 -> testDeleteByDocumentId(sampleFile);
            default -> System.out.println("Unknown test number: " + testNumber);
        }
        // Lettuce/HikariCP 的非守护线程会阻止 JVM 退出，测试类直接结束进程
        System.exit(0);
    }

    static void testIndexWithDocumentStore(Path sampleFile) throws Exception {
        RawDocument rawDocument = new LocalDocumentReader(sampleFile).read();
        String documentId = DocumentIdGenerator.getDocId(rawDocument.content());
        List<RagDocument> chunks = parseAndChunk(rawDocument);
        List<RagDocument> parents = filterByRole(chunks, ChunkRole.PARENT);

        DataSource pgDataSource = TestConfig.createPgDataSource();
        EmbeddingStore<TextSegment> vectorStore = TestConfig.createPgVectorStore(pgDataSource, RAG_VECTOR_TABLE);
        EmbeddingModel embeddingModel = TestConfig.createEmbeddingModel();
        JdbcDocumentStore documentStore = new JdbcDocumentStore(pgDataSource);

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
        System.out.printf("父块回查校验通过：%d 个父块均落 DocumentStore%n", parents.size());

        var result = vectorStore.search(EmbeddingSearchRequest.builder()
                .queryEmbedding(embeddingModel.embed("验证工具").content())
                .maxResults(5)
                .minScore(0.3)
                .build());
        requireState(!result.matches().isEmpty(), "子块必须写入向量库");
        System.out.println("\n校验通过：父块/子块按角色正确路由");
    }

    static void testIndexWithoutDocumentStore(Path sampleFile) throws Exception {
        RawDocument rawDocument = new LocalDocumentReader(sampleFile).read();
        String documentId = DocumentIdGenerator.getDocId(rawDocument.content());
        List<RagDocument> chunks = parseAndChunk(rawDocument);
        List<RagDocument> parents = filterByRole(chunks, ChunkRole.PARENT);

        EmbeddingStore<TextSegment> vectorStore = TestConfig.createPgVectorStore(
                TestConfig.createPgDataSource(), RAG_VECTOR_TABLE);

        RagDocumentStore store = RagDocumentStore.builder()
                .vectorStore(vectorStore)
                .embeddingModel(TestConfig.createEmbeddingModel())
                .build();
        store.index(documentId, chunks);
        System.out.printf("已写入 %d 个块（父块 %d 个，未配 DocumentStore 应被跳过）%n",
                chunks.size(), parents.size());

        var result = vectorStore.search(EmbeddingSearchRequest.builder()
                .queryEmbedding(TestConfig.createEmbeddingModel().embed("验证工具").content())
                .maxResults(5)
                .minScore(0.3)
                .build());
        requireState(!result.matches().isEmpty(), "子块必须写入向量库");
        System.out.println("\n校验通过：未配 DocumentStore 时子块正常写入，父块跳过（见上方 warn）");
    }

    static void testDeleteByDocumentId(Path sampleFile) throws Exception {
        RawDocument rawDocument = new LocalDocumentReader(sampleFile).read();
        String documentId = DocumentIdGenerator.getDocId(rawDocument.content());
        List<RagDocument> chunks = parseAndChunk(rawDocument);
        List<RagDocument> parents = filterByRole(chunks, ChunkRole.PARENT);

        DataSource pgDataSource = TestConfig.createPgDataSource();
        EmbeddingStore<TextSegment> vectorStore = TestConfig.createPgVectorStore(pgDataSource, RAG_VECTOR_TABLE);
        JdbcDocumentStore documentStore = new JdbcDocumentStore(pgDataSource);

        RagDocumentStore store = RagDocumentStore.builder()
                .vectorStore(vectorStore)
                .embeddingModel(TestConfig.createEmbeddingModel())
                .documentStore(documentStore)
                .build();
        store.index(documentId, chunks);
        store.deleteByDocumentId(documentId);
        System.out.println("已按 documentId 删除");

        for (RagDocument parent : parents) {
            requireState(documentStore.get(parent.getId()) == null,
                    "删除后父块必须不存在: " + parent.getId());
        }
        System.out.println("\n校验通过：按 documentId 删除父块");
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
