package com.agentx.ai.samples.rag;

import com.agentx.ai.rag.common.MetadataKeys;
import com.agentx.ai.rag.common.ChunkRole;
import com.agentx.ai.rag.common.RagDocument;
import com.agentx.ai.rag.parser.DocumentParser;
import com.agentx.ai.rag.parser.tika.TikaDocumentParser;
import com.agentx.ai.rag.reader.LocalDocumentReader;
import com.agentx.ai.rag.reader.RawDocument;
import com.agentx.ai.rag.splitter.HeadingSplitter;
import com.agentx.ai.rag.splitter.RecursiveSplitter;
import com.agentx.ai.rag.store.RagVectorStores;
import com.agentx.ai.samples.TestConfig;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.filter.MetadataFilterBuilder;

import javax.sql.DataSource;
import java.io.FileDescriptor;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.List;

/**
 * RAG 存储层测试。
 *
 * <p>修改 testNumber 切换测试场景：
 * <pre>
 * 1: PgVector 语义检索（需要 PG 环境）
 * 2: ParentChild 过滤检索 — 父块/子块 metadata filter 验证
 * </pre>
 *
 * @author bigchui
 */
public class RagStoreTest {

    static int testNumber = 1;

    private static final Path DEFAULT_SAMPLE_FILE = Path.of(
            "C:\\Users\\Lenovo\\Desktop\\notes\\new\\tc\\25-执行验证工具-台词.md");

    public static void main(String[] args) throws Exception {
        System.setOut(new PrintStream(new FileOutputStream(FileDescriptor.out), true, StandardCharsets.UTF_8));

        Path sampleFile = args.length > 0 ? Path.of(args[0]) : DEFAULT_SAMPLE_FILE;
        TestConfig.printTestHeader("RAG Store Test #" + testNumber);

        switch (testNumber) {
            case 1 -> testPgVectorStore(sampleFile);
            case 2 -> testParentChildFilter(sampleFile);
            default -> System.out.println("Unknown test number: " + testNumber);
        }
    }

    static void testPgVectorStore(Path sampleFile) throws Exception {
        List<RagDocument> chunks = parseAndChunk(sampleFile);

        DataSource pgDataSource = TestConfig.createPgDataSource();
        EmbeddingModel embeddingModel = TestConfig.createEmbeddingModel();

        EmbeddingStore<TextSegment> store = TestConfig.createPgVectorStore(pgDataSource, "agentx_rag_vector");
        indexAll(store, embeddingModel, chunks);
        System.out.printf("已向量化写入 %d 个块到 PgVector%n", chunks.size());

        String[] queries = {"如何使用验证工具", "执行验证的流程是什么", "验证工具的台词怎么写"};
        for (String query : queries) {
            List<RagDocument> results = search(store, embeddingModel, query, 3, 0.5);
            System.out.printf("%n语义检索: \"%s\" -> %d 个结果%n", query, results.size());
            for (int i = 0; i < results.size(); i++) {
                RagDocument doc = results.get(i);
                System.out.printf("  [%d] id=%s, score=%s, text=%s%n",
                        i, doc.getId(), doc.getScore(), preview(doc.getText(), 100));
            }
        }

        requireState(!search(store, embeddingModel, "验证工具", 1, 0.0).isEmpty(),
                "语义检索必须返回结果");
        System.out.println("\n校验通过：PgVector 语义检索正常");
    }

    static void testParentChildFilter(Path sampleFile) throws Exception {
        List<RagDocument> chunks = parseAndChunkWithParentChild(sampleFile);

        int parentCount = 0;
        int childCount = 0;
        for (RagDocument chunk : chunks) {
            String role = (String) chunk.getMetadata().get(MetadataKeys.CHUNK_ROLE);
            if (ChunkRole.PARENT.code().equals(role)) parentCount++;
            else if (ChunkRole.CHILD.code().equals(role)) childCount++;
        }
        System.out.printf("总块=%d, 父块=%d, 子块=%d%n", chunks.size(), parentCount, childCount);

        DataSource pgDataSource = TestConfig.createPgDataSource();
        EmbeddingModel embeddingModel = TestConfig.createEmbeddingModel();
        EmbeddingStore<TextSegment> store = TestConfig.createPgVectorStore(pgDataSource, "agentx_rag_vector");
        // 只向量化子块，父块 skipEmbedding=1 不入库
        List<RagDocument> embeddable = chunks.stream()
                .filter(c -> !Integer.valueOf(1).equals(c.getMetadata().get(MetadataKeys.SKIP_EMBEDDING)))
                .toList();
        indexAll(store, embeddingModel, embeddable);
        System.out.printf("已写入 %d 个块（含父块）%n", chunks.size());

        List<RagDocument> allResults = search(store, embeddingModel, "验证工具", 10, 0.3);
        System.out.printf("%n不带 filter 检索 -> %d 个结果%n", allResults.size());

        var filter = MetadataFilterBuilder.metadataKey(MetadataKeys.SKIP_EMBEDDING).isNotEqualTo("1");
        List<RagDocument> filteredResults = search(store, embeddingModel, "验证工具", 10, 0.3, filter);
        System.out.printf("排除父块 filter 检索 -> %d 个结果%n", filteredResults.size());

        for (int i = 0; i < filteredResults.size(); i++) {
            RagDocument doc = filteredResults.get(i);
            Object skipEmbedding = doc.getMetadata().get(MetadataKeys.SKIP_EMBEDDING);
            requireState(skipEmbedding == null || !Integer.valueOf(1).equals(skipEmbedding),
                    "过滤后的结果不应包含 skipEmbedding=1 的父块");
            System.out.printf("  [%d] role=%s, text=%s%n",
                    i, doc.getMetadata().get(MetadataKeys.CHUNK_ROLE), preview(doc.getText(), 80));
        }

        requireState(parentCount > 0, "必须有父块");
        requireState(childCount > 0, "必须有子块");
        System.out.println("\n校验通过：父块/子块 metadata filter 正常");
    }

    private static void indexAll(EmbeddingStore<TextSegment> store, EmbeddingModel embeddingModel,
                                 List<RagDocument> documents) {
        for (List<RagDocument> batch : partition(documents, 10)) {
            List<TextSegment> segments = batch.stream().map(RagVectorStores::toTextSegment).toList();
            var embeddings = embeddingModel.embedAll(segments).content();
            store.addAll(batch.stream().map(RagDocument::getId).toList(), embeddings, segments);
        }
    }

    private static List<RagDocument> search(EmbeddingStore<TextSegment> store, EmbeddingModel embeddingModel,
                                            String query, int topK, double minScore) {
        return search(store, embeddingModel, query, topK, minScore, null);
    }

    private static List<RagDocument> search(EmbeddingStore<TextSegment> store, EmbeddingModel embeddingModel,
                                            String query, int topK, double minScore,
                                            dev.langchain4j.store.embedding.filter.Filter filter) {
        EmbeddingSearchRequest.EmbeddingSearchRequestBuilder builder = EmbeddingSearchRequest.builder()
                .queryEmbedding(embeddingModel.embed(query).content())
                .maxResults(topK)
                .minScore(minScore);
        if (filter != null) {
            builder.filter(filter);
        }
        EmbeddingSearchResult<TextSegment> result = store.search(builder.build());
        return result.matches().stream().map(RagVectorStores::toRagDocument).toList();
    }

    private static List<List<RagDocument>> partition(List<RagDocument> documents, int size) {
        List<List<RagDocument>> batches = new java.util.ArrayList<>();
        for (int i = 0; i < documents.size(); i += size) {
            batches.add(documents.subList(i, Math.min(i + size, documents.size())));
        }
        return batches;
    }

    private static List<RagDocument> parseAndChunk(Path sampleFile) throws Exception {
        DocumentParser parser = new TikaDocumentParser();
        RawDocument rawDocument = new LocalDocumentReader(sampleFile).read();
        List<RagDocument> sourceDocuments = parser.parseToDocuments(rawDocument);
        return RecursiveSplitter.builder()
                .chunkSize(300).overlap(50).build()
                .split(sourceDocuments);
    }

    private static List<RagDocument> parseAndChunkWithParentChild(Path sampleFile) throws Exception {
        DocumentParser parser = new TikaDocumentParser();
        RawDocument rawDocument = new LocalDocumentReader(sampleFile).read();
        List<RagDocument> sourceDocuments = parser.parseToDocuments(rawDocument);
        return HeadingSplitter.builder()
                .maxHeadingLevel(2).chunkSize(300).overlap(50)
                .enableParentChild(true).build()
                .split(sourceDocuments);
    }

    private static String preview(String value, int maxLength) {
        String normalized = value == null ? "" : value.replaceAll("\\s+", " ").trim();
        if (normalized.codePointCount(0, normalized.length()) <= maxLength) return normalized;
        return normalized.substring(0, normalized.offsetByCodePoints(0, maxLength)) + "...";
    }

    private static void requireState(boolean condition, String message) {
        if (!condition) throw new IllegalStateException(message);
    }
}
