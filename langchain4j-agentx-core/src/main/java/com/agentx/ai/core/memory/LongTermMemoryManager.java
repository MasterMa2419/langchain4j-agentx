package com.agentx.ai.core.memory;

import com.agentx.ai.core.prompt.PromptConstants;
import com.agentx.ai.core.stage.ThinkTagParser;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.filter.Filter;
import dev.langchain4j.store.embedding.filter.MetadataFilterBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

/**
 * 长期记忆管理器。
 *
 * <p>封装跨会话记忆的「抽取 - 去重 - 合并 - 检索」全流程，所有数据落在
 * 单一 doc type {@value #DOC_TYPE} 的 {@link EmbeddingStore} 中。
 *
 * <h3>写入路径</h3>
 * <ol>
 *   <li>LLM 调用 #1：从本次调用 transcript 抽取候选记忆（JSON 数组）</li>
 *   <li>对每条候选：embedding 检索 top-K 相似记忆</li>
 *   <li>命中 → LLM 调用 #2 合并旧 + 新 → delete 旧 + insert 合并</li>
 *   <li>未命中 → 直接 insert</li>
 * </ol>
 *
 * <h3>读取路径</h3>
 * 按 userId + query 语义检索 top-K 相关记忆，注入 SystemMessage。
 *
 * <p>2.0 说明：Spring AI VectorStore 内嵌 EmbeddingModel；LC4j 的 {@link EmbeddingStore}
 * 不负责向量化，因此本类显式持有 {@link EmbeddingModel}，写入与检索前先完成 embedding。
 *
 * @author bigchui
 */
public class LongTermMemoryManager {

    private static final Logger log = LoggerFactory.getLogger(LongTermMemoryManager.class);
    private static final DateTimeFormatter DATE_FORMATTER = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
    private static final ObjectMapper objectMapper = new ObjectMapper();

    public static final String DOC_TYPE = "memory";
    public static final String META_TYPE = "type";
    public static final String META_USER_ID = "user_id";
    public static final String META_CONVERSATION_ID = "conversation_id";
    public static final String META_CREATED_AT = "created_at";

    private final EmbeddingStore<TextSegment> embeddingStore;
    private final EmbeddingModel embeddingModel;
    private final ChatModel chatModel;
    private final LongTermMemoryConfig config;

    public LongTermMemoryManager(LongTermMemoryConfig config, ChatModel chatModel) {
        this.config = config;
        this.chatModel = chatModel;
        this.embeddingStore = config.getEmbeddingStore();
        this.embeddingModel = config.getEmbeddingModel();
    }

    // ==================== 写入：抽取 + 去重/合并 ====================

    /**
     * 处理一次完整 ReAct 调用产生的消息链：抽取候选 → 检索去重 → 合并或新增。
     */
    public void ingestTranscript(String userId, String conversationId, List<ChatMessage> transcript) {
        if (userId == null || transcript == null || transcript.isEmpty()) {
            return;
        }
        try {
            List<String> candidates = extractCandidates(transcript);
            if (candidates.isEmpty()) {
                log.debug("No memory candidates extracted: userId={}, conversationId={}", userId, conversationId);
                return;
            }
            int upserted = 0;
            for (String candidate : candidates) {
                if (upsertMemory(userId, conversationId, candidate)) {
                    upserted++;
                }
            }
            log.info("Ingested memories: userId={}, conversationId={}, candidates={}, written={}",
                    userId, conversationId, candidates.size(), upserted);
        } catch (Exception e) {
            log.error("Memory ingestion failed: userId={}, conversationId={}, err={}",
                    userId, conversationId, e.getMessage(), e);
        }
    }

    /**
     * 单条候选记忆的去重/合并入库。
     *
     * @return true 表示已写入（新增或合并），false 表示跳过
     */
    private boolean upsertMemory(String userId, String conversationId, String candidate) {
        List<EmbeddingMatch<TextSegment>> similar = searchSimilar(userId, candidate);
        if (similar.isEmpty()) {
            insertMemory(userId, conversationId, candidate);
            log.debug("Inserted new memory: userId={}, preview={}", userId, preview(candidate));
            return true;
        }
        String merged = mergeMemories(similar, candidate);
        if (merged == null || merged.isBlank()) {
            log.debug("Merge produced empty result, skipping");
            return false;
        }
        List<String> idsToDelete = similar.stream().map(EmbeddingMatch::embeddingId).toList();
        embeddingStore.removeAll(idsToDelete);
        insertMemory(userId, conversationId, merged);
        log.debug("Merged memory: userId={}, replaced={} docs, preview={}",
                userId, idsToDelete.size(), preview(merged));
        return true;
    }

    private String extractPrompt() {
        String p = config.getExtractPrompt();
        return p != null ? p : PromptConstants.MEMORY_EXTRACT_PROMPT;
    }

    private String mergePrompt() {
        String p = config.getMergePrompt();
        return p != null ? p : PromptConstants.MEMORY_MERGE_PROMPT;
    }

    private List<String> extractCandidates(List<ChatMessage> transcript) {
        String transcriptText = formatTranscript(transcript);
        if (transcriptText.isBlank()) {
            return List.of();
        }
        try {
            String raw = chatModel.chat(ChatRequest.builder()
                    .messages(List.of(
                            SystemMessage.from(extractPrompt()),
                            UserMessage.from(transcriptText + "\n<no_think>\n"
                                    + "请严格输出 JSON 字符串数组，形如 [\"记忆1\", \"记忆2\"]，不要输出任何其他内容。")
                    ))
                    .build())
                    .aiMessage().text();
            List<String> parsed = parseStringArray(raw);
            List<String> filtered = new ArrayList<>(parsed.size());
            for (String s : parsed) {
                if (s != null && !s.isBlank()) {
                    filtered.add(s.trim());
                    if (filtered.size() >= config.getMaxCandidatesPerCall()) {
                        break;
                    }
                }
            }
            return filtered;
        } catch (Exception e) {
            log.warn("Memory extraction failed: {}", e.getMessage());
            return List.of();
        }
    }

    /**
     * 解析 LLM 返回的字符串数组（容错：剥离 markdown 代码块）。
     */
    private static List<String> parseStringArray(String raw) {
        if (raw == null || raw.isBlank()) {
            return List.of();
        }
        String json = raw.trim();
        if (json.startsWith("```")) {
            int start = json.indexOf('[');
            int end = json.lastIndexOf(']');
            if (start >= 0 && end > start) {
                json = json.substring(start, end + 1);
            }
        }
        try {
            return objectMapper.readValue(json, new TypeReference<List<String>>() {});
        } catch (Exception e) {
            log.debug("Failed to parse memory candidates JSON: {}", e.getMessage());
            return List.of();
        }
    }

    private String mergeMemories(List<EmbeddingMatch<TextSegment>> existing, String candidate) {
        StringBuilder sb = new StringBuilder();
        sb.append("## 新记忆\n").append(candidate).append("\n\n");
        sb.append("## 已有相似记忆\n");
        for (int i = 0; i < existing.size(); i++) {
            sb.append("### ").append(i + 1).append("\n")
                    .append(existing.get(i).embedded().text()).append("\n\n");
        }
        try {
            String result = chatModel.chat(ChatRequest.builder()
                    .messages(List.of(
                            SystemMessage.from(mergePrompt()),
                            UserMessage.from(sb.toString() + "\n<no_think>")
                    ))
                    .build())
                    .aiMessage().text();
            return result == null ? null : ThinkTagParser.stripThinkTags(result.trim());
        } catch (Exception e) {
            log.warn("Memory merge failed, falling back to new candidate: {}", e.getMessage());
            return candidate;
        }
    }

    private void insertMemory(String userId, String conversationId, String text) {
        Metadata metadata = new Metadata();
        metadata.put(META_TYPE, DOC_TYPE);
        metadata.put(META_USER_ID, userId);
        metadata.put(META_CONVERSATION_ID, conversationId != null ? conversationId : "");
        metadata.put(META_CREATED_AT, LocalDateTime.now().format(DATE_FORMATTER));

        TextSegment segment = TextSegment.from(text, metadata);
        Embedding embedding = embeddingModel.embed(text).content();
        embeddingStore.addAll(
                List.of(UUID.randomUUID().toString()),
                List.of(embedding),
                List.of(segment));
    }

    // ==================== 读取：检索注入 ====================

    /**
     * 按用户和当前 query 检索 top-K 相关记忆，供 SystemMessage 注入。
     */
    public List<EmbeddingMatch<TextSegment>> searchRelevant(String userId, String query) {
        if (userId == null || query == null || query.isBlank()) {
            return List.of();
        }
        try {
            return search(userId, query, config.getTopK(), config.getSimilarityThreshold());
        } catch (Exception e) {
            log.warn("Memory search failed: userId={}, err={}", userId, e.getMessage());
            return List.of();
        }
    }

    private List<EmbeddingMatch<TextSegment>> searchSimilar(String userId, String candidate) {
        return search(userId, candidate, config.getDedupTopK(), config.getDedupThreshold());
    }

    private List<EmbeddingMatch<TextSegment>> search(String userId, String query, int maxResults, double minScore) {
        Embedding queryEmbedding = embeddingModel.embed(query).content();
        EmbeddingSearchResult<TextSegment> result = embeddingStore.search(new EmbeddingSearchRequest(
                queryEmbedding, maxResults, minScore, userMemoryFilter(userId)));
        return result.matches();
    }

    private static Filter userMemoryFilter(String userId) {
        return Filter.and(
                MetadataFilterBuilder.metadataKey(META_USER_ID).isEqualTo(userId),
                MetadataFilterBuilder.metadataKey(META_TYPE).isEqualTo(DOC_TYPE)
        );
    }

    // ==================== 工具方法 ====================

    private static String formatTranscript(List<ChatMessage> transcript) {
        StringBuilder sb = new StringBuilder();
        for (ChatMessage msg : transcript) {
            String line = formatMessage(msg);
            if (line != null && !line.isBlank()) {
                sb.append(line).append("\n");
            }
        }
        return sb.toString();
    }

    private static String formatMessage(ChatMessage msg) {
        if (msg instanceof UserMessage um) {
            String text = um.singleText();
            return (text == null || text.isBlank()) ? null : "用户：" + text;
        }
        if (msg instanceof AiMessage am) {
            String text = am.text();
            var calls = am.toolExecutionRequests();
            StringBuilder sb = new StringBuilder();
            if (text != null && !text.isBlank()) {
                sb.append("助手：").append(text);
            }
            if (calls != null && !calls.isEmpty()) {
                for (var tc : calls) {
                    if (sb.length() > 0) sb.append("\n");
                    sb.append("工具调用：").append(tc.name())
                            .append("(").append(tc.arguments()).append(")");
                }
            }
            return sb.length() == 0 ? null : sb.toString();
        }
        if (msg instanceof ToolExecutionResultMessage trm) {
            String text = trm.text();
            if (text == null || text.isBlank()) {
                return null;
            }
            return "工具结果[" + trm.toolName() + "]：" + text;
        }
        return null;
    }

    private static String preview(String text) {
        if (text == null) return "";
        return text.length() > 60 ? text.substring(0, 60) + "..." : text;
    }
}
