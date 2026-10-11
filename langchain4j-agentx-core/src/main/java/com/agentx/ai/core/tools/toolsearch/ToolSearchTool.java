package com.agentx.ai.core.tools.toolsearch;

import com.huaban.analysis.jieba.JiebaSegmenter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import dev.langchain4j.model.chat.ChatModel;
import com.agentx.ai.core.model.AgentTool;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.service.tool.ToolExecutor;


import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * 工具搜索元工具 — 让 LLM 按需搜索并加载 deferred 工具。
 * <p>
 * 注册为 alwaysLoad 的 AgentTool，LLM 在需要更多工具时调用 tool_search，
 * 通过关键词匹配（Jieba 分词 + 打分）或 LLM 选择发现合适的工具。
 * <p>
 * 发现的工具名称写入 {@code discoveredNames}，供 AgentLoopExecutor 每轮
 * 重建工具列表时读取。
 *
 * @author bigchui
 * 
 */
public class ToolSearchTool {

    private static final Logger log = LoggerFactory.getLogger(ToolSearchTool.class);

    private static final JiebaSegmenter SEGMENTER = new JiebaSegmenter();

    /** camelCase 拆分正则 */
    private static final Pattern CAMEL_PATTERN = Pattern.compile("([a-z])([A-Z])");

    /**
     * 创建 ToolSearchTool 的 AgentTool。
     *
     * @param config          搜索配置
     * @param deferredPool    延迟工具池（name → AgentTool）
     * @param discoveredNames 已发现工具名称集合（共享可变集合）
     * @param chatModel       ChatModel（LLM 模式用，nullable）
     * @return AgentTool
     */
    public static AgentTool create(ToolSearchConfig config,
                                      Map<String, AgentTool> deferredPool,
                                      Set<String> discoveredNames,
                                      ChatModel chatModel) {
        List<TokenizedTool> catalog = buildCatalog(deferredPool);
        return createWithCatalog(config, deferredPool, catalog, discoveredNames, chatModel);
    }

    /**
     * 创建 ToolSearchTool 的 AgentTool（使用预构建的 catalog）。
     * <p>
     * 用于 DeferredToolRegistry 在会话间共享 catalog（避免重复分词），
     * 同时为每个会话创建独立的 discoveredNames。
     *
     * @param config          搜索配置
     * @param deferredPool    延迟工具池（name → AgentTool）
     * @param catalog         预构建的分词索引
     * @param discoveredNames 已发现工具名称集合（会话级）
     * @param chatModel       ChatModel（LLM 模式用，nullable）
     * @return AgentTool
     */
    public static AgentTool createWithCatalog(ToolSearchConfig config,
                                                  Map<String, AgentTool> deferredPool,
                                                  List<TokenizedTool> catalog,
                                                  Set<String> discoveredNames,
                                                  ChatModel chatModel) {
        ToolSearchFunction fn = new ToolSearchFunction(config, deferredPool, discoveredNames, chatModel, catalog);

        ToolSpecification spec = ToolSpecification.builder()
                .name("tool_search")
                .description("搜索可用的工具。当当前工具不足以完成用户任务时，调用此工具搜索更多工具。")
                .build();
        ToolExecutor executor = (request, memoryId) -> fn.apply(extractQuery(request.arguments()));
        return new AgentTool(spec, executor);
    }

    /**
     * 从工具调用参数中提取 query。
     * 参数形如 {"query": "..."}；异常时回退为原文。
     */
    static String extractQuery(String arguments) {
        if (arguments == null || arguments.isBlank()) {
            return "";
        }
        try {
            var node = new com.fasterxml.jackson.databind.ObjectMapper().readTree(arguments);
            if (node.has("query")) {
                return node.get("query").asText("");
            }
        } catch (Exception ignored) {
            // 非 JSON，按原文处理
        }
        return arguments;
    }

    // ==================== 分词工具方法 ====================

    /**
     * 对工具名进行英文分词（按 _ 和 camelCase 拆分）。
     */
    static List<String> tokenizeName(String name) {
        if (name == null || name.isEmpty()) {
            return List.of();
        }
        // 先按 _ 拆分
        String[] parts = name.split("_");
        List<String> tokens = new ArrayList<>();
        for (String part : parts) {
            // 再按 camelCase 拆分
            String separated = CAMEL_PATTERN.matcher(part).replaceAll("$1 $2");
            for (String token : separated.split("\\s+")) {
                String lower = token.toLowerCase();
                if (!lower.isEmpty()) {
                    tokens.add(lower);
                }
            }
        }
        return tokens;
    }

    /**
     * 对文本进行 Jieba 分词（中文 + 英文混合）。
     */
    static List<String> tokenizeText(String text) {
        if (text == null || text.isEmpty()) {
            return List.of();
        }
        List<String> tokens = new ArrayList<>();
        for (String word : SEGMENTER.sentenceProcess(text)) {
            String lower = word.trim().toLowerCase();
            if (!lower.isEmpty() && lower.length() > 1) {
                tokens.add(lower);
            }
        }
        return tokens;
    }

    // ==================== Catalog 构建 ====================

    /**
     * 构建分词索引 catalog（可跨会话共享）。
     */
    static List<TokenizedTool> buildCatalog(Map<String, AgentTool> deferredPool) {
        List<TokenizedTool> catalog = new ArrayList<>();
        for (Map.Entry<String, AgentTool> entry : deferredPool.entrySet()) {
            String name = entry.getKey();
            AgentTool tool = entry.getValue();
            String description = tool.getToolDefinition().description();
            catalog.add(new TokenizedTool(
                    name,
                    description != null ? description : "",
                    tokenizeName(name),
                    tokenizeText(name + " " + (description != null ? description : ""))
            ));
        }
        return Collections.unmodifiableList(catalog);
    }

    // ==================== 关键词搜索 ====================

    static List<ScoredTool> keywordSearch(List<TokenizedTool> catalog, String query, int maxResults) {
        List<String> queryTokens = tokenizeText(query);
        List<String> queryNameTokens = tokenizeName(query.toLowerCase());
        queryTokens.addAll(queryNameTokens);

        if (queryTokens.isEmpty()) {
            return List.of();
        }

        List<ScoredTool> scored = new ArrayList<>();
        for (TokenizedTool tool : catalog) {
            int score = scoreTool(tool, query, queryTokens);
            if (score > 0) {
                scored.add(new ScoredTool(tool, score));
            }
        }

        // 按分数降序，取 Top N
        scored.sort(Comparator.comparingInt(ScoredTool::score).reversed());
        return scored.stream().limit(maxResults).toList();
    }

    private static int scoreTool(TokenizedTool tool, String rawQuery, List<String> queryTokens) {
        int score = 0;
        String lowerQuery = rawQuery.toLowerCase();

        // 精确名称匹配
        if (tool.name.equalsIgnoreCase(rawQuery) || tool.name.equalsIgnoreCase(lowerQuery)) {
            score += 100;
        }

        // 名称 token 匹配
        for (String qt : queryTokens) {
            for (String nt : tool.nameTokens) {
                if (nt.equals(qt)) {
                    score += 50;
                } else if (nt.contains(qt) || qt.contains(nt)) {
                    score += 20;
                }
            }
        }

        // 名称子串匹配
        if (tool.name.toLowerCase().contains(lowerQuery)) {
            score += 20;
        }

        // 描述 token 匹配
        for (String qt : queryTokens) {
            for (String dt : tool.descTokens) {
                if (dt.equals(qt)) {
                    score += 10;
                } else if (dt.contains(qt) || qt.contains(dt)) {
                    score += 5;
                }
            }
        }

        // 描述子串匹配
        if (tool.description.toLowerCase().contains(lowerQuery)) {
            score += 5;
        }

        return score;
    }

    // ==================== LLM 搜索 ====================

    static List<String> llmSearch(ChatModel chatModel, List<TokenizedTool> catalog,
                                  String query, int maxResults) {
        // 构建精简 catalog
        String catalogStr = catalog.stream()
                .map(t -> "- " + t.name + ": " + t.description)
                .collect(Collectors.joining("\n"));

        String prompt = """
                你是一个工具选择助手。用户需要完成一个任务，请从以下工具列表中选出最相关的工具名称。

                工具列表：
                %s

                用户需求：%s

                请返回最相关的 %d 个工具名称，每行一个，只返回名称，不要包含其他内容。
                如果没有相关工具，返回空。
                """.formatted(catalogStr, query, maxResults);

        try {
            String response = chatModel.chat(prompt);

            if (response == null || response.isBlank()) {
                return List.of();
            }

            // 解析返回的 tool name
            List<String> found = new ArrayList<>();
            for (String line : response.split("\n")) {
                String trimmed = line.trim()
                        .replaceAll("^[-*\\d.]+\\s*", "")  // 移除列表标记
                        .replaceAll("`", "")                 // 移除反引号
                        .trim();
                if (!trimmed.isEmpty()) {
                    found.add(trimmed);
                }
            }
            return found.stream().limit(maxResults).toList();
        } catch (Exception e) {
            log.warn("LLM tool search failed: {}", e.getMessage());
            return List.of();
        }
    }

    // ==================== 内部数据结构 ====================

    record TokenizedTool(String name, String description,
                         List<String> nameTokens, List<String> descTokens) {}

    record ScoredTool(TokenizedTool tool, int score) {}

    // ==================== Function 实现 ====================

    /**
     * 工具搜索的执行逻辑。
     */
    static class ToolSearchFunction {

        private final ToolSearchConfig config;
        private final Map<String, AgentTool> deferredPool;
        private final Set<String> discoveredNames;
        private final ChatModel chatModel;
        private final List<TokenizedTool> catalog;

        ToolSearchFunction(ToolSearchConfig config, Map<String, AgentTool> deferredPool,
                           Set<String> discoveredNames, ChatModel chatModel,
                           List<TokenizedTool> catalog) {
            this.config = config;
            this.deferredPool = deferredPool;
            this.discoveredNames = discoveredNames;
            this.chatModel = chatModel;
            this.catalog = catalog;
        }

        String apply(String query) {
            log.debug("tool_search called with query: {}", query);

            List<String> foundNames = new ArrayList<>();

            if (config.mode() == ToolSearchConfig.Mode.KEYWORD || config.mode() == ToolSearchConfig.Mode.HYBRID) {
                List<ScoredTool> results = keywordSearch(catalog, query, config.maxResults());
                for (ScoredTool st : results) {
                    foundNames.add(st.tool.name());
                }
                log.debug("keyword search found {} tools", foundNames.size());
            }

            if (config.mode() == ToolSearchConfig.Mode.LLM
                    || (config.mode() == ToolSearchConfig.Mode.HYBRID && foundNames.isEmpty())) {
                List<String> llmResults = llmSearch(chatModel, catalog, query, config.maxResults());
                foundNames.addAll(llmResults);
                log.debug("LLM search found {} tools", llmResults.size());
            }

            // 去重并限制数量
            foundNames = foundNames.stream().distinct().limit(config.maxResults()).toList();

            if (foundNames.isEmpty()) {
                return "未找到相关工具。请尝试使用不同的关键词，或者用其他方式描述你的需求。";
            }

            // 标记为已发现
            discoveredNames.addAll(foundNames);

            // 构建返回结果
            StringBuilder sb = new StringBuilder("找到以下工具：\n");
            for (String name : foundNames) {
                AgentTool tool = deferredPool.get(name);
                if (tool != null) {
                    String desc = tool.getToolDefinition().description();
                    sb.append("- ").append(name);
                    if (desc != null && !desc.isEmpty()) {
                        // 截取描述前100字符
                        String shortDesc = desc.length() > 100 ? desc.substring(0, 100) + "..." : desc;
                        sb.append(": ").append(shortDesc);
                    }
                    sb.append("\n");
                }
            }
            sb.append("这些工具已加载到当前会话，你可以在下一轮直接调用。");
            return sb.toString();
        }
    }
}
