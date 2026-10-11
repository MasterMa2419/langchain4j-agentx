package com.agentx.ai.rag.pipeline;

import com.agentx.ai.rag.common.RagDocument;
import com.agentx.ai.core.model.AgentTool;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import dev.langchain4j.service.tool.ToolExecutor;

import java.util.List;

/**
 * RAG 检索工具。
 *
 * 把 RagPipeline 封装成 {@link AgentTool}，注册到 ReactAgent 等智能体后，
 * 由 LLM 自主决定何时检索、用什么关键词，实现 Agentic RAG。
 *
 * @author bigchui
 */
public final class RagRetrievalTool {

    private static final String TOOL_NAME = "ragSearch";

    private static final String TOOL_DESCRIPTION = """
            检索本地知识库，返回与搜索语句相关的文档片段。
            注意：
            1. query 必须是独立完整的搜索语句，自行消解代词，不要传"它"、"这个"之类的指代
            2. 问题涉及知识库文档内容时才调用本工具
            3. 首次结果不相关时，可改写关键词后再次调用
            """;

    private RagRetrievalTool() {
    }

    public static AgentTool of(RagPipeline pipeline) {
        ToolSpecification specification = ToolSpecification.builder()
                .name(TOOL_NAME)
                .description(TOOL_DESCRIPTION)
                .parameters(JsonObjectSchema.builder()
                        .addStringProperty("query", "独立完整的搜索语句，自行消解代词与指代")
                        .required("query")
                        .build())
                .build();

        ToolExecutor executor = (request, memoryId) -> {
            String query = extractQuery(request.arguments());
            return joinTexts(pipeline.retrieve(query, List.of()));
        };
        return new AgentTool(specification, executor);
    }

    private static String extractQuery(String arguments) {
        try {
            com.fasterxml.jackson.databind.JsonNode root =
                    new com.fasterxml.jackson.databind.ObjectMapper().readTree(arguments);
            String query = root.path("query").asText(null);
            return query == null ? "" : query;
        } catch (Exception e) {
            // 容错：模型偶发输出非 JSON 参数时直接按原文当查询
            return arguments == null ? "" : arguments;
        }
    }

    private static String joinTexts(List<RagDocument> documents) {
        StringBuilder result = new StringBuilder();
        for (RagDocument document : documents) {
            if (!result.isEmpty()) {
                result.append("\n\n");
            }
            result.append(document.getText());
        }
        return result.toString();
    }
}
