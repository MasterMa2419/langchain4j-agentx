package com.agentx.ai.rag.query;

import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatModel;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * QueryExpander 快速构建工厂。
 *
 * 传入 ChatModel 即得默认配置。需要自定义扩展数量或提示词时，
 * 直接实现 {@link QueryExpander} 接口。
 *
 * @author bigchui
 */
public final class QueryExpanders {

    private static final String PROMPT = """
            你是查询扩展助手。针对用户问题生成多个语义等价或互补的检索查询变体，
            每行一个，不要编号，不要解释。
            """;

    private QueryExpanders() {
    }

    /**
     * 多查询扩展：1→N 生成多个语义变体，包含原问题。默认扩展 3 个变体。
     */
    public static QueryExpander multiQuery(ChatModel chatModel) {
        return multiQuery(chatModel, 3, true);
    }

    /**
     * 多查询扩展，自定义扩展数量与是否包含原问题。
     */
    public static QueryExpander multiQuery(ChatModel chatModel, int numberOfQueries, boolean includeOriginal) {
        Objects.requireNonNull(chatModel, "chatModel");
        return query -> {
            Objects.requireNonNull(query, "query");
            String answer = chatModel.chat(List.of(
                    SystemMessage.from(PROMPT),
                    UserMessage.from("生成 " + numberOfQueries + " 个查询变体。用户问题：" + query.text())
            )).aiMessage().text();
            List<Query> result = new ArrayList<>();
            if (includeOriginal) {
                result.add(query);
            }
            if (answer != null && !answer.isBlank()) {
                for (String line : answer.lines().toList()) {
                    String variant = line.replaceAll("^[-*•\\d.\\s]+", "").trim();
                    if (!variant.isBlank() && result.stream().noneMatch(q -> q.text().equals(variant))) {
                        result.add(Query.builder()
                                .text(variant)
                                .history(query.history())
                                .context(query.context())
                                .build());
                    }
                }
            }
            return result;
        };
    }
}
