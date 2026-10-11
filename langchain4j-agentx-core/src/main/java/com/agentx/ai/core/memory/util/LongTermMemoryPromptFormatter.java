package com.agentx.ai.core.memory.util;

import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.store.embedding.EmbeddingMatch;

import java.util.List;

/**
 * 长期记忆提示词格式化工具。
 *
 * 将向量库检索到的相关记忆文档格式化为 system prompt 区块。
 *
 * @author bigchui
 */
public class LongTermMemoryPromptFormatter {

    private LongTermMemoryPromptFormatter() {
    }

    /**
     * 将检索到的长期记忆格式化为提示词区块。
     */
    public static String formatSection(List<EmbeddingMatch<TextSegment>> matches) {
        if (matches == null || matches.isEmpty()) {
            return "";
        }
        StringBuilder sb = new StringBuilder();
        sb.append("## 长期记忆\n");
        sb.append("以下是从历史对话中提炼的、可跨会话复用的知识片段：\n\n");
        for (int i = 0; i < matches.size(); i++) {
            EmbeddingMatch<TextSegment> match = matches.get(i);
            Metadata metadata = match.embedded().metadata();
            String createdAt = metadata.getString("created_at");
            String datePrefix = "";
            if (createdAt != null && createdAt.length() >= 10) {
                datePrefix = "[" + createdAt.substring(0, 10) + "] ";
            }
            sb.append(i + 1).append(". ").append(datePrefix).append(match.embedded().text()).append("\n");
        }
        sb.append("\n注意：以上记忆来自历史对话，可能已过时。如与用户当前说法矛盾，以用户当前说法为准。");
        return sb.toString();
    }
}
