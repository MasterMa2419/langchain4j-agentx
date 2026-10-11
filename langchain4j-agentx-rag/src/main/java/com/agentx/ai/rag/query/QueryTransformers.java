package com.agentx.ai.rag.query;

import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatModel;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * QueryTransformer 快速构建工厂。
 *
 * 传入 ChatModel 即得默认配置，适合首次跑通流程。需要自定义提示词或参数时，
 * 直接实现 {@link QueryTransformer} 接口。
 *
 * @author bigchui
 */
public final class QueryTransformers {

    private QueryTransformers() {
    }

    /**
     * 问题压缩：结合对话历史消解指代，把追问合成独立 query，同时做精炼。
     *
     * 例：历史"如何创建 Skill" + 追问"那第三种方法呢" → "如何用 npx skills add 安装 Skill"。
     * 多轮对话场景使用。需要传 chatHistory 到 pipeline 的 retrieve 方法才会带上历史。
     */
    public static QueryTransformer compression(ChatModel chatModel) {
        Objects.requireNonNull(chatModel, "chatModel");
        return new CompressionTransformer(chatModel);
    }

    /**
     * 问题改写：只针对当前 query 本身，去除口语、冗余与歧义，不涉及对话历史。
     *
     * 例："那个 AI 里的技能怎么弄来着" → "如何创建 Claude Code 的 Skill"。
     * 单轮检索场景使用。与 compression 二选一即可，不必同时启用。
     */
    public static QueryTransformer rewrite(ChatModel chatModel) {
        Objects.requireNonNull(chatModel, "chatModel");
        return new RewriteTransformer(chatModel);
    }

    /**
     * HyDE 假设答案：先让 LLM 生成一个假设答案，用假设答案的向量去检索，
     * 缓解短问题与文档之间的语义鸿沟。
     */
    public static QueryTransformer hyde(ChatModel chatModel) {
        return new HydeQueryTransformer(chatModel);
    }

    private abstract static class LlmQueryTransformer implements QueryTransformer {

        protected final ChatModel chatModel;

        protected LlmQueryTransformer(ChatModel chatModel) {
            this.chatModel = Objects.requireNonNull(chatModel, "chatModel");
        }

        protected abstract String systemPrompt();

        protected String userPrompt(Query query) {
            return query.text();
        }

        protected List<ChatMessage> historyMessages(Query query) {
            return query.history();
        }

        @Override
        public Query transform(Query query) {
            Objects.requireNonNull(query, "query");

            List<ChatMessage> messages = new ArrayList<>();
            messages.add(SystemMessage.from(systemPrompt()));
            messages.addAll(historyMessages(query));
            messages.add(UserMessage.from(userPrompt(query)));

            String rewritten = chatModel.chat(messages).aiMessage().text();
            String text = rewritten == null || rewritten.isBlank() ? query.text() : rewritten.trim();
            return Query.builder()
                    .text(text)
                    .history(query.history())
                    .context(query.context())
                    .build();
        }
    }

    private static final class CompressionTransformer extends LlmQueryTransformer {

        private static final String PROMPT = """
                你是查询改写助手。结合对话历史，把用户最新问题改写成一个独立、完整、
                不依赖上下文指代的检索查询。消解代词，合并历史中的关键信息，去除口语。
                只输出改写后的查询本身，不要解释。
                """;

        private CompressionTransformer(ChatModel chatModel) {
            super(chatModel);
        }

        @Override
        protected String systemPrompt() {
            return PROMPT;
        }

        @Override
        protected String userPrompt(Query query) {
            StringBuilder builder = new StringBuilder();
            if (!query.history().isEmpty()) {
                builder.append("对话历史：\n");
                for (ChatMessage message : query.history()) {
                    builder.append("- ").append(extractText(message)).append('\n');
                }
            }
            builder.append("最新问题：").append(query.text());
            return builder.toString();
        }
    }

    private static final class RewriteTransformer extends LlmQueryTransformer {

        private static final String PROMPT = """
                你是查询改写助手。把用户问题改写成适合知识库检索的独立查询：
                去除口语、冗余与歧义，不引入对话历史。
                只输出改写后的查询本身，不要解释。
                """;

        private RewriteTransformer(ChatModel chatModel) {
            super(chatModel);
        }

        @Override
        protected String systemPrompt() {
            return PROMPT;
        }
    }

    private static String extractText(ChatMessage message) {
        if (message instanceof UserMessage userMessage) {
            return userMessage.singleText();
        }
        if (message instanceof SystemMessage systemMessage) {
            return systemMessage.text();
        }
        return message.toString();
    }
}
