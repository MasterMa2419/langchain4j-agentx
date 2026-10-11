package com.agentx.ai.core.chatmodels;

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.model.openai.OpenAiStreamingChatModel;

/**
 * DeepSeek V4 思考模型工厂（2.0 LangChain4j 版）。
 *
 * <h2>与 1.x 的差异</h2>
 * <p>1.x 中本类是 Spring AI {@code DeepSeekChatModel} 的 fork，用于修复
 * {@code reasoning_content} 未随历史消息回传的 bug（spring-ai#6026）——
 * 该 bug 存在于 Spring AI 的 DeepSeek 实现内部，导致思考模型 + 工具调用组合必然报错。
 *
 * <p>2.0 迁移到 LangChain4j 后，该 bug 的根因不再存在：
 * LC4j 的 OpenAI 客户端原生解析 {@code reasoning_content} 字段
 * （{@code AiMessage.thinking()}），并通过 {@code sendThinking(true)}
 * 在多轮请求中把历史思考内容回传给服务端。
 * 因此本类退化为一个「DeepSeek V4 + 思考模式」的便捷工厂，
 * 底层走 DeepSeek 的 OpenAI 兼容接口。
 *
 * <h2>用法</h2>
 * <pre>{@code
 * // 同步 + 流式一次拿到，注入 ReactAgent
 * AgentModels models = DeepSeekV4ChatModel.builder()
 *         .apiKey("sk-...")
 *         .modelName("deepseek-v4")
 *         .buildWithStreaming();
 * ReactAgent.builder()
 *         .models(models)
 *         .thinkingMode(ThinkingMode.REASONING_CONTENT)
 *         ...
 * }</pre>
 *
 * @author bigchui
 */
public final class DeepSeekV4ChatModel {

    /** DeepSeek OpenAI 兼容接口地址 */
    public static final String DEFAULT_BASE_URL = "https://api.deepseek.com/v1";

    private DeepSeekV4ChatModel() {
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String baseUrl = DEFAULT_BASE_URL;
        private String apiKey;
        private String modelName = "deepseek-v4";
        private Double temperature;
        private Double topP;
        private Integer maxTokens;
        private Boolean returnThinking = true;
        private Boolean sendThinking = true;
        private org.slf4j.Logger logger;

        public Builder baseUrl(String baseUrl) {
            this.baseUrl = baseUrl;
            return this;
        }

        public Builder apiKey(String apiKey) {
            this.apiKey = apiKey;
            return this;
        }

        public Builder modelName(String modelName) {
            this.modelName = modelName;
            return this;
        }

        public Builder temperature(Double temperature) {
            this.temperature = temperature;
            return this;
        }

        public Builder topP(Double topP) {
            this.topP = topP;
            return this;
        }

        public Builder maxTokens(Integer maxTokens) {
            this.maxTokens = maxTokens;
            return this;
        }

        /**
         * 是否解析并返回思考内容（默认 true，写入 {@code AiMessage.thinking()}）。
         */
        public Builder returnThinking(Boolean returnThinking) {
            this.returnThinking = returnThinking;
            return this;
        }

        /**
         * 多轮请求是否把历史思考内容回传给服务端（默认 true；思考模型 + 工具调用的组合需要）。
         */
        public Builder sendThinking(Boolean sendThinking) {
            this.sendThinking = sendThinking;
            return this;
        }

        public Builder logger(org.slf4j.Logger logger) {
            this.logger = logger;
            return this;
        }

        public ChatModel build() {
            OpenAiChatModel.OpenAiChatModelBuilder b = OpenAiChatModel.builder()
                    .baseUrl(baseUrl)
                    .apiKey(apiKey)
                    .modelName(modelName)
                    .returnThinking(returnThinking)
                    .sendThinking(sendThinking);
            if (temperature != null) b.temperature(temperature);
            if (topP != null) b.topP(topP);
            if (maxTokens != null) b.maxTokens(maxTokens);
            if (logger != null) b.logger(logger);
            return b.build();
        }

        public StreamingChatModel buildStreaming() {
            OpenAiStreamingChatModel.OpenAiStreamingChatModelBuilder b = OpenAiStreamingChatModel.builder()
                    .baseUrl(baseUrl)
                    .apiKey(apiKey)
                    .modelName(modelName)
                    .returnThinking(returnThinking);
            if (temperature != null) b.temperature(temperature);
            if (topP != null) b.topP(topP);
            if (maxTokens != null) b.maxTokens(maxTokens);
            if (logger != null) b.logger(logger);
            return b.build();
        }

        /**
         * 同时构建同步与流式模型，返回 {@link AgentModels} 统一入口。
         */
        public AgentModels buildWithStreaming() {
            return AgentModels.of(build(), buildStreaming());
        }
    }
}
