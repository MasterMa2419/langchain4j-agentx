package com.agentx.ai.core.chatmodels;

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.StreamingChatModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Objects;

/**
 * Agent 模型统一入口：一次构造、同时持有同步 {@link ChatModel} 与流式 {@link StreamingChatModel}。
 *
 * <h2>为什么需要一对模型</h2>
 * <p>ReAct 主循环走流式（对外 {@code stream} / {@code streamForResult} 的 token 级事件），
 * 上下文压缩摘要、长期记忆抽取、ToolSearch 等旁路走同步阻塞调用。LangChain4j 将两者拆成两个接口，
 * 对接 OpenAI 兼容 API 时往往是两个实例，因此推荐通过本类一次收口，避免配置分叉。
 *
 * <h2>推荐用法</h2>
 * <pre>{@code
 * // 1) 同一实例同时实现两个接口时，自动复用（无需适配器）
 * AgentModels models = AgentModels.of(chatModel);
 *
 * // 2) 同步 + 流式两个实例（OpenAI 兼容 API 常见）
 * AgentModels models = AgentModels.of(
 *         OpenAiChatModel.builder()...build(),
 *         OpenAiStreamingChatModel.builder()...build());
 *
 * ReactAgent.builder().models(models)...build();
 * }</pre>
 *
 * <h2>降级适配</h2>
 * <p>只提供一侧模型时，框架会包装出另一侧并打 WARN 日志：
 * <ul>
 *   <li>仅同步 → 流式侧为「整段一次吐出」伪流式，无 token 级增量</li>
 *   <li>仅流式 → 同步侧收集整段流后返回，旁路调用需等流结束</li>
 * </ul>
 * 生产环境若依赖真实流式体验，请显式提供 {@link StreamingChatModel}。
 *
 * @author bigchui
 */
public final class AgentModels {

    private static final Logger log = LoggerFactory.getLogger(AgentModels.class);

    private final ChatModel chatModel;
    private final StreamingChatModel streamingChatModel;
    private final boolean streamingAdapted;
    private final boolean chatAdapted;

    private AgentModels(ChatModel chatModel, StreamingChatModel streamingChatModel,
                        boolean streamingAdapted, boolean chatAdapted) {
        this.chatModel = chatModel;
        this.streamingChatModel = streamingChatModel;
        this.streamingAdapted = streamingAdapted;
        this.chatAdapted = chatAdapted;
    }

    /**
     * 显式指定同步 + 流式一对模型（推荐对接外部 API 时使用）。
     */
    public static AgentModels of(ChatModel chatModel, StreamingChatModel streamingChatModel) {
        Objects.requireNonNull(chatModel, "chatModel must not be null");
        Objects.requireNonNull(streamingChatModel, "streamingChatModel must not be null");
        return new AgentModels(chatModel, streamingChatModel, false, false);
    }

    /**
     * 只传一个同步模型。若其实现了 {@link StreamingChatModel} 则原样复用；
     * 否则包装出伪流式适配器（整段一次吐出）。
     */
    public static AgentModels of(ChatModel chatModel) {
        Objects.requireNonNull(chatModel, "chatModel must not be null");
        if (chatModel instanceof StreamingChatModel streaming) {
            return new AgentModels(chatModel, streaming, false, false);
        }
        log.warn("[AgentModels] chatModel 未实现 StreamingChatModel，已包装为伪流式适配器"
                + "（无 token 级增量）。需要真实流式请显式传入 StreamingChatModel，"
                + "或使用 AgentModels.of(chatModel, streamingChatModel)");
        return new AgentModels(chatModel, new SyncAsStreamingChatModel(chatModel), true, false);
    }

    /**
     * 只传一个流式模型。包装出同步适配器（收集整段流），供压缩 / 记忆 / ToolSearch 等旁路使用。
     * <p>方法名独立（而非 {@code of} 重载），避免与「同时实现两个接口」的实例产生歧义。
     */
    public static AgentModels ofStreaming(StreamingChatModel streamingChatModel) {
        Objects.requireNonNull(streamingChatModel, "streamingChatModel must not be null");
        if (streamingChatModel instanceof ChatModel chatModel) {
            return new AgentModels(chatModel, streamingChatModel, false, false);
        }
        log.warn("[AgentModels] 仅提供 StreamingChatModel，同步侧已包装为收集整段流的适配器。"
                + "旁路调用（上下文压缩 / 长期记忆 / ToolSearch）将阻塞至流结束。");
        return new AgentModels(new StreamingAsSyncChatModel(streamingChatModel), streamingChatModel,
                false, true);
    }

    /**
     * 按已有的可空字段解析：优先双实例，其次单侧 + 自动复用 / 适配。
     *
     * @param chatModel        同步模型，可空
     * @param streamingChatModel 流式模型，可空
     * @throws IllegalArgumentException 两侧都为空
     */
    public static AgentModels resolve(ChatModel chatModel, StreamingChatModel streamingChatModel) {
        if (chatModel != null && streamingChatModel != null) {
            return of(chatModel, streamingChatModel);
        }
        if (chatModel != null) {
            return of(chatModel);
        }
        if (streamingChatModel != null) {
            return ofStreaming(streamingChatModel);
        }
        throw new IllegalArgumentException(
                "至少需要提供一个模型：chatModel 或 streamingChatModel（推荐 AgentModels.of(...))");
    }

    /**
     * 同步模型：上下文压缩、长期记忆、ToolSearch 等阻塞调用使用。
     */
    public ChatModel chatModel() {
        return chatModel;
    }

    /**
     * 流式模型：ReAct 主循环与对外 token 流使用。
     */
    public StreamingChatModel streamingChatModel() {
        return streamingChatModel;
    }

    /**
     * 流式侧是否为适配器（伪流式，无 token 级增量）。
     */
    public boolean isStreamingAdapted() {
        return streamingAdapted;
    }

    /**
     * 同步侧是否为适配器（收集整段流）。
     */
    public boolean isChatAdapted() {
        return chatAdapted;
    }

    @Override
    public String toString() {
        return "AgentModels[chatModel=" + chatModel.getClass().getSimpleName()
                + ", streamingChatModel=" + streamingChatModel.getClass().getSimpleName()
                + ", streamingAdapted=" + streamingAdapted
                + ", chatAdapted=" + chatAdapted + "]";
    }
}
