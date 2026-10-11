package com.agentx.ai.core.agent.internal;

import com.agentx.ai.core.exception.AgentErrorCode;
import com.agentx.ai.core.model.AgentStreamEvent;
import com.agentx.ai.core.tools.toolsearch.DeferredToolRegistry;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.PartialThinking;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import reactor.core.publisher.Flux;
import reactor.core.publisher.FluxSink;
import reactor.core.publisher.Sinks;
import reactor.core.scheduler.Schedulers;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * LLM 调用器 — 负责流式 LLM 调用（含重试）与每轮 ChatRequest 组装。
 *
 * <ul>
 *   <li>直调 LC4j low-level {@link StreamingChatModel}，TokenStream 桥接为 Flux&lt;LlmChunk&gt;</li>
 *   <li>每轮根据 deferred 工具发现状态重建 {@link ChatRequest#toolSpecifications()}</li>
 *   <li>手写重试循环（1.x 依赖 spring-retry / WebClientResponseException，已移除）</li>
 * </ul>
 *
 * @author bigchui
 */
public class LlmInvoker {

    private static final Logger log = LoggerFactory.getLogger(LlmInvoker.class);

    static final long RETRY_INTERVAL_MS = 10000;

    private final ChatModel chatModel;
    private final StreamingChatModel streamingChatModel;
    private final int maxRetries;
    private final List<AgentToolHolder> alwaysLoadTools;
    private final DeferredToolRegistry deferredToolRegistry;
    private final DeferredToolRegistry.Session deferredToolSession;

    /**
     * 工具持有者：工具定义 + 引擎侧执行器。
     */
    public record AgentToolHolder(ToolSpecification specification,
                                  com.agentx.ai.core.model.AgentTool agentTool) {
    }

    public LlmInvoker(ChatModel chatModel, StreamingChatModel streamingChatModel,
                      int maxRetries, List<AgentToolHolder> alwaysLoadTools,
                      DeferredToolRegistry deferredToolRegistry,
                      DeferredToolRegistry.Session deferredToolSession) {
        this.chatModel = chatModel;
        this.streamingChatModel = streamingChatModel != null ? streamingChatModel
                : (chatModel instanceof StreamingChatModel scm ? scm : null);
        this.maxRetries = maxRetries;
        this.alwaysLoadTools = alwaysLoadTools;
        this.deferredToolRegistry = deferredToolRegistry;
        this.deferredToolSession = deferredToolSession;
    }

    /**
     * 构建当前轮次的工具规格列表。
     * 无 deferredToolRegistry 时返回固定列表；否则合并已发现的 deferred 工具与 tool_search 元工具。
     */
    public List<ToolSpecification> buildRoundToolSpecifications() {
        List<ToolSpecification> specs = new ArrayList<>();
        for (AgentToolHolder holder : alwaysLoadTools) {
            specs.add(holder.specification());
        }
        if (deferredToolRegistry != null && deferredToolSession != null) {
            for (com.agentx.ai.core.model.AgentTool tool : deferredToolSession.getActiveDeferredTools()) {
                specs.add(tool.getToolDefinition());
            }
            specs.add(deferredToolSession.getToolSearchCallback().getToolDefinition());
        }
        return specs;
    }

    /**
     * 流式调用 LLM，把 LC4j TokenStream 回调桥接为 Flux&lt;LlmChunk&gt;。
     * 背压由 FluxSink 缓冲策略承接；onError / cancel 通过 cancel 标志传递给上游回调。
     *
     * @param request 已组装的 ChatRequest（含 messages + toolSpecifications）
     */
    public Flux<LlmChunk> stream(ChatRequest request) {
        return Flux.create(sink -> {
            AtomicBoolean cancelled = new AtomicBoolean(false);
            sink.onCancel(() -> cancelled.set(true));
            sink.onDispose(() -> cancelled.set(true));

            try {
                streamingChatModel.chat(request, new StreamingChatResponseHandler() {
                    @Override
                    public void onPartialResponse(String token) {
                        if (!cancelled.get() && token != null && !token.isEmpty()) {
                            sink.next(LlmChunk.text(token));
                        }
                    }

                    @Override
                    public void onPartialThinking(PartialThinking thinking) {
                        if (!cancelled.get() && thinking != null && thinking.text() != null
                                && !thinking.text().isEmpty()) {
                            sink.next(LlmChunk.thinking(thinking.text()));
                        }
                    }

                    @Override
                    public void onCompleteResponse(ChatResponse complete) {
                        if (cancelled.get()) {
                            return;
                        }
                        // 最终 AiMessage 携带完整的 toolExecutionRequests（LC4j 已聚合分片）
                        if (complete != null && complete.aiMessage() != null
                                && complete.aiMessage().hasToolExecutionRequests()) {
                            sink.next(LlmChunk.toolCalls(complete.aiMessage().toolExecutionRequests()));
                        }
                        sink.next(LlmChunk.metadata(complete));
                        sink.complete();
                    }

                    @Override
                    public void onError(Throwable error) {
                        if (!cancelled.get()) {
                            sink.error(error);
                        }
                    }
                });
            } catch (Exception e) {
                sink.error(e);
            }
        });
    }

    /**
     * 处理流式路径的重试/失败逻辑。
     * 统一 scheduleRound 和 forceFinalStream 中的 onErrorResume 重复代码。
     *
     * @param err               异常
     * @param retryAttempt      当前重试次数
     * @param sink              事件接收器
     * @param retryAction       重试时执行的操作
     * @param logLabel          日志标识（如 "LLM stream error"）
     * @param onTerminalFailure 最终失败时（重试耗尽）的回调，在 sink.tryEmitComplete() 之前执行，
     *                          供调用方做终态落库（避免 doFinally 时序竞态）
     * @return Flux.empty() 供 onErrorResume 使用
     */
    public Flux<LlmChunk> handleStreamError(Throwable err, int retryAttempt,
                                            Sinks.Many<AgentStreamEvent> sink,
                                            Runnable retryAction, String logLabel,
                                            Runnable onTerminalFailure) {
        if (retryAttempt < maxRetries) {
            log.warn("{} (attempt {}/{}), retrying in {}ms: {}",
                    logLabel, retryAttempt + 1, maxRetries, RETRY_INTERVAL_MS, err.getMessage(), err);
            String retryMsg = "LLM 调用失败，正在重试 (" + (retryAttempt + 1) + "/" + maxRetries + ")";
            sink.tryEmitNext(new AgentStreamEvent.Error(
                    AgentErrorCode.LLM_CALL_FAILED, retryMsg, err.getMessage()));
            Schedulers.boundedElastic().schedule(retryAction, RETRY_INTERVAL_MS, TimeUnit.MILLISECONDS);
        } else {
            log.error("{} failed after {} retries: {}",
                    logLabel, maxRetries, err.getMessage(), err);
            // 在发射 Error/Complete 之前先做终态落库，避免 doFinally 时序竞态
            if (onTerminalFailure != null) {
                try {
                    onTerminalFailure.run();
                } catch (Exception e) {
                    log.warn("onTerminalFailure callback failed: {}", e.getMessage());
                }
            }
            String failMsg = "LLM 调用失败（已重试 " + maxRetries + " 次）";
            sink.tryEmitNext(new AgentStreamEvent.Error(
                    AgentErrorCode.LLM_CALL_FAILED, failMsg, err.getMessage()));
            sink.tryEmitNext(new AgentStreamEvent.Complete());
            sink.tryEmitComplete();
        }
        return Flux.empty();
    }

    public int getMaxRetries() {
        return maxRetries;
    }

    // ==================== 轮次块模型 ====================

    /**
     * 流式轮次内的数据块：文本 delta、思考 delta、完整工具调用、尾部元数据。
     * LC4j 的 onCompleteResponse 会聚合所有分片并给出完整 toolExecutionRequests，
     * 因此工具调用不再需要逐分片合并（对比 1.x mergeToolCall 逻辑）。
     */
    public sealed interface LlmChunk {

        /** 文本增量 */
        record Text(String token) implements LlmChunk {
        }

        /** 思考内容增量（reasoning_content / thinking 通道） */
        record Thinking(String delta) implements LlmChunk {
        }

        /** 完整的工具调用列表（流结束时由最终 AiMessage 提供） */
        record ToolCalls(List<dev.langchain4j.agent.tool.ToolExecutionRequest> requests) implements LlmChunk {
        }

        /** 尾部元数据（token 用量、finishReason） */
        record Metadata(ChatResponse response) implements LlmChunk {
        }

        static LlmChunk text(String token) {
            return new Text(token);
        }

        static LlmChunk thinking(String delta) {
            return new Thinking(delta);
        }

        static LlmChunk toolCalls(List<dev.langchain4j.agent.tool.ToolExecutionRequest> requests) {
            return new ToolCalls(requests);
        }

        static LlmChunk metadata(ChatResponse response) {
            return new Metadata(response);
        }
    }
}
