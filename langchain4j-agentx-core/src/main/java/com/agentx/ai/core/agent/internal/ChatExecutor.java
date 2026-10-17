package com.agentx.ai.core.agent.internal;

import com.agentx.ai.core.exception.AgentErrorCode;
import com.agentx.ai.core.exception.AgentException;
import com.agentx.ai.core.hook.AfterCallEvent;
import com.agentx.ai.core.hook.AfterReasoningEvent;
import com.agentx.ai.core.hook.AgentHook;
import com.agentx.ai.core.hook.BeforeCallEvent;
import com.agentx.ai.core.hook.BeforeReasoningEvent;
import com.agentx.ai.core.hook.ErrorEvent;
import com.agentx.ai.core.hook.HookManager;
import com.agentx.ai.core.memory.store.ConversationStore;
import com.agentx.ai.core.memory.store.SessionMessageStore;
import com.agentx.ai.core.memory.util.MemoryInjector;
import com.agentx.ai.core.memory.util.MemoryPersistor;
import com.agentx.ai.core.model.AgentResult;
import com.agentx.ai.core.model.AgentStreamEvent;
import com.agentx.ai.core.model.AgentTool;
import com.agentx.ai.core.model.RunnableParams;
import com.agentx.ai.core.model.ThinkingMode;
import com.agentx.ai.core.stage.AgentRuntimeContext;
import com.agentx.ai.core.trace.TraceStore;
import com.agentx.ai.core.utils.JsonRepairUtil;
import com.agentx.ai.core.utils.MessageJsonSerializer;
import com.fasterxml.jackson.databind.ObjectMapper;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.output.TokenUsage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import reactor.core.Disposable;
import reactor.core.publisher.Flux;
import reactor.core.publisher.SignalType;
import reactor.core.publisher.Sinks;
import reactor.core.scheduler.Schedulers;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

/**
 * 普通 Agent（ChatAgent）的执行内核：单轮对话 + 有界工具调用。
 * <p>
 * 与 {@link AgentLoopExecutor}（ReAct 多轮循环）共享同一套基础设施：
 * {@link LoopMessageBuilder}（指令 + 会话历史 + 输出格式指令）、
 * {@link ThinkingModeProcessor}（思考模型双通道适配）、
 * {@link LlmInvoker}（流式调用 + 透明重试）、{@link ToolCallExecutor}（工具执行与回填）、
 * {@link SessionPersister}（agentx_session / agentx_conversation / agentx_trace 落库）、
 * {@link HookManager}（生命周期事件）、{@link AgentTaskManager}（并发控制与流式停止）。
 * <p>
 * 与 ReAct 引擎的区别是"有界"：工具轮次由 {@code maxToolRounds} 限制（默认 1），
 * 用尽后最后一轮不再携带工具规格，模型必然产出最终答案（LLM 调用次数 ≤ maxToolRounds + 1）。
 * 因此没有无界循环，也没有 HITL 拦截、暂停快照、上下文压缩、ToolSearch 与沙箱。
 *
 * @author bigchui
 */
public class ChatExecutor {

    private static final Logger log = LoggerFactory.getLogger(ChatExecutor.class);

    private final AgentTaskManager taskManager;
    private final LoopMessageBuilder messageBuilder;
    private final HookManager hookManager;
    private final ThinkingMode thinkingMode;
    private final ThinkingModeProcessor thinkingModeProcessor;
    private final LlmInvoker llmInvoker;
    private final SessionPersister sessionPersister;
    private final ToolCallExecutor toolCallExecutor;

    /** 允许的工具执行轮数；0 表示不向 LLM 暴露工具（等价普通对话） */
    private final int maxToolRounds;
    /** 是否启用工具：注册了工具且 maxToolRounds > 0 */
    private final boolean toolsEnabled;

    private ChatExecutor(Builder builder) {
        this.taskManager = builder.taskManager;
        this.thinkingMode = builder.thinkingMode;
        this.thinkingModeProcessor = new ThinkingModeProcessor(builder.thinkingMode);
        this.hookManager = builder.hooks != null && !builder.hooks.isEmpty()
                ? new HookManager(builder.hooks)
                : HookManager.EMPTY;
        this.maxToolRounds = Math.max(0, builder.maxToolRounds);

        // 工具注册表（基础 Function Calling：无 deferred 工具 / HITL 拦截）
        List<AgentTool> tools = builder.tools != null ? builder.tools : List.of();
        Map<String, AgentTool> toolMap = new HashMap<>();
        List<LlmInvoker.AgentToolHolder> toolHolders = new ArrayList<>();
        for (AgentTool t : tools) {
            toolMap.put(t.getToolDefinition().name(), t);
            toolHolders.add(new LlmInvoker.AgentToolHolder(t.getToolDefinition(), t));
        }
        this.toolsEnabled = !toolMap.isEmpty() && maxToolRounds > 0;
        this.toolCallExecutor = new ToolCallExecutor(toolMap, new ObjectMapper(), null, hookManager);

        this.llmInvoker = new LlmInvoker(builder.chatModel, builder.streamingChatModel,
                builder.maxRetries, toolHolders, null, null);

        // 会话/Trace/暂停状态 统一持久化入口（ChatAgent 不使用暂停状态，stateStore 传 null）
        MemoryPersistor memoryPersistor = new MemoryPersistor(null);
        this.sessionPersister = new SessionPersister(
                builder.enableSession, builder.enableTrace,
                builder.conversationStore, builder.sessionMessageStore,
                builder.traceStore, null, memoryPersistor);

        // 消息构建器（无 deferred 工具与 TodoWrite，仅指令 + 历史 + 输出格式）
        this.messageBuilder = new LoopMessageBuilder(
                builder.instructions, builder.sessionMessageStore,
                new MemoryInjector(null), builder.thinkingMode,
                null, false, builder.enableSession);
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private AgentTaskManager taskManager;
        private SessionMessageStore sessionMessageStore;
        private ConversationStore conversationStore;
        private String instructions;
        private ChatModel chatModel;
        private StreamingChatModel streamingChatModel;
        private boolean enableSession = true;
        private boolean enableTrace = true;
        private List<AgentHook> hooks;
        private ThinkingMode thinkingMode = ThinkingMode.DISABLED;
        private int maxRetries = 3;
        private TraceStore traceStore;
        private List<AgentTool> tools;
        private int maxToolRounds = 1;

        public Builder taskManager(AgentTaskManager v) {
            this.taskManager = v;
            return this;
        }

        public Builder sessionMessageStore(SessionMessageStore v) {
            this.sessionMessageStore = v;
            return this;
        }

        public Builder conversationStore(ConversationStore v) {
            this.conversationStore = v;
            return this;
        }

        public Builder instructions(String v) {
            this.instructions = v;
            return this;
        }

        public Builder chatModel(ChatModel v) {
            this.chatModel = v;
            return this;
        }

        public Builder streamingChatModel(StreamingChatModel v) {
            this.streamingChatModel = v;
            return this;
        }

        public Builder enableSession(boolean v) {
            this.enableSession = v;
            return this;
        }

        public Builder enableTrace(boolean v) {
            this.enableTrace = v;
            return this;
        }

        public Builder hooks(List<AgentHook> v) {
            this.hooks = v;
            return this;
        }

        public Builder thinkingMode(ThinkingMode v) {
            this.thinkingMode = v != null ? v : ThinkingMode.DISABLED;
            return this;
        }

        public Builder maxRetries(int v) {
            this.maxRetries = v;
            return this;
        }

        public Builder traceStore(TraceStore v) {
            this.traceStore = v;
            return this;
        }

        public Builder tools(List<AgentTool> v) {
            this.tools = v;
            return this;
        }

        public Builder maxToolRounds(int v) {
            this.maxToolRounds = v;
            return this;
        }

        public ChatExecutor build() {
            if (chatModel == null && streamingChatModel == null) {
                throw new IllegalArgumentException("chatModel / streamingChatModel 至少提供一个");
            }
            return new ChatExecutor(this);
        }
    }

    // ==================== 同步调用 ====================

    /**
     * 非流式执行。REASONING_CONTENT 模式下内部走流式收集思考内容。
     */
    public AgentResult call(String query, RunnableParams params) {
        String conversationId = params != null ? params.getConversationId() : null;
        if (taskManager != null && conversationId != null) {
            if (taskManager.registerTask(conversationId, null) == null) {
                return new AgentResult.Failed("该会话正在执行中，请稍后再试: " + conversationId,
                        AgentErrorCode.CONCURRENT_EXECUTION);
            }
        }
        try {
            return callViaStreamForResult(query, params);
        } finally {
            if (taskManager != null && conversationId != null) {
                taskManager.removeTask(conversationId);
            }
        }
    }

    /**
     * REASONING_CONTENT 模式内部走流式收集，blockLast 后同步取回结果（对齐 AgentLoopExecutor）。
     */
    private AgentResult callViaStreamForResult(String query, RunnableParams params) {
        BuiltMessages built = messageBuilder.buildInitialMessages(query, params);
        List<ChatMessage> messages = built.messages();
        Sinks.Many<AgentStreamEvent> sink = Sinks.many().unicast().onBackpressureBuffer();
        AgentRuntimeContext execCtx = new AgentRuntimeContext(query, params);
        execCtx.setNewMsgStartIndex(built.newMsgStartIndex());
        execCtx.setOriginalMessagesSnapshot(new ArrayList<>(messages));
        sessionPersister.initSession(execCtx, params, query);

        if (!hookManager.isEmpty()) {
            hookManager.fireEvent(new BeforeCallEvent(execCtx));
        }

        AtomicLong roundCounter = new AtomicLong(0);
        executeRound(messages, sink, params, execCtx, roundCounter, 0, toolsEnabled, 0);

        StringBuilder answer = new StringBuilder();
        StringBuilder think = new StringBuilder();
        String[] errorHolder = {null};

        sink.asFlux()
                .doOnNext(event -> {
                    switch (event) {
                        case AgentStreamEvent.Text t -> answer.append(t.content());
                        case AgentStreamEvent.Thinking t -> think.append(t.content());
                        case AgentStreamEvent.Error e -> errorHolder[0] = e.message();
                        // 工具调用轮清空缓冲，只保留最后一轮（与 ReAct 引擎一致）
                        case AgentStreamEvent.ToolStart ts -> {
                            answer.setLength(0);
                            think.setLength(0);
                        }
                        default -> {
                        }
                    }
                })
                .blockLast();

        if (errorHolder[0] != null) {
            // 重试耗尽路径已由 handleStreamError 的 onTerminalFailure 显式落库，这里仅兜底
            execCtx.markTerminal("error");
            sessionPersister.persistOnTerminal(execCtx, messages, SignalType.ON_ERROR);
            fireAfterCallOnce(execCtx, null, 0);
            return new AgentResult.Failed(errorHolder[0], AgentErrorCode.LLM_CALL_FAILED);
        }

        String finalAnswer = answer.toString();
        if (params != null && params.getOutputType() != null) {
            finalAnswer = JsonRepairUtil.fixJson(finalAnswer);
        }

        return new AgentResult.Completed(
                finalAnswer,
                think.length() > 0 ? think.toString() : null);
    }

    // ==================== 流式调用 ====================

    /**
     * 流式执行，返回 AgentStreamEvent 流。
     */
    public Flux<AgentStreamEvent> stream(String query, RunnableParams params) {
        String conversationId = params != null ? params.getConversationId() : null;
        BuiltMessages built = messageBuilder.buildInitialMessages(query, params);
        List<ChatMessage> messages = built.messages();

        Sinks.Many<AgentStreamEvent> sink = Sinks.many().unicast().onBackpressureBuffer();

        if (taskManager != null && conversationId != null) {
            AgentTaskManager.TaskInfo taskInfo = taskManager.registerTask(conversationId, sink);
            if (taskInfo == null) {
                return Flux.error(new AgentException(AgentErrorCode.CONCURRENT_EXECUTION,
                        "该会话正在执行中，请稍后再试: " + conversationId));
            }
        }

        AgentRuntimeContext execCtx = new AgentRuntimeContext(query, params);
        execCtx.setNewMsgStartIndex(built.newMsgStartIndex());
        execCtx.setOriginalMessagesSnapshot(new ArrayList<>(messages));
        execCtx.setMessages(messages);
        execCtx.setEmitter(sink::tryEmitNext);
        sessionPersister.initSession(execCtx, params, query);

        if (!hookManager.isEmpty()) {
            hookManager.fireEvent(new BeforeCallEvent(execCtx));
        }

        AtomicLong roundCounter = new AtomicLong(0);
        executeRound(messages, sink, params, execCtx, roundCounter, 0, toolsEnabled, 0);

        return wrapStreamFlux(sink, conversationId, messages, execCtx);
    }

    // ==================== 内部执行 ====================

    /**
     * 发起一轮 LLM 调用。
     *
     * @param toolRoundsUsed 已执行的工具轮数（用于 maxToolRounds 判定）
     * @param includeTools   本轮是否携带工具规格（用尽工具轮次后的最后一轮为 false）
     */
    private void executeRound(List<ChatMessage> messages, Sinks.Many<AgentStreamEvent> sink,
                              RunnableParams params, AgentRuntimeContext execCtx,
                              AtomicLong roundCounter, int toolRoundsUsed,
                              boolean includeTools, int retryAttempt) {
        long round = roundCounter.incrementAndGet();
        String conversationId = params != null ? params.getConversationId() : null;
        log.debug("Scheduling chat round: conversationId={}, round={}, includeTools={}, toolRoundsUsed={}, retryAttempt={}",
                conversationId, round, includeTools, toolRoundsUsed, retryAttempt);

        RoundState roundState = new RoundState();
        roundState.requestJson = serializeRequestJson(messages);
        long startTime = System.currentTimeMillis();

        if (!hookManager.isEmpty()) {
            hookManager.fireEvent(new BeforeReasoningEvent(execCtx, round));
        }

        ChatRequest.Builder requestBuilder = ChatRequest.builder().messages(messages);
        if (includeTools) {
            requestBuilder.toolSpecifications(llmInvoker.buildRoundToolSpecifications());
        }

        Disposable disposable = llmInvoker.stream(requestBuilder.build())
                .publishOn(Schedulers.boundedElastic())
                .doOnNext(chunk -> processChunk(chunk, roundState, sink))
                .doOnComplete(() -> finishRound(messages, sink, roundState, params, execCtx, round,
                        System.currentTimeMillis() - startTime, roundCounter, toolRoundsUsed, includeTools))
                .onErrorResume(err -> {
                    if (!hookManager.isEmpty()) {
                        hookManager.fireEvent(new ErrorEvent(execCtx, err, "chat",
                                retryAttempt, retryAttempt < llmInvoker.getMaxRetries()));
                    }
                    return llmInvoker.handleStreamError(err, retryAttempt, sink,
                            () -> executeRound(messages, sink, params, execCtx, roundCounter,
                                    toolRoundsUsed, includeTools, retryAttempt + 1),
                            "Chat stream error",
                            () -> {
                                execCtx.markTerminal("error");
                                sessionPersister.persistOnTerminal(execCtx, messages, SignalType.ON_ERROR);
                            });
                })
                .subscribe();

        // 刷新 taskManager 的 disposable，保证 stopTask 能取消当前在飞的调用
        if (taskManager != null && conversationId != null) {
            taskManager.setDisposable(conversationId, disposable);
        }
    }

    private void processChunk(LlmInvoker.LlmChunk chunk, RoundState state,
                              Sinks.Many<AgentStreamEvent> sink) {
        if (chunk instanceof LlmInvoker.LlmChunk.Text t) {
            thinkingModeProcessor.processStreamChunk(t.token(), state, sink);
        } else if (chunk instanceof LlmInvoker.LlmChunk.Thinking th) {
            // 独立思考通道（reasoning_content / thinking）：REASONING_CONTENT 模式才累积
            if (thinkingMode == ThinkingMode.REASONING_CONTENT) {
                thinkingModeProcessor.processReasoningChunk(th.delta(), state, sink);
            }
        } else if (chunk instanceof LlmInvoker.LlmChunk.ToolCalls tc) {
            state.toolCalls.addAll(tc.requests());
        } else if (chunk instanceof LlmInvoker.LlmChunk.Metadata meta) {
            ChatResponse response = meta.response();
            if (response != null) {
                TokenUsage usage = response.tokenUsage();
                if (usage != null) {
                    state.promptTokens = usage.inputTokenCount() != null ? usage.inputTokenCount() : 0;
                    state.completionTokens = usage.outputTokenCount() != null ? usage.outputTokenCount() : 0;
                }
            }
        }
    }

    /**
     * 一轮 LLM 调用完成：无工具调用则收敛终态；有工具调用则执行工具并进入下一轮
     * （用尽工具轮次后下一轮不再携带工具规格，模型必然产出最终答案）。
     */
    private void finishRound(List<ChatMessage> messages, Sinks.Many<AgentStreamEvent> sink,
                             RoundState state, RunnableParams params, AgentRuntimeContext execCtx,
                             long round, long durationMs, AtomicLong roundCounter,
                             int toolRoundsUsed, boolean includeTools) {
        if (state.toolCalls.isEmpty() || !includeTools) {
            if (!state.toolCalls.isEmpty()) {
                // 未携带工具规格时模型不应返回 tool calls，兜底按最终回答处理
                log.warn("本轮未启用工具但模型返回了 tool calls，按最终回答处理: round={}, conversationId={}",
                        round, params != null ? params.getConversationId() : null);
            }
            completeWithAnswer(messages, sink, state, params, execCtx, round, durationMs);
            return;
        }

        // === 工具调用轮 ===
        List<ToolExecutionRequest> safeToolCalls = toolCallExecutor.sanitizeToolCalls(state.toolCalls);

        AiMessage.Builder assistantBuilder = AiMessage.builder()
                .text(state.textBuffer.isEmpty() ? "" : state.textBuffer.toString())
                .toolExecutionRequests(safeToolCalls);
        if (!state.reasoningBuffer.isEmpty()) {
            assistantBuilder.thinking(state.reasoningBuffer.toString());
        }
        AiMessage assistantMsg = assistantBuilder.build();
        messages.add(assistantMsg);
        execCtx.appendOriginalMessage(assistantMsg);

        String roundThink = state.reasoningBuffer.length() > 0 ? state.reasoningBuffer.toString() : null;

        // trace（工具调用轮）：记录模型给出的 tool calls
        sessionPersister.recordTrace(execCtx, (int) round, state.requestJson,
                sessionPersister.serializeToolCalls(safeToolCalls), roundThink,
                state.promptTokens, state.completionTokens, durationMs);

        if (!hookManager.isEmpty()) {
            hookManager.fireEvent(new AfterReasoningEvent(execCtx, state.textBuffer.toString(),
                    safeToolCalls, round, state.promptTokens, state.completionTokens, durationMs));
        }

        int usedAfter = toolRoundsUsed + 1;
        boolean nextIncludeTools = usedAfter < maxToolRounds;
        log.debug("工具轮次 {}/{}，下一轮 includeTools={}, conversationId={}",
                usedAfter, maxToolRounds, nextIncludeTools,
                params != null ? params.getConversationId() : null);

        toolCallExecutor.executeToolCallsAsync(sink, safeToolCalls, messages, params, execCtx,
                () -> executeRound(messages, sink, params, execCtx, roundCounter,
                        usedAfter, nextIncludeTools, 0));
    }

    /**
     * 收敛终态：把本轮文本作为最终回答落库并发射 Complete。
     */
    private void completeWithAnswer(List<ChatMessage> messages, Sinks.Many<AgentStreamEvent> sink,
                                    RoundState state, RunnableParams params, AgentRuntimeContext execCtx,
                                    long round, long durationMs) {
        String conversationId = params != null ? params.getConversationId() : null;

        // 将累积的文本作为 AiMessage 添加到 messages，供终态落库与下一轮历史捕获
        if (!state.textBuffer.isEmpty() || !state.reasoningBuffer.isEmpty()) {
            AiMessage.Builder aiBuilder = AiMessage.builder().text(state.textBuffer.toString());
            if (!state.reasoningBuffer.isEmpty()) {
                aiBuilder.thinking(state.reasoningBuffer.toString());
            }
            AiMessage finalAssistant = aiBuilder.build();
            messages.add(finalAssistant);
            execCtx.appendOriginalMessage(finalAssistant);
        }

        String answer = state.textBuffer.toString();
        String roundThink = state.reasoningBuffer.length() > 0 ? state.reasoningBuffer.toString() : null;

        sessionPersister.recordTrace(execCtx, (int) round, state.requestJson, answer,
                roundThink, state.promptTokens, state.completionTokens, durationMs);

        if (!hookManager.isEmpty()) {
            hookManager.fireEvent(new AfterReasoningEvent(execCtx, answer, List.of(), round,
                    state.promptTokens, state.completionTokens, durationMs));
        }

        // 标记终态：显式落库后再 complete sink，避免 doFinally 时序竞态
        execCtx.markTerminal("completed");
        sessionPersister.persistOnTerminal(execCtx, messages, SignalType.ON_COMPLETE);
        fireAfterCallOnce(execCtx, answer, durationMs);
        Sinks.EmitResult completeResult = sink.tryEmitNext(new AgentStreamEvent.Complete(
                execCtx.getTotalPromptTokens(), execCtx.getTotalCompletionTokens(),
                conversationId, execCtx.getSessionId(), null));
        log.debug("tryEmitNext(Complete) result={}, conversationId={}", completeResult, conversationId);
        sink.tryEmitComplete();
    }

    /**
     * 流式 Flux 统一加终态落库兜底和任务清理（对齐 AgentLoopExecutor.wrapStreamFlux）。
     */
    private Flux<AgentStreamEvent> wrapStreamFlux(Sinks.Many<AgentStreamEvent> sink,
                                                  String conversationId,
                                                  List<ChatMessage> messages,
                                                  AgentRuntimeContext execCtx) {
        return sink.asFlux()
                .doFinally(signal -> {
                    // 异常终止兜底：正常完成路径已在 completeWithAnswer 显式落库，CAS 保证不双写
                    SignalType fallbackSignal = signal == SignalType.ON_ERROR ? SignalType.ON_ERROR : SignalType.CANCEL;
                    sessionPersister.persistOnTerminal(execCtx, messages, fallbackSignal);
                    fireAfterCallOnce(execCtx, null, 0);
                    if (taskManager != null && conversationId != null) {
                        taskManager.stopTask(conversationId);
                    }
                });
    }

    private void fireAfterCallOnce(AgentRuntimeContext execCtx, String answer, long durationMs) {
        if (!hookManager.isEmpty() && execCtx.tryFireAfterCall()) {
            hookManager.fireEvent(new AfterCallEvent(execCtx, answer, durationMs));
        }
    }

    /**
     * 序列化请求消息为 JSON（trace 用，对齐 AgentLoopExecutor 的 requestJson 记录方式）。
     */
    private String serializeRequestJson(List<ChatMessage> messages) {
        try {
            return MessageJsonSerializer.toJson(messages);
        } catch (Exception e) {
            return null;
        }
    }
}
