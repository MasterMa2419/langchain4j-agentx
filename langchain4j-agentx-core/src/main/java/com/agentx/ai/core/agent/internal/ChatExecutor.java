package com.agentx.ai.core.agent.internal;

import com.agentx.ai.core.exception.AgentErrorCode;
import com.agentx.ai.core.exception.AgentException;
import com.agentx.ai.core.hook.AgentHook;
import com.agentx.ai.core.hook.AfterCallEvent;
import com.agentx.ai.core.hook.AfterReasoningEvent;
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
import com.agentx.ai.core.model.RunnableParams;
import com.agentx.ai.core.model.ThinkingMode;
import com.agentx.ai.core.stage.AgentRuntimeContext;
import com.agentx.ai.core.trace.TraceStore;
import com.agentx.ai.core.utils.JsonRepairUtil;
import com.agentx.ai.core.utils.MessageJsonSerializer;
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
import java.util.List;

/**
 * 单轮对话执行器 — 普通 Agent（ChatAgent）的执行内核。
 * <p>
 * 与 {@link AgentLoopExecutor}（ReAct 多轮循环）共享同一套基础设施：
 * {@link LoopMessageBuilder}（指令 + 会话历史 + 输出格式指令）、
 * {@link ThinkingModeProcessor}（思考模型双通道适配）、
 * {@link LlmInvoker}（流式调用 + 透明重试）、
 * {@link SessionPersister}（agentx_session / agentx_conversation / agentx_trace 落库）、
 * {@link HookManager}（Before / AfterReasoning / AfterCall / Error 生命周期）、
 * {@link AgentTaskManager}（并发控制与流式停止）。
 * <p>
 * 区别在于：不注册工具规格、没有 ReAct 多轮迭代，一次 LLM 调用收敛到终态，
 * 因此没有 HITL 拦截、暂停快照与上下文压缩。
 *
 * <p>REASONING_CONTENT 思考模式内部走流式收集（对齐 AgentLoopExecutor.call 的处理方式）。
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

    private ChatExecutor(Builder builder) {
        this.taskManager = builder.taskManager;
        this.thinkingMode = builder.thinkingMode;
        this.thinkingModeProcessor = new ThinkingModeProcessor(builder.thinkingMode);
        this.hookManager = builder.hooks != null && !builder.hooks.isEmpty()
                ? new HookManager(builder.hooks)
                : HookManager.EMPTY;

        this.llmInvoker = new LlmInvoker(builder.chatModel, builder.streamingChatModel,
                builder.maxRetries, List.of(), null, null);

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

        public ChatExecutor build() {
            if (chatModel == null && streamingChatModel == null) {
                throw new IllegalArgumentException("chatModel / streamingChatModel 至少提供一个");
            }
            return new ChatExecutor(this);
        }
    }

    // ==================== 同步调用 ====================

    /**
     * 非流式执行单轮对话。REASONING_CONTENT 模式下内部走流式收集思考内容。
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

        scheduleRound(messages, sink, params, execCtx, 0);

        StringBuilder answer = new StringBuilder();
        StringBuilder think = new StringBuilder();
        String[] errorHolder = {null};

        sink.asFlux()
                .doOnNext(event -> {
                    switch (event) {
                        case AgentStreamEvent.Text t -> answer.append(t.content());
                        case AgentStreamEvent.Thinking t -> think.append(t.content());
                        case AgentStreamEvent.Error e -> errorHolder[0] = e.message();
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
     * 流式执行单轮对话，返回 AgentStreamEvent 流。
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

        scheduleRound(messages, sink, params, execCtx, 0);

        return wrapStreamFlux(sink, conversationId, messages, execCtx);
    }

    // ==================== 内部执行 ====================

    private Disposable scheduleRound(List<ChatMessage> messages, Sinks.Many<AgentStreamEvent> sink,
                                     RunnableParams params, AgentRuntimeContext execCtx, int retryAttempt) {
        String conversationId = params != null ? params.getConversationId() : null;
        log.debug("Scheduling chat round: conversationId={}, retryAttempt={}", conversationId, retryAttempt);

        RoundState roundState = new RoundState();
        roundState.requestJson = serializeRequestJson(messages);
        long startTime = System.currentTimeMillis();

        if (!hookManager.isEmpty()) {
            hookManager.fireEvent(new BeforeReasoningEvent(execCtx, 1));
        }

        ChatRequest request = ChatRequest.builder()
                .messages(messages)
                .build();

        Disposable disposable = llmInvoker.stream(request)
                .publishOn(Schedulers.boundedElastic())
                .doOnNext(chunk -> processChunk(chunk, roundState, sink))
                .doOnComplete(() -> finishRound(messages, sink, roundState, params, execCtx,
                        System.currentTimeMillis() - startTime))
                .onErrorResume(err -> {
                    if (!hookManager.isEmpty()) {
                        hookManager.fireEvent(new ErrorEvent(execCtx, err, "chat",
                                retryAttempt, retryAttempt < llmInvoker.getMaxRetries()));
                    }
                    return llmInvoker.handleStreamError(err, retryAttempt, sink,
                            () -> scheduleRound(messages, sink, params, execCtx, retryAttempt + 1),
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
        return disposable;
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
        // ToolCalls 分片不会出现：单轮对话不注册工具规格
    }

    private void finishRound(List<ChatMessage> messages, Sinks.Many<AgentStreamEvent> sink,
                             RoundState state, RunnableParams params,
                             AgentRuntimeContext execCtx, long durationMs) {
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

        // 记录 trace（单轮对话固定 round=1）
        sessionPersister.recordTrace(execCtx, 1, state.requestJson, answer,
                roundThink, state.promptTokens, state.completionTokens, durationMs);

        if (!hookManager.isEmpty()) {
            hookManager.fireEvent(new AfterReasoningEvent(execCtx, answer, List.of(), 1,
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
                    // 异常终止兜底：正常完成路径已在 finishRound 显式落库，CAS 保证不双写
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
