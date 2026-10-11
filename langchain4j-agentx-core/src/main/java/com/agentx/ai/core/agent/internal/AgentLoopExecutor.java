package com.agentx.ai.core.agent.internal;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.agentx.ai.core.exception.AgentErrorCode;
import com.agentx.ai.core.exception.AgentException;
import com.agentx.ai.core.interrupt.PauseStateStore;
import com.agentx.ai.core.interrupt.SafePoint;
import com.agentx.ai.core.interrupt.PauseReason;
import com.agentx.ai.core.model.AgentResult;
import com.agentx.ai.core.model.AgentStreamEvent;
import com.agentx.ai.core.model.AgentTool;
import com.agentx.ai.core.model.PendingToolCall;
import com.agentx.ai.core.model.PauseState;
import com.agentx.ai.core.prompt.PromptConstants;
import com.agentx.ai.core.model.RunnableParams;
import com.agentx.ai.core.model.ThinkingMode;
import com.agentx.ai.core.hook.AgentHook;
import com.agentx.ai.core.hook.HookManager;
import com.agentx.ai.core.hook.AfterCallEvent;
import com.agentx.ai.core.hook.AfterReasoningEvent;
import com.agentx.ai.core.hook.BeforeCallEvent;
import com.agentx.ai.core.hook.BeforeReasoningEvent;
import com.agentx.ai.core.hook.ErrorEvent;
import com.agentx.ai.core.stage.AgentRuntimeContext;
import com.agentx.ai.core.utils.JsonRepairUtil;

import com.agentx.ai.core.memory.LongTermMemoryManager;
import com.agentx.ai.core.memory.store.ConversationStore;
import com.agentx.ai.core.memory.store.SessionMessageStore;
import com.agentx.ai.core.tools.toolsearch.DeferredToolRegistry;
import com.agentx.ai.core.memory.util.MemoryInjector;
import com.agentx.ai.core.memory.util.MemoryPersistor;
import com.agentx.ai.core.trace.TraceStore;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
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
import reactor.core.publisher.Sinks.EmitResult;
import reactor.core.publisher.Sinks;
import reactor.core.scheduler.Schedulers;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Agent ReAct 循环执行器：多轮迭代调用 LLM、执行工具，直到产出最终答案或达到上限。
 * 工具调用由本类控制（LC4j low-level API 天然不自动执行工具）。
 *
 * <p>2.0 变更：
 * <ul>
 *   <li>直调 LC4j low-level {@link ChatModel}/{@link StreamingChatModel}，ChatClient/Advisor 链移除</li>
 *   <li>HITL 拦截从 PauseAdvisor 下沉为引擎逻辑：{@code interceptTools} 命中即暂停等待外部输入</li>
 *   <li>每轮按 deferred 工具发现状态重建 {@link ChatRequest#toolSpecifications()}</li>
 *   <li>LC4j 流结束时由最终 AiMessage 提供完整 toolExecutionRequests，无需逐分片合并</li>
 * </ul>
 *
 * @author bigchui
 */
public class AgentLoopExecutor {

    private static final Logger log = LoggerFactory.getLogger(AgentLoopExecutor.class);

    /**
     * 最大循环轮次。
     */
    private final int maxRounds;
    private final int maxRetries;
    private final Map<String, AgentTool> toolMap;
    private final String askUserToolName;
    /**
     * HITL 拦截工具名单（下沉自 1.x PauseAdvisor）。命中任一名称则暂停等待外部输入。
     */
    private final Set<String> interceptToolNames;
    private final AgentTaskManager taskManager;
    private final MemoryInjector memoryInjector;
    private final LoopMessageBuilder messageBuilder;
    private final HookManager hookManager;
    private final ThinkingMode thinkingMode;
    private final ThinkingModeProcessor thinkingModeProcessor;
    private final ToolCallExecutor toolCallExecutor;
    private final LlmInvoker llmInvoker;
    private final ChatModel chatModel;
    /**
     * 会话/Trace/暂停状态的持久化入口，集中所有落库副作用。
     */
    private final SessionPersister sessionPersister;

    /**
     * 每次执行（stream/call）独立创建，跟踪当前轮阶段以支持用户主动中断。
     */
    private InterruptContext interruptContext;

    private AgentLoopExecutor(Builder builder) {
        this.maxRounds = builder.maxRounds;
        this.maxRetries = builder.maxRetries;
        this.askUserToolName = builder.askUserToolName;
        this.interceptToolNames = builder.interceptToolNames != null
                ? Set.copyOf(builder.interceptToolNames) : Set.of();
        this.taskManager = builder.taskManager;
        this.thinkingMode = builder.thinkingMode;
        this.thinkingModeProcessor = new ThinkingModeProcessor(builder.thinkingMode);
        this.chatModel = builder.chatModel;

        DeferredToolRegistry.Session deferredToolSession = builder.deferredToolRegistry != null
                ? builder.deferredToolRegistry.createSession()
                : null;
        List<LlmInvoker.AgentToolHolder> alwaysLoadTools = new ArrayList<>();
        this.hookManager = builder.hooks != null && !builder.hooks.isEmpty()
                ? new HookManager(builder.hooks)
                : HookManager.EMPTY;

        // 构建 tool lookup Map（包含所有工具：alwaysLoad + deferred）
        Map<String, AgentTool> map = new HashMap<>();
        if (builder.tools != null) {
            for (AgentTool t : builder.tools) {
                map.put(t.getToolDefinition().name(), t);
                alwaysLoadTools.add(new LlmInvoker.AgentToolHolder(t.getToolDefinition(), t));
            }
        }
        if (builder.deferredToolRegistry != null) {
            for (Map.Entry<String, AgentTool> entry : builder.deferredToolRegistry.getAllDeferredTools().entrySet()) {
                map.put(entry.getKey(), entry.getValue());
            }
            AgentTool searchCallback = deferredToolSession.getToolSearchCallback();
            map.put(searchCallback.getToolDefinition().name(), searchCallback);
        }
        this.toolMap = map;

        // LLM 调用器
        this.llmInvoker = new LlmInvoker(builder.chatModel, builder.streamingChatModel,
                builder.maxRetries, alwaysLoadTools,
                builder.deferredToolRegistry, deferredToolSession);

        // 记忆注入器（对话开始时加载）
        this.memoryInjector = new MemoryInjector(builder.longTermMemoryManager);

        // 记忆持久化器（委托给 SessionPersister 使用）
        MemoryPersistor memoryPersistor = new MemoryPersistor(builder.longTermMemoryManager);

        // 会话/Trace/暂停状态 统一持久化入口
        this.sessionPersister = new SessionPersister(
                builder.enableSession, builder.enableTrace,
                builder.conversationStore, builder.sessionMessageStore,
                builder.traceStore, builder.stateStore, memoryPersistor);

        // 消息构建器
        boolean hasTodoWrite = map.containsKey("TodoWrite");
        this.messageBuilder = new LoopMessageBuilder(
                builder.instructions, builder.sessionMessageStore,
                memoryInjector, builder.thinkingMode, builder.deferredToolRegistry,
                hasTodoWrite, builder.enableSession);

        // 工具调用执行器
        this.toolCallExecutor = new ToolCallExecutor(map, new ObjectMapper(),
                builder.askUserToolName, hookManager);
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private int maxRounds = 100;
        private List<AgentTool> tools;
        private AgentTaskManager taskManager;
        private SessionMessageStore sessionMessageStore;
        private ConversationStore conversationStore;
        private String instructions;
        private ChatModel chatModel;
        private StreamingChatModel streamingChatModel;
        private LongTermMemoryManager longTermMemoryManager;
        private boolean enableSession = true;
        private boolean enableTrace = true;
        private String askUserToolName;
        private Set<String> interceptToolNames;
        private List<AgentHook> hooks;
        private ThinkingMode thinkingMode = ThinkingMode.DISABLED;
        private int maxRetries = 3;
        private DeferredToolRegistry deferredToolRegistry;
        private TraceStore traceStore;
        private PauseStateStore stateStore;

        public Builder maxRounds(int v) {
            this.maxRounds = v;
            return this;
        }

        public Builder tools(List<AgentTool> v) {
            this.tools = v;
            return this;
        }

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

        public Builder longTermMemoryManager(LongTermMemoryManager v) {
            this.longTermMemoryManager = v;
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

        public Builder askUserToolName(String v) {
            this.askUserToolName = v;
            return this;
        }

        public Builder interceptToolNames(Set<String> v) {
            this.interceptToolNames = v;
            return this;
        }

        public Builder hooks(List<AgentHook> v) {
            this.hooks = v;
            return this;
        }

        public Builder thinkingMode(ThinkingMode v) {
            this.thinkingMode = v;
            return this;
        }

        public Builder maxRetries(int v) {
            this.maxRetries = v;
            return this;
        }

        public Builder deferredToolRegistry(DeferredToolRegistry v) {
            this.deferredToolRegistry = v;
            return this;
        }

        public Builder traceStore(TraceStore v) {
            this.traceStore = v;
            return this;
        }

        public Builder stateStore(PauseStateStore v) {
            this.stateStore = v;
            return this;
        }

        public AgentLoopExecutor build() {
            Objects.requireNonNull(chatModel, "chatModel must not be null");
            return new AgentLoopExecutor(this);
        }
    }

    /**
     * 非流式执行 ReAct 循环。REASONING_CONTENT 模式下内部走流式收集思考内容。
     */
    public AgentResult call(String query, RunnableParams params) {
        String conversationId = params != null ? params.getConversationId() : null;
        if (taskManager != null && conversationId != null) {
            if (taskManager.registerTask(conversationId, null) == null) {
                return new AgentResult.Failed("该会话正在执行中，请稍后再试: " + conversationId, AgentErrorCode.CONCURRENT_EXECUTION);
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
     * REASONING_CONTENT 模式内部走流式收集，blockLast 后同步落库避免 doFinally 时序问题。
     */
    private AgentResult callViaStreamForResult(String query, RunnableParams params) {
        BuiltMessages built = messageBuilder.buildInitialMessages(query, params);
        List<ChatMessage> messages = built.messages();
        Sinks.Many<AgentStreamEvent> sink = Sinks.many().unicast().onBackpressureBuffer();
        AgentRuntimeContext execCtx = new AgentRuntimeContext(query, params);
        execCtx.setNewMsgStartIndex(built.newMsgStartIndex());
        execCtx.setOriginalMessagesSnapshot(new ArrayList<>(messages));
        sessionPersister.initSession(execCtx, params, query);
        AtomicLong roundCounter = new AtomicLong(0);

        if (!hookManager.isEmpty()) {
            hookManager.fireEvent(new BeforeCallEvent(execCtx));
        }

        registerInterruptContext(messages, sink, params, query, execCtx, roundCounter);

        return blockForResult(messages, sink, roundCounter, params, execCtx, query);
    }

    /**
     * sink 阻塞收集：scheduleRound → doOnNext 累积 → blockLast → AgentResult，供 call/resume 共用。
     */
    private AgentResult blockForResult(List<ChatMessage> messages,
                                       Sinks.Many<AgentStreamEvent> sink,
                                       AtomicLong roundCounter,
                                       RunnableParams params,
                                       AgentRuntimeContext execCtx,
                                       String query) {
        scheduleRound(messages, sink, roundCounter, params, execCtx, query);

        StringBuilder answer = new StringBuilder();
        StringBuilder think = new StringBuilder();
        PauseState[] pauseHolder = {null};
        String[] errorHolder = {null};

        sink.asFlux()
                .doOnNext(event -> {
                    switch (event) {
                        case AgentStreamEvent.Text t -> answer.append(t.content());
                        case AgentStreamEvent.Thinking t -> think.append(t.content());
                        case AgentStreamEvent.Paused p -> pauseHolder[0] = p.state();
                        case AgentStreamEvent.Error e -> errorHolder[0] = e.message();
                        // 工具调用轮清空 answer 和 think，只保留最后一轮（与 AgentScope 一致）
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
            execCtx.markTerminal("error");
            sessionPersister.persistOnTerminal(execCtx, messages, SignalType.ON_ERROR);
            fireAfterCallOnce(execCtx, null, 0);
            return new AgentResult.Failed(errorHolder[0], AgentErrorCode.LLM_CALL_FAILED);
        }
        if (pauseHolder[0] != null) {
            execCtx.markTerminal("interrupted");
            sessionPersister.persistOnTerminal(execCtx, messages, SignalType.CANCEL);
            fireAfterCallOnce(execCtx, null, 0);
            return new AgentResult.Paused(pauseHolder[0]);
        }

        String finalAnswer = answer.toString();
        if (params != null && params.getOutputType() != null) {
            finalAnswer = JsonRepairUtil.fixJson(finalAnswer);
        }

        execCtx.markTerminal("completed");
        sessionPersister.persistOnTerminal(execCtx, messages, SignalType.ON_COMPLETE);
        fireAfterCallOnce(execCtx, null, 0);

        return new AgentResult.Completed(
                finalAnswer,
                think.length() > 0 ? think.toString() : null);
    }

    /**
     * 从暂停状态恢复执行（非流式）。
     */
    public AgentResult resume(PauseState state, Map<String, String> toolResults) {
        List<ChatMessage> messages = buildResumeMessages(state, toolResults, null);

        String conversationId = state.getParams() != null ? state.getParams().getConversationId() : null;
        if (taskManager != null && conversationId != null) {
            if (taskManager.registerTask(conversationId, null) == null) {
                return new AgentResult.Failed("该会话正在执行中，请稍后再试: " + conversationId,
                        AgentErrorCode.CONCURRENT_EXECUTION);
            }
        }
        try {
            AgentResult result = resumeViaStreamForResult(messages, state, null);
            if (result instanceof AgentResult.Completed) {
                sessionPersister.deletePauseState(conversationId);
            }
            return result;
        } finally {
            if (taskManager != null && conversationId != null) {
                taskManager.removeTask(conversationId);
            }
        }
    }

    /**
     * 流式-backed 恢复执行，委托 blockForResult。
     */
    private AgentResult resumeViaStreamForResult(List<ChatMessage> messages, PauseState state,
                                                 String resumeQuery) {
        Sinks.Many<AgentStreamEvent> sink = Sinks.many().unicast().onBackpressureBuffer();
        AtomicLong roundCounter = new AtomicLong(state.getCurrentRound());
        AgentRuntimeContext execCtx = new AgentRuntimeContext(state.getQuery(), state.getParams());
        // 暂停时已落库 [query .. assistant(tool_calls)]，本次恢复只持久化新增的 tool 响应及后续消息
        execCtx.setNewMsgStartIndex(state.getMessages().size());
        execCtx.setOriginalMessagesSnapshot(new ArrayList<>(messages));
        String effectiveQuery = sessionPersister.initSessionFromResume(execCtx, state, resumeQuery);

        if (!hookManager.isEmpty()) {
            hookManager.fireEvent(new BeforeCallEvent(execCtx));
        }

        registerInterruptContext(messages, sink, state.getParams(), effectiveQuery, execCtx, roundCounter);

        return blockForResult(messages, sink, roundCounter, state.getParams(), execCtx, effectiveQuery);
    }

    /**
     * 按暂停原因分发工具结果解析：USER_INTERRUPT+TOOL_EXECUTION 走占位/重跑，其余走默认。
     */
    private String resolveToolResult(PendingToolCall ptc,
                                     ToolExecutionRequest toolCall,
                                     Map<String, String> toolResults,
                                     PauseState state,
                                     String resumeQuery) {
        if (state.getReason() == PauseReason.USER_INTERRUPT
                && state.getSafePoint() == SafePoint.TOOL_EXECUTION) {
            // 用户中断后带着新消息恢复 → 不重跑工具，注入占位让 LLM 根据新指令决策
            if (resumeQuery != null && !resumeQuery.isBlank()) {
                return PromptConstants.buildInterruptToolSkippedMessage(ptc.name(), ptc.arguments());
            }
            return toolCallExecutor.resolveInterruptToolResult(ptc, toolCall, state.getParams());
        }
        return toolCallExecutor.resolveResumeToolResult(ptc, toolCall, toolResults, state.getParams());
    }

    /**
     * 从 PauseState 重建消息列表，注入 pending tool calls 的工具结果消息。
     */
    private List<ChatMessage> buildResumeMessages(PauseState state, Map<String, String> toolResults,
                                                  String resumeQuery) {
        List<ChatMessage> messages = new ArrayList<>(state.getMessages());
        for (PendingToolCall ptc : state.getPendingToolCalls()) {
            ToolExecutionRequest toolCall = ToolExecutionRequest.builder()
                    .id(ptc.id())
                    .name(ptc.name())
                    .arguments(ptc.arguments())
                    .build();
            String result = resolveToolResult(ptc, toolCall, toolResults, state, resumeQuery);
            toolCallExecutor.addNormalToolMessage(toolCall, result, messages);
        }
        if (resumeQuery != null && !resumeQuery.isBlank()) {
            messages.add(UserMessage.from(
                    PromptConstants.buildResumeInterruptMessage(state.getQuery(), resumeQuery)));
        }
        return messages;
    }

    /**
     * 流式执行 ReAct 循环，返回 AgentStreamEvent 流。
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

        registerInterruptContext(messages, sink, params, query, execCtx, roundCounter);

        scheduleRound(messages, sink, roundCounter, params, execCtx, query);

        return wrapStreamFlux(sink, conversationId, messages, execCtx);
    }

    /**
     * 从暂停状态恢复流式执行。
     */
    public Flux<AgentStreamEvent> resumeStream(PauseState state, Map<String, String> toolResults) {
        return resumeStream(state, toolResults, null);
    }

    /**
     * 从暂停状态恢复流式执行，可附带用户中断后补充的新消息。
     */
    public Flux<AgentStreamEvent> resumeStream(PauseState state, Map<String, String> toolResults,
                                               String resumeQuery) {
        List<ChatMessage> messages = buildResumeMessages(state, toolResults, resumeQuery);

        String conversationId = state.getParams() != null ? state.getParams().getConversationId() : null;

        Sinks.Many<AgentStreamEvent> sink = Sinks.many().unicast().onBackpressureBuffer();

        if (taskManager != null && conversationId != null) {
            // 注册真正的 sink（非 null），让 stopTask 的 tryEmitComplete 能正常结束下游 Flux
            AgentTaskManager.TaskInfo taskInfo = taskManager.registerTask(conversationId, sink);
            if (taskInfo == null) {
                return Flux.error(new AgentException(AgentErrorCode.CONCURRENT_EXECUTION,
                        "该会话正在执行中，请稍后再试: " + conversationId));
            }
        }

        AtomicLong roundCounter = new AtomicLong(state.getCurrentRound());
        AgentRuntimeContext execCtx = new AgentRuntimeContext(state.getQuery(), state.getParams());
        execCtx.setEmitter(sink::tryEmitNext);
        execCtx.setMessages(messages);
        // 暂停时已落库 [query .. assistant(tool_calls)]，本次恢复只持久化新增的 tool 响应及后续消息
        execCtx.setNewMsgStartIndex(state.getMessages().size());
        execCtx.setOriginalMessagesSnapshot(new ArrayList<>(messages));
        String effectiveQuery = sessionPersister.initSessionFromResume(execCtx, state, resumeQuery);

        if (!hookManager.isEmpty()) {
            hookManager.fireEvent(new BeforeCallEvent(execCtx));
        }

        registerInterruptContext(messages, sink, state.getParams(), effectiveQuery, execCtx, roundCounter);

        // disposable 的注册由 scheduleRound 内部每轮刷新负责，此处不再手动 setDisposable
        scheduleRound(messages, sink, roundCounter, state.getParams(), execCtx, effectiveQuery);

        // 正常完成时清理快照（Paused 不能删，会通过 UPSERT 覆盖）
        var wasPaused = new AtomicBoolean(false);
        return wrapStreamFlux(sink, conversationId, messages, execCtx)
                .doOnNext(event -> {
                    if (event instanceof AgentStreamEvent.Paused) {
                        wasPaused.set(true);
                    }
                })
                .doOnComplete(() -> {
                    if (!wasPaused.get()) {
                        sessionPersister.deletePauseState(conversationId);
                    }
                });
    }

    /**
     * 流式 Flux 统一加错误处理、终态落库兜底和任务清理。
     */
    private Flux<AgentStreamEvent> wrapStreamFlux(Sinks.Many<AgentStreamEvent> sink,
                                                  String conversationId,
                                                  List<ChatMessage> messages,
                                                  AgentRuntimeContext execCtx) {
        return sink.asFlux()
                .doOnError(err -> handleStreamError(conversationId, err))
                .doFinally(signal -> {
                    log.debug("Stream terminated: conversationId={}, signal={}", conversationId, signal);
                    // 异常终止兜底：不信任 Reactor signal，统一按 CANCEL 处理（自然完成已显式落库）
                    SignalType fallbackSignal = signal == SignalType.ON_ERROR ? SignalType.ON_ERROR : SignalType.CANCEL;
                    sessionPersister.persistOnTerminal(execCtx, messages, fallbackSignal);
                    // 确保沙箱在所有终止路径（暂停/中断/错误/取消/断连）上释放
                    fireAfterCallOnce(execCtx, null, 0);
                    if (taskManager != null && conversationId != null) {
                        taskManager.stopTask(conversationId);
                    }
                });
    }

    private void handleStreamError(String conversationId, Throwable err) {
        log.error("\n\n Stream error: conversationId={}", conversationId, err);
    }

    /**
     * 幂等触发 AfterCallEvent。保证一次调用中恰好触发一次，
     * 覆盖正常完成、HITL 暂停、用户中断、LLM 错误、stopTask、客户端断连等所有终止路径。
     * SandboxHook 依赖此事件释放容器资源。
     */
    private void fireAfterCallOnce(AgentRuntimeContext execCtx, String answer, long durationMs) {
        if (!hookManager.isEmpty() && execCtx.tryFireAfterCall()) {
            hookManager.fireEvent(new AfterCallEvent(execCtx, answer, durationMs));
        }
    }

    /**
     * 注册中断上下文和处理器，供 stream / resumeStream / callViaStreamForResult / resumeViaStreamForResult 共用。
     */
    private void registerInterruptContext(List<ChatMessage> messages, Sinks.Many<AgentStreamEvent> sink,
                                          RunnableParams params, String query,
                                          AgentRuntimeContext execCtx, AtomicLong roundCounter) {
        String conversationId = params != null ? params.getConversationId() : null;
        this.interruptContext = new InterruptContext(messages, sink, params, query);
        if (taskManager != null && conversationId != null) {
            taskManager.setInterruptHandler(conversationId, msg -> {
                // 中断前把本轮部分回复刷进 messages，纯文本（无工具调用）回复也能落库
                interruptContext.flushPartialOutput(execCtx);
                PauseState snapshot = interruptContext.buildSnapshot(msg, execCtx, roundCounter.get());
                sessionPersister.persistPauseState(snapshot);
                // 标记中断态并显式落库后再发射 Paused + complete sink，避免 doFinally 时序竞态
                execCtx.markTerminal("interrupted");
                sessionPersister.persistOnTerminal(execCtx, messages, SignalType.CANCEL);
                interruptContext.emitPausedAndComplete(snapshot);
            });
        }
    }

    private Disposable scheduleRound(List<ChatMessage> messages, Sinks.Many<AgentStreamEvent> sink,
                                     AtomicLong roundCounter, RunnableParams params,
                                     AgentRuntimeContext execCtx, String query) {
        return scheduleRound(messages, sink, roundCounter, params, execCtx, query, 0);
    }

    private Disposable scheduleRound(List<ChatMessage> messages, Sinks.Many<AgentStreamEvent> sink,
                                     AtomicLong roundCounter, RunnableParams params,
                                     AgentRuntimeContext execCtx, String query, int retryAttempt) {
        long round = roundCounter.incrementAndGet();
        String conversationId = params != null ? params.getConversationId() : null;
        log.debug("Scheduling round: {}, conversationId={}, retryAttempt={}", round, conversationId, retryAttempt);

        // 进入 LLM_STREAMING 阶段，可被中断
        if (interruptContext != null) {
            interruptContext.enterLlmStreaming();
        }

        RoundState roundState = new RoundState();
        // 序列化本轮请求 JSON（trace 用；1.x 由 RequestLoggingAdvisor 通过 advisor context 传递）
        roundState.requestJson = serializeRequestJson(messages);

        // 注册当前轮缓冲区，使中断回调能读取部分输出并持久化
        if (interruptContext != null) {
            interruptContext.setRoundBuffers(roundState.textBuffer, roundState.reasoningBuffer);
        }
        long startTime = System.currentTimeMillis();

        if (!hookManager.isEmpty()) {
            hookManager.fireEvent(new BeforeReasoningEvent(execCtx, round));
        }

        ChatRequest request = ChatRequest.builder()
                .messages(messages)
                .toolSpecifications(llmInvoker.buildRoundToolSpecifications())
                .build();

        Disposable disposable = llmInvoker.stream(request)
                .publishOn(Schedulers.boundedElastic())
                .doOnNext(chunk -> processChunk(chunk, sink, roundState, execCtx))
                .doOnComplete(() -> {
                    long durationMs = System.currentTimeMillis() - startTime;
                    if (!hookManager.isEmpty()) {
                        hookManager.fireEvent(new AfterReasoningEvent(
                                execCtx, roundState.textBuffer.toString(),
                                roundState.toolCalls, round,
                                roundState.promptTokens, roundState.completionTokens, durationMs));
                    }
                    finishRound(messages, sink, roundState, roundCounter, params, execCtx, query, durationMs);
                })
                .onErrorResume(err -> {
                    if (!hookManager.isEmpty()) {
                        hookManager.fireEvent(new ErrorEvent(
                                execCtx, err, "reasoning", retryAttempt, retryAttempt < maxRetries));
                    }
                    llmInvoker.handleStreamError(err, retryAttempt, sink,
                            () -> scheduleRound(messages, sink, roundCounter, params, execCtx, query, retryAttempt + 1),
                            "LLM stream error",
                            () -> {
                                execCtx.markTerminal("error");
                                sessionPersister.persistOnTerminal(execCtx, messages, SignalType.ON_ERROR);
                            });
                    return Flux.empty();
                })
                .subscribe();

        // 每轮刷新 taskManager 的 disposable，保证 stopTask 能中断当前在飞的轮次
        if (taskManager != null && conversationId != null) {
            taskManager.setDisposable(conversationId, disposable);
        }
        return disposable;
    }

    /**
     * 序列化请求消息为 JSON（trace 用，对齐 1.x RequestLoggingAdvisor 的 LLM_REQUEST_JSON）。
     */
    private String serializeRequestJson(List<ChatMessage> messages) {
        try {
            return com.agentx.ai.core.utils.MessageJsonSerializer.toJson(messages);
        } catch (Exception e) {
            return null;
        }
    }

    private void processChunk(LlmInvoker.LlmChunk chunk, Sinks.Many<AgentStreamEvent> sink,
                              RoundState state, AgentRuntimeContext execCtx) {
        if (chunk instanceof LlmInvoker.LlmChunk.Text t) {
            state.mode = RoundMode.TEXT;
            String text = t.token();
            thinkingModeProcessor.processStreamChunk(text, state, sink);
        } else if (chunk instanceof LlmInvoker.LlmChunk.Thinking th) {
            // 独立思考通道（reasoning_content / thinking）：REASONING_CONTENT 模式才累积
            if (thinkingMode == ThinkingMode.REASONING_CONTENT) {
                thinkingModeProcessor.processReasoningChunk(th.delta(), state, sink);
            }
        } else if (chunk instanceof LlmInvoker.LlmChunk.ToolCalls tc) {
            state.mode = RoundMode.TOOL_CALL;
            state.toolCalls.addAll(tc.requests());
        } else if (chunk instanceof LlmInvoker.LlmChunk.Metadata meta) {
            ChatResponse response = meta.response();
            if (response != null) {
                TokenUsage usage = response.tokenUsage();
                if (usage != null) {
                    state.promptTokens = usage.inputTokenCount() != null ? usage.inputTokenCount() : 0;
                    state.completionTokens = usage.outputTokenCount() != null ? usage.outputTokenCount() : 0;
                }
                if (response.finishReason() != null) {
                    state.finishReason = response.finishReason().name();
                }
            }
        }
    }

    private void finishRound(List<ChatMessage> messages, Sinks.Many<AgentStreamEvent> sink,
                             RoundState state, AtomicLong roundCounter, RunnableParams params,
                             AgentRuntimeContext execCtx, String query, long durationMs) {
        String conversationId = params != null ? params.getConversationId() : null;
        int round = (int) roundCounter.get();
        String requestJson = state.requestJson;

        if (state.promptTokens >= 0 || state.finishReason != null) {
            log.debug("LLM response detail: conversationId={}, promptTokens={}, completionTokens={}, finishReason={}",
                    conversationId, state.promptTokens, state.completionTokens, state.finishReason);
        }

        if (state.toolCalls.isEmpty()) {
            log.debug("No tool calls detected, stream completed: conversationId={}", conversationId);
            // 将累积的文本作为 AiMessage 添加到 messages
            if (!state.textBuffer.isEmpty()) {
                AiMessage.Builder aiBuilder = AiMessage.builder().text(state.textBuffer.toString());
                if (!state.reasoningBuffer.isEmpty()) {
                    aiBuilder.thinking(state.reasoningBuffer.toString());
                }
                AiMessage finalAssistant = aiBuilder.build();
                messages.add(finalAssistant);
                execCtx.appendOriginalMessage(finalAssistant);
            }

            // 记录 trace（最终答案轮，trace 保持单轮 think）
            String roundThink = state.reasoningBuffer.length() > 0 ? state.reasoningBuffer.toString() : null;
            sessionPersister.recordTrace(execCtx, round, requestJson, state.textBuffer.toString(),
                    roundThink, state.promptTokens, state.completionTokens, durationMs);

            // 标记终态：显式落库后再 complete sink，避免 doFinally 时序竞态导致 blockLast 提前返回
            execCtx.markTerminal("completed");
            sessionPersister.persistOnTerminal(execCtx, messages, SignalType.ON_COMPLETE);
            // AFTER_CALL hook（终态已标记、已持久化）
            fireAfterCallOnce(execCtx, state.textBuffer.toString(), durationMs);
            EmitResult completeResult = sink.tryEmitNext(new AgentStreamEvent.Complete(
                    execCtx.getTotalPromptTokens(), execCtx.getTotalCompletionTokens(),
                    conversationId, execCtx.getSessionId(), null));
            log.debug("tryEmitNext(Complete) result={}, conversationId={}", completeResult, conversationId);
            EmitResult emitResult = sink.tryEmitComplete();
            log.debug("tryEmitComplete result={}, conversationId={}", emitResult, conversationId);
            return;
        }

        // 校验并修复不合法的 tool call arguments，防止后续 API 调用 400
        List<ToolExecutionRequest> safeToolCalls = toolCallExecutor.sanitizeToolCalls(state.toolCalls);

        // 先标记工具阶段并缓存工具清单：tool_calls 入链后任意时刻中断，快照都能表达待执行状态
        if (interruptContext != null) {
            interruptContext.enterToolExecution(safeToolCalls);
        }

        // tool call 路径：构建 AiMessage 携带思考内容（thinking 字段）
        AiMessage.Builder assistantBuilder = AiMessage.builder()
                .text(state.textBuffer.isEmpty() ? "" : state.textBuffer.toString())
                .toolExecutionRequests(safeToolCalls);
        if (!state.reasoningBuffer.isEmpty()) {
            assistantBuilder.thinking(state.reasoningBuffer.toString());
        }
        AiMessage assistantMsg = assistantBuilder.build();
        messages.add(assistantMsg);
        execCtx.appendOriginalMessage(assistantMsg);

        if (maxRounds > 0 && roundCounter.get() >= maxRounds) {
            log.debug("Max rounds reached, forcing final answer: conversationId={}", conversationId);
            // 记录 trace（工具调用轮，达到上限）
            sessionPersister.recordTrace(execCtx, round, requestJson, sessionPersister.serializeToolCalls(safeToolCalls), null,
                    state.promptTokens, state.completionTokens, durationMs);
            forceFinalStream(messages, sink, params, execCtx, query);
            return;
        }

        // === 流式暂停检查（HITL 拦截下沉为引擎逻辑，替代 1.x PauseAdvisor） ===
        List<PendingToolCall> pending = findInterceptedTools(safeToolCalls);
        if (!pending.isEmpty()) {
            // 执行非拦截工具
            toolCallExecutor.executeNonPendingTools(safeToolCalls, pending, messages, params, execCtx);

            PauseState pauseState = PauseState.builder()
                    .messages(List.copyOf(messages))
                    .currentRound((int) roundCounter.get())
                    .pendingToolCalls(pending)
                    .params(params)
                    .query(query)
                    .sessionId(execCtx.getSessionId())
                    .totalPromptTokens(execCtx.getTotalPromptTokens())
                    .totalCompletionTokens(execCtx.getTotalCompletionTokens())
                    .reason(PauseReason.HITL_TOOL_REQUEST)
                    .interruptedAt(System.currentTimeMillis())
                    .build();

            sessionPersister.persistPauseState(pauseState);

            // 记录 trace（暂停前）
            sessionPersister.recordTrace(execCtx, round, requestJson, sessionPersister.serializeToolCalls(safeToolCalls), null,
                    state.promptTokens, state.completionTokens, durationMs);

            log.debug("Stream paused at round {}, pending tools: {}", roundCounter.get(), pending.size());
            // 标记中断态并显式落库后再 complete sink，避免 doFinally 时序竞态
            execCtx.markTerminal("interrupted");
            sessionPersister.persistOnTerminal(execCtx, messages, SignalType.CANCEL);
            sink.tryEmitNext(new AgentStreamEvent.Paused(pauseState));
            sink.tryEmitComplete();
            return;
        }

        // 记录 trace（工具调用轮）
        sessionPersister.recordTrace(execCtx, round, requestJson, sessionPersister.serializeToolCalls(safeToolCalls), null,
                state.promptTokens, state.completionTokens, durationMs);

        toolCallExecutor.executeToolCallsAsync(sink, safeToolCalls, messages, params, execCtx, () -> {
            scheduleRound(messages, sink, roundCounter, params, execCtx, query);
        });
    }

    /**
     * 从本轮 tool calls 中筛选出被 HITL 拦截的工具（下沉自 PauseAdvisor.shouldIntercept）。
     */
    private List<PendingToolCall> findInterceptedTools(List<ToolExecutionRequest> toolCalls) {
        if (interceptToolNames.isEmpty() || toolCalls.isEmpty()) {
            return List.of();
        }
        List<PendingToolCall> pending = new ArrayList<>();
        for (ToolExecutionRequest tc : toolCalls) {
            if (interceptToolNames.contains(tc.name())) {
                pending.add(new PendingToolCall(tc.id(), tc.name(), tc.arguments()));
            }
        }
        return pending;
    }

    private void forceFinalStream(List<ChatMessage> messages, Sinks.Many<AgentStreamEvent> sink,
                                  RunnableParams params, AgentRuntimeContext execCtx, String query) {
        forceFinalStream(messages, sink, params, execCtx, query, 0);
    }

    private void forceFinalStream(List<ChatMessage> messages, Sinks.Many<AgentStreamEvent> sink,
                                  RunnableParams params, AgentRuntimeContext execCtx, String query, int retryAttempt) {
        // 闭合最后一轮未执行的 tool_calls（直接追加到原 messages，使终态落库能完整记录）
        if (!messages.isEmpty() && messages.get(messages.size() - 1) instanceof AiMessage lastMsg) {
            if (lastMsg.hasToolExecutionRequests()) {
                for (ToolExecutionRequest tc : lastMsg.toolExecutionRequests()) {
                    toolCallExecutor.addNormalToolMessage(tc, "Agent maximum rounds reached. Tool execution skipped.", messages);
                }
            }
        }

        StringBuilder answerBuffer = new StringBuilder();
        StringBuilder reasoningBuffer = new StringBuilder();
        boolean[] inThink = {false};
        final long[] finalUsage = {0, 0};

        ChatRequest request = ChatRequest.builder()
                .messages(messages)
                .build();

        Disposable disposable = llmInvoker.stream(request)
                .publishOn(Schedulers.boundedElastic())
                .doOnNext(chunk -> {
                    if (chunk instanceof LlmInvoker.LlmChunk.Text t) {
                        String text = t.token();
                        if (thinkingMode == ThinkingMode.REASONING_CONTENT) {
                            // 文本即正文（思考走独立通道）
                            thinkingModeProcessor.processForceFinalChunk(text, inThink, answerBuffer, reasoningBuffer, sink);
                        } else {
                            thinkingModeProcessor.processForceFinalChunk(text, inThink, answerBuffer, reasoningBuffer, sink);
                        }
                    } else if (chunk instanceof LlmInvoker.LlmChunk.Thinking th) {
                        if (thinkingMode == ThinkingMode.REASONING_CONTENT && th.delta() != null && !th.delta().isEmpty()) {
                            reasoningBuffer.append(th.delta());
                            sink.tryEmitNext(new AgentStreamEvent.Thinking(th.delta()));
                        }
                    } else if (chunk instanceof LlmInvoker.LlmChunk.Metadata meta) {
                        ChatResponse response = meta.response();
                        if (response != null && response.tokenUsage() != null) {
                            TokenUsage usage = response.tokenUsage();
                            finalUsage[0] = usage.inputTokenCount() != null ? usage.inputTokenCount() : 0;
                            finalUsage[1] = usage.outputTokenCount() != null ? usage.outputTokenCount() : 0;
                        }
                    }
                })
                .doOnComplete(() -> {
                    String convId = params != null ? params.getConversationId() : null;
                    execCtx.accumulateTokens(finalUsage[0], finalUsage[1]);
                    // 将最终答案作为 AiMessage 追加到 messages，供终态落库捕获
                    if (!answerBuffer.isEmpty() || !reasoningBuffer.isEmpty()) {
                        AiMessage.Builder aiBuilder = AiMessage.builder().text(answerBuffer.toString());
                        if (!reasoningBuffer.isEmpty()) {
                            aiBuilder.thinking(reasoningBuffer.toString());
                        }
                        AiMessage finalAssistant = aiBuilder.build();
                        messages.add(finalAssistant);
                        execCtx.appendOriginalMessage(finalAssistant);
                    }
                    // 标记终态：显式落库后再 complete sink，避免 doFinally 时序竞态
                    execCtx.markTerminal("completed");
                    sessionPersister.persistOnTerminal(execCtx, messages, SignalType.ON_COMPLETE);
                    fireAfterCallOnce(execCtx, answerBuffer.toString(), 0);
                    sink.tryEmitNext(new AgentStreamEvent.Complete(
                            execCtx.getTotalPromptTokens(), execCtx.getTotalCompletionTokens(),
                            convId, execCtx.getSessionId(), null));
                    sink.tryEmitComplete();
                })
                .onErrorResume(err -> {
                    if (!hookManager.isEmpty()) {
                        hookManager.fireEvent(new ErrorEvent(
                                execCtx, err, "forceFinal", retryAttempt, retryAttempt < maxRetries));
                    }
                    return llmInvoker.handleStreamError(err, retryAttempt, sink,
                        () -> forceFinalStream(messages, sink, params, execCtx, query, retryAttempt + 1),
                        "forceFinal stream error",
                        () -> {
                            execCtx.markTerminal("error");
                            sessionPersister.persistOnTerminal(execCtx, messages, SignalType.ON_ERROR);
                        });
                })
                .subscribe();

        // 同 scheduleRound：每轮刷新 disposable，保证 stopTask 可中断当前在飞的 forceFinal 流。
        String conversationId = params != null ? params.getConversationId() : null;
        if (taskManager != null && conversationId != null) {
            taskManager.setDisposable(conversationId, disposable);
        }
    }

}
