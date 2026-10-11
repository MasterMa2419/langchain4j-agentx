package com.agentx.ai.core.agent;

import com.agentx.ai.core.context.ContextCompactor;
import com.agentx.ai.core.hook.ContextCompactionHook;
import com.agentx.ai.core.context.ContextPolicy;
import com.agentx.ai.core.context.compress.CompressionStrategy;
import com.agentx.ai.core.context.compress.LlmSummarizer;
import com.agentx.ai.core.context.compress.NoOpOffloadStore;
import com.agentx.ai.core.context.compress.OffloadStore;
import com.agentx.ai.core.context.compress.SessionBackedOffloadStore;
import com.agentx.ai.core.context.compress.strategy.HistoricalToolListStrategy;
import com.agentx.ai.core.context.compress.strategy.LargeMsgOffloadWithKeepStrategy;
import com.agentx.ai.core.context.compress.strategy.LargeMsgOffloadNoKeepStrategy;
import com.agentx.ai.core.context.compress.strategy.HistoricalRoundSummaryStrategy;
import com.agentx.ai.core.context.compress.strategy.CurrentRoundLargeMsgStrategy;
import com.agentx.ai.core.context.compress.strategy.CurrentRoundOverallStrategy;
import com.agentx.ai.core.agent.internal.AgentLoopExecutor;
import com.agentx.ai.core.agent.internal.AgentTaskManager;
import com.agentx.ai.core.interrupt.PauseStateStore;
import com.agentx.ai.core.interrupt.InMemoryPauseStateStore;
import com.agentx.ai.core.interrupt.PauseReason;
import com.agentx.ai.core.model.AgentResult;
import com.agentx.ai.core.model.AgentStreamEvent;
import com.agentx.ai.core.model.AgentTool;
import com.agentx.ai.core.model.PauseState;
import com.agentx.ai.core.model.RunnableParams;
import com.agentx.ai.core.model.ThinkingMode;
import com.agentx.ai.core.hook.AgentHook;
import com.agentx.ai.core.memory.LongTermMemoryConfig;
import com.agentx.ai.core.memory.LongTermMemoryManager;
import com.agentx.ai.core.memory.store.DataSourceStorageFactory;
import com.agentx.ai.core.memory.store.SessionMessageStore;
import com.agentx.ai.core.memory.store.ConversationStore;
import com.agentx.ai.core.sandbox.*;
import com.agentx.ai.core.sandbox.snapshot.InMemorySnapshotMetadataStore;
import com.agentx.ai.core.sandbox.snapshot.JdbcSnapshotMetadataStore;
import com.agentx.ai.core.sandbox.snapshot.LocalSnapshotStorage;
import com.agentx.ai.core.sandbox.snapshot.SnapshotMetadataStore;
import com.agentx.ai.core.sandbox.snapshot.SnapshotStorage;
import com.agentx.ai.core.tools.AskUserTool;
import com.agentx.ai.core.tools.ContextReloadTool;
import com.agentx.ai.core.trace.TraceStore;
import com.agentx.ai.core.tools.SubAgentTool;
import com.agentx.ai.core.utils.ToolMergeUtil;
import com.agentx.ai.core.chatmodels.AgentModels;
import com.agentx.ai.core.tools.toolsearch.DeferredToolRegistry;
import com.agentx.ai.core.tools.toolsearch.ToolSearchConfig;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.StreamingChatModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import reactor.core.publisher.Flux;

import javax.sql.DataSource;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.Supplier;

/**
 * ReactAgent - 基于 ReAct 范式的智能体实现。
 * <p>
 * 实现 Reasoning（推理）+ Acting（行动）循环模式，通过多轮对话完成复杂任务。
 * 支持自动多轮推理和工具调用、流式输出、会话记忆管理。
 * <p>
 * 2.0 起 AI 层基于 LangChain4j：推荐通过 {@link com.agentx.ai.core.chatmodels.AgentModels}
 * 一次构造同步 + 流式模型对（{@code .models(...)}），框架直调 low-level API 并自行接管工具执行。
 *
 * @author bigchui
 */
public class ReactAgent {

    private static final Logger log = LoggerFactory.getLogger(ReactAgent.class);

    private final String name;
    private final String description;
    private final ChatModel chatModel;
    private final StreamingChatModel streamingChatModel;
    private final int maxRounds;
    private final List<AgentTool> tools;
    private final Set<String> interceptToolNames;
    private final AgentTaskManager taskManager;
    private final String instructions;
    private final SessionMessageStore sessionMessageStore;
    private final ConversationStore conversationStore;
    private final LongTermMemoryManager longTermMemoryManager;
    private final DataSource dataSource;
    private boolean enableSession;
    private final List<AgentHook> hooks;
    private final ThinkingMode thinkingMode;
    private final int maxRetries;
    private final ContextPolicy contextPolicy;
    private final SandboxConfig sandboxConfig;
    private final SandboxManager sandboxManager;
    private final DeferredToolRegistry deferredToolRegistry;
    private TraceStore traceStore;
    private final boolean enableTrace;

    /**
     * Agent 暂停状态存储。默认内存实现，可通过 {@link Builder#stateStore(PauseStateStore)} 自定义。
     * <p>用于 {@link #interrupt(String)} 时自动持久化 PauseState，
     * 配合 {@link #hasInterruptedState(String)} / {@link #resumeStream(String)} 实现断点重连。
     */
    private final PauseStateStore stateStore;

    private ReactAgent(Builder builder, SessionMessageStore sessionMessageStore,
                       ConversationStore conversationStore,
                       LongTermMemoryManager longTermMemoryManager,
                       TraceStore traceStore,
                       PauseStateStore stateStore) {
        this.deferredToolRegistry = builder.deferredToolRegistry;

        // 开启上下文压缩且有会话存储时，注入 context_reload 工具（LLM 按需取回 offload 原文）
        if (builder.contextPolicy != null && builder.enableSession && sessionMessageStore != null) {
            builder.tools.addAll(AgentTool.fromAnnotated(new ContextReloadTool(sessionMessageStore)));
        }

        this.name = builder.name;
        this.description = builder.description;
        this.chatModel = builder.chatModel;
        this.streamingChatModel = builder.streamingChatModel;
        this.maxRounds = builder.maxRounds;
        this.tools = List.copyOf(builder.tools);
        this.interceptToolNames = Set.copyOf(builder.interceptToolNames);
        this.taskManager = builder.taskManager;
        this.instructions = builder.instructions;
        this.sessionMessageStore = sessionMessageStore;
        this.conversationStore = conversationStore;
        this.longTermMemoryManager = longTermMemoryManager;
        this.dataSource = builder.dataSource;
        this.enableSession = builder.enableSession;
        this.hooks = builder.hooks;
        this.thinkingMode = builder.thinkingMode;
        this.maxRetries = builder.maxRetries;
        this.contextPolicy = builder.contextPolicy;
        this.sandboxConfig = builder.sandboxConfig;
        this.sandboxManager = buildSandboxManager(builder);
        this.traceStore = traceStore;
        this.enableTrace = builder.enableTrace;
        this.stateStore = stateStore;
    }

    private static SandboxManager buildSandboxManager(Builder builder) {
        SandboxConfig config = builder.sandboxConfig;
        if (config == null) {
            return null;
        }
        SandboxBackend backend = config.getBackend();
        if (!backend.isAvailable()) {
            if (config.isStrictMode()) {
                throw new IllegalStateException("沙箱后端不可用（严格模式）: "
                        + backend.getClass().getSimpleName());
            }
            log.warn("[ReactAgent] 沙箱后端不可用，跳过沙箱，退化为宿主执行");
            return null;
        }
        SnapshotStorage snapshotStorage = config.getSnapshotStorage();
        if (snapshotStorage == null) {
            snapshotStorage = new LocalSnapshotStorage(config.getSnapshotDir());
            log.info("[ReactAgent] 快照存储: LocalSnapshotStorage(dir={})", config.getSnapshotDir());
        } else {
            log.info("[ReactAgent] 快照存储: {}", snapshotStorage.getClass().getSimpleName());
        }
        SnapshotMetadataStore metadataStore = config.getSnapshotMetadataStore();
        if (metadataStore == null) {
            metadataStore = builder.dataSource != null
                    ? new JdbcSnapshotMetadataStore(builder.dataSource)
                    : new InMemorySnapshotMetadataStore();
            if (metadataStore instanceof JdbcSnapshotMetadataStore jdbc) {
                jdbc.initialize();
            }
        }
        log.info("[ReactAgent] 沙箱已启用: backend={}, scope={}",
                backend.getClass().getSimpleName(), config.getIsolationScope());
        return new SandboxManager(config, backend, snapshotStorage, metadataStore);
    }

    /**
     * 创建 Builder
     */
    public static Builder builder() {
        return new Builder();
    }

    private AgentLoopExecutor createExecutor() {
        // Hook 列表：用户 Hook + 按需引入的上下文压缩 Hook（压缩 Hook 优先级最高，置列表首部）
        List<AgentHook> allHooks = new ArrayList<>(hooks != null ? hooks : List.of());

        var executorBuilder = AgentLoopExecutor.builder()
                .maxRounds(maxRounds)
                .tools(tools)
                .taskManager(taskManager)
                .sessionMessageStore(sessionMessageStore)
                .conversationStore(conversationStore)
                .instructions(instructions)
                .chatModel(chatModel)
                .streamingChatModel(streamingChatModel)
                .longTermMemoryManager(longTermMemoryManager)
                .enableSession(enableSession)
                .enableTrace(enableTrace)
                .askUserToolName(askUserToolNameFromIntercepts())
                .interceptToolNames(interceptToolNames)
                .thinkingMode(thinkingMode)
                .maxRetries(maxRetries);

        // 上下文压缩（可选，按需引入 ContextCompactionHook）
        if (this.contextPolicy != null) {
            OffloadStore offloadStore = (enableSession && sessionMessageStore != null)
                    ? new SessionBackedOffloadStore(sessionMessageStore)
                    : new NoOpOffloadStore();
            LlmSummarizer summarizer = new LlmSummarizer(this.chatModel);
            List<CompressionStrategy> chain = new ArrayList<>();
            chain.add(new HistoricalToolListStrategy());
            chain.add(new LargeMsgOffloadWithKeepStrategy());
            chain.add(new LargeMsgOffloadNoKeepStrategy());
            chain.add(new HistoricalRoundSummaryStrategy(summarizer));
            chain.add(new CurrentRoundLargeMsgStrategy(summarizer));
            chain.add(new CurrentRoundOverallStrategy(summarizer));
            ContextCompactor compactor = new ContextCompactor(
                    this.contextPolicy, this.chatModel,
                    offloadStore, sessionMessageStore,
                    chain);
            // 压缩 Hook 置列表首部（priority=Integer.MAX_VALUE 已确保最先执行）
            allHooks.add(0, new ContextCompactionHook(compactor));
        }

        // 沙箱（可选，按需引入 SandboxHook）
        if (this.sandboxManager != null) {
            allHooks.add(new com.agentx.ai.core.hook.SandboxHook(this.sandboxManager));
        }

        executorBuilder.hooks(allHooks);

        // 长期记忆（可选）
        if (longTermMemoryManager != null) {
            executorBuilder.longTermMemoryManager(longTermMemoryManager);
        }

        // 延迟工具注册（可选）
        if (deferredToolRegistry != null) {
            executorBuilder.deferredToolRegistry(deferredToolRegistry);
        }

        // 审计 trace（可选）
        if (traceStore != null) {
            executorBuilder.traceStore(traceStore);
        }

        // 中断状态持久化（必有默认实现）
        if (stateStore != null) {
            executorBuilder.stateStore(stateStore);
        }

        return executorBuilder.build();
    }

    private String askUserToolNameFromIntercepts() {
        // 输入类工具语义：ask_user（1.x PauseAdvisor.askUserTool 的对等能力）。
        // 拦截名单中第一个命中 ask_user 语义的工具名即为输入工具；审批类工具走确认语义。
        for (String name : interceptToolNames) {
            if ("ask_user".equalsIgnoreCase(name) || "AskUser".equals(name)) {
                return name;
            }
        }
        return null;
    }

    /**
     * 同步调用 Agent
     *
     * @param query 用户消息
     * @return Agent 响应文本
     */
    public String call(String query) {
        return call(query, RunnableParams.empty());
    }

    /**
     * 同步调用 Agent（带参数）
     *
     * @param query  用户消息
     * @param params 调用参数
     * @return Agent 响应文本
     */
    public String call(String query, RunnableParams params) {
        if (params == null) {
            params = RunnableParams.empty();
        }
        AgentResult result = createExecutor().call(query, params);
        return result.answer();
    }

    /**
     * 流式调用 Agent
     *
     * @param query 用户消息
     * @return 文本流
     */
    public Flux<String> stream(String query) {
        return stream(query, RunnableParams.empty());
    }

    /**
     * 流式调用 Agent（带参数）
     *
     * @param query  用户消息
     * @param params 调用参数
     * @return 文本流（过滤掉暂停事件）
     */
    public Flux<String> stream(String query, RunnableParams params) {
        if (params == null) {
            params = RunnableParams.empty();
        }
        return createExecutor().stream(query, params)
                .filter(e -> e instanceof AgentStreamEvent.Text)
                .map(e -> ((AgentStreamEvent.Text) e).content());
    }

    // ==================== AgentResult API (完整场景) ====================

    /**
     * 同步调用 Agent，返回完整结果（包含暂停状态）。
     *
     * @param query  用户消息
     * @param params 调用参数
     * @return AgentResult（Completed 或 Paused）
     */
    public AgentResult callForResult(String query, RunnableParams params) {
        if (params == null) {
            params = RunnableParams.empty();
        }
        return createExecutor().call(query, params);
    }

    /**
     * 流式调用 Agent，返回完整事件流（包含暂停事件）。
     *
     * @param query  用户消息
     * @param params 调用参数
     * @return AgentStreamEvent 流
     */
    public Flux<AgentStreamEvent> streamForResult(String query, RunnableParams params) {
        if (params == null) {
            params = RunnableParams.empty();
        }
        return createExecutor().stream(query, params);
    }

    // ==================== 恢复 API ====================

    /**
     * 从暂停状态恢复同步执行。
     *
     * @param state       暂停状态
     * @param toolResults 工具调用结果 (key=toolCallId, value=结果文本)
     * @return AgentResult（Completed 或 Paused）
     */
    public AgentResult resume(PauseState state, Map<String, String> toolResults) {
        return createExecutor().resume(state, toolResults);
    }

    /**
     * 从暂停状态恢复流式执行。
     *
     * @param state       暂停状态
     * @param toolResults 工具调用结果
     * @return AgentStreamEvent 流
     */
    public Flux<AgentStreamEvent> resumeStream(PauseState state, Map<String, String> toolResults) {
        return createExecutor().resumeStream(state, toolResults);
    }

    /**
     * 停止指定会话的流式任务
     *
     * @param conversationId 会话 ID
     * @return 是否成功停止
     */
    public boolean stopStream(String conversationId) {
        if (taskManager == null) {
            return false;
        }
        return taskManager.stopTask(conversationId);
    }

    // ==================== 用户主动中断与恢复 ====================

    /**
     * 用户主动中断当前会话的执行。
     *
     * @param conversationId 会话 ID
     * @param message        中断说明消息（可为 null，会附加到 PauseState.interruptMessage）
     * @return true 表示任务存在并已触发中断流程
     */
    public boolean interrupt(String conversationId, String message) {
        if (taskManager == null) {
            return false;
        }
        return taskManager.interrupt(conversationId, message);
    }

    /**
     * 用户主动中断（无消息说明）。
     */
    public boolean interrupt(String conversationId) {
        return interrupt(conversationId, null);
    }

    /**
     * 检查指定会话是否存在未恢复的中断状态。
     *
     * @param conversationId 会话 ID
     * @return true 表示存在可恢复的 PauseState
     */
    public boolean hasInterruptedState(String conversationId) {
        if (stateStore == null || conversationId == null) {
            return false;
        }
        return stateStore.exists(conversationId);
    }

    /**
     * 获取指定会话的中断状态（用于应用层展示"上次有未完成任务"）。
     *
     * @param conversationId 会话 ID
     * @return PauseState，不存在返回 null
     */
    public PauseState getInterruptedState(String conversationId) {
        if (stateStore == null || conversationId == null) {
            return null;
        }
        return stateStore.findByConversationId(conversationId);
    }

    /**
     * 丢弃指定会话的中断状态（用户选择"放弃"）。
     *
     * @param conversationId 会话 ID
     * @return true 表示实际删除了状态
     */
    public boolean discardInterruptedState(String conversationId) {
        if (stateStore == null || conversationId == null) {
            return false;
        }
        return stateStore.delete(conversationId);
    }

    /**
     * 按会话 ID 从中断状态恢复流式执行。
     *
     * @param conversationId 会话 ID
     * @return AgentStreamEvent 流；状态不存在时返回 Flux.error
     */
    public Flux<AgentStreamEvent> resumeStream(String conversationId) {
        return resumeStream(conversationId, null);
    }

    /**
     * 按会话 ID 从中断状态恢复流式执行，并附带用户中断后补充的新消息。
     *
     * @param conversationId 会话 ID
     * @param resumeQuery    用户中断后补充的新消息（可为 null）
     * @return AgentStreamEvent 流；状态不存在时返回 Flux.error
     */
    public Flux<AgentStreamEvent> resumeStream(String conversationId, String resumeQuery) {
        PauseState state = getInterruptedState(conversationId);
        if (state == null) {
            return Flux.error(new IllegalStateException(
                    "No interrupted state for conversation: " + conversationId));
        }
        return createExecutor().resumeStream(state, Map.of(), resumeQuery);
    }

    /**
     * 检查指定会话是否有正在运行的任务
     *
     * @param conversationId 会话 ID
     * @return 是否有正在运行的任务
     */
    public boolean hasRunningTask(String conversationId) {
        if (taskManager == null) {
            return false;
        }
        return taskManager.hasRunningTask(conversationId);
    }

    /**
     * 获取当前运行中的任务数量
     *
     * @return 任务数量
     */
    public int getRunningTaskCount() {
        if (taskManager == null) {
            return 0;
        }
        return taskManager.getTaskCount();
    }

    // ==================== Getters ====================

    public String getName() {
        return name;
    }

    public String getDescription() {
        return description;
    }

    /**
     * 配置为 SubAgent 模式（由父 Agent 的 build() 在 wrap supplier 中调用）。
     * <p>
     * - 禁用 session：不记录 agentx_session<br>
     * - 注入父 TraceStore：复用父 Agent 的 trace 审计（受父 enableTrace 控制，null 则不记录）<br>
     * - 长期记忆天然无：SubAgent 不持有 LongTermMemoryManager
     */
    void configureAsSubAgent(TraceStore parentTraceStore) {
        this.enableSession = false;
        this.traceStore = parentTraceStore;
    }

    public ChatModel getChatModel() {
        return chatModel;
    }

    public String getInstructions() {
        return instructions;
    }

    /**
     * 获取 HITL 拦截工具名单（只读）。
     */
    public Set<String> getInterceptToolNames() {
        return interceptToolNames;
    }

    public List<AgentTool> getTools() {
        return tools;
    }

    public AgentTaskManager getTaskManager() {
        return taskManager;
    }

    public int getMaxRounds() {
        return maxRounds;
    }

    /**
     * 合并多个工具数组，按工具名称去重。
     * 委托给 {@link ToolMergeUtil#mergeTools}。
     *
     * @param toolArrays 工具数组
     * @return 合并并去重后的工具数组
     */
    @SafeVarargs
    public static AgentTool[] mergeTools(AgentTool[]... toolArrays) {
        return ToolMergeUtil.mergeTools(toolArrays);
    }

    /**
     * 合并多个工具集合（List），按工具名称去重。
     *
     * @param toolLists 工具集合
     * @return 合并并去重后的工具数组
     */
    @SafeVarargs
    public static AgentTool[] mergeTools(List<AgentTool>... toolLists) {
        return ToolMergeUtil.mergeTools(toolLists);
    }

    /**
     * Builder 模式构建 ReactAgent
     */
    public static class Builder {
        private String name;
        private String description;
        private AgentModels models;
        private ChatModel chatModel;
        private StreamingChatModel streamingChatModel;
        private final List<AgentTool> tools = new ArrayList<>();
        private final Set<String> interceptToolNames = new LinkedHashSet<>();
        private int maxRounds = 100;
        private AgentTaskManager taskManager;
        private String instructions;
        private DataSource dataSource;
        private LongTermMemoryConfig longTermMemoryConfig;
        private boolean enableSession = true;
        private boolean askUser = false;
        private final List<AgentHook> hooks = new ArrayList<>();
        private ThinkingMode thinkingMode = ThinkingMode.DISABLED;
        private int maxRetries = 3;
        private ContextPolicy contextPolicy;
        private SandboxConfig sandboxConfig;
        private DeferredToolRegistry deferredToolRegistry;
        private ToolSearchConfig deferredToolSearchConfig;
        private final List<AgentTool> pendingDeferredTools = new ArrayList<>();
        private boolean enableTrace = true;
        private PauseStateStore stateStore;
        private final List<Supplier<ReactAgent>> subAgentProviders = new ArrayList<>();

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder description(String description) {
            this.description = description;
            return this;
        }

        /**
         * 推荐入口：一次注入同步 + 流式模型对。
         * <p>可直接 {@code AgentModels.of(chatModel)}（双接口自动复用），
         * 或 {@code AgentModels.of(chatModel, streamingChatModel)}（外部 API 双实例）。
         * <p>设置后优先于 {@link #chatModel(ChatModel)} / {@link #streamingChatModel(StreamingChatModel)}。
         */
        public Builder models(AgentModels models) {
            this.models = models;
            return this;
        }

        /**
         * 同步模型。若其实现了 {@link StreamingChatModel}，流式侧自动复用；
         * 否则建议改用 {@link #models(AgentModels)} 显式提供流式实例，避免伪流式。
         */
        public Builder chatModel(ChatModel chatModel) {
            this.chatModel = chatModel;
            return this;
        }

        /**
         * 可选：显式指定流式模型。不传时若 chatModel 本身实现了 {@link StreamingChatModel}
         * 自动复用；两者都未提供时 {@link #build()} 会失败。
         */
        public Builder streamingChatModel(StreamingChatModel streamingChatModel) {
            this.streamingChatModel = streamingChatModel;
            return this;
        }

        public Builder tools(AgentTool... tools) {
            if (tools != null) {
                for (AgentTool tool : tools) {
                    this.tools.add(tool);
                }
            }
            return this;
        }

        public Builder tools(List<AgentTool> tools) {
            if (tools != null) {
                this.tools.addAll(tools);
            }
            return this;
        }

        /**
         * 配置 HITL 拦截工具名单。
         * <p>
         * 名单中的工具被 LLM 调用时不执行，而是暂停等待外部输入：
         * <ul>
         *   <li>ask_user 等输入类工具：用户回答即工具结果</li>
         *   <li>其他审批类工具：用户确认后框架执行，拒绝则不执行</li>
         * </ul>
         * <p>
         * 替代 1.x 的 {@code advisors(PauseAdvisor.builder()...)}。
         *
         * @param toolNames 拦截工具名
         */
        public Builder interceptTools(String... toolNames) {
            if (toolNames != null) {
                for (String n : toolNames) {
                    if (n != null && !n.isBlank()) {
                        this.interceptToolNames.add(n);
                    }
                }
            }
            return this;
        }

        public Builder maxRounds(int maxRounds) {
            this.maxRounds = maxRounds;
            return this;
        }

        public Builder taskManager(AgentTaskManager taskManager) {
            this.taskManager = taskManager;
            return this;
        }

        public Builder instructions(String instructions) {
            this.instructions = instructions;
            return this;
        }

        /**
         * 配置 DataSource，框架自动管理会话存储与追踪存储。
         *
         * @param dataSource 数据源
         */
        public Builder dataSource(DataSource dataSource) {
            this.dataSource = dataSource;
            return this;
        }

        /**
         * 启用长期记忆。传入 {@link LongTermMemoryConfig} 即启用，不调用即不启用。
         *
         * @param config 长期记忆配置（封装 EmbeddingStore + EmbeddingModel）
         */
        public Builder longTermMemory(LongTermMemoryConfig config) {
            this.longTermMemoryConfig = config;
            return this;
        }

        /**
         * 是否启用会话历史记录（agentx_session）。
         * 默认启用。
         *
         * @param enabled true 启用（默认），false 禁用
         */
        public Builder enableSession(boolean enabled) {
            this.enableSession = enabled;
            return this;
        }

        /**
         * 启用 Human-in-the-Loop：Agent 可主动向用户提问。
         * <p>
         * 启用后框架自动：
         * - 注册 AskUserTool（LLM 可调用 ask_user 工具）
         * - 将 ask_user 加入 HITL 拦截名单，暂停循环等待外部回答
         * <p>
         * 使用 callForResult / streamForResult 获取 AgentResult.Paused，
         * 然后调用 resume 继续对话。
         *
         * @param askUser true 启用
         */
        public Builder askUser(boolean askUser) {
            this.askUser = askUser;
            return this;
        }

        /**
         * 注册 Hook，在 Agent 生命周期关键节点被调用。
         *
         * @param hooks Hook 实例
         */
        public Builder hooks(AgentHook... hooks) {
            if (hooks != null) {
                this.hooks.addAll(List.of(hooks));
            }
            return this;
        }

        /**
         * 注册 Hook（List 形式）。
         *
         * @param hooks Hook 列表
         */
        public Builder hooks(List<AgentHook> hooks) {
            if (hooks != null) {
                this.hooks.addAll(hooks);
            }
            return this;
        }

        /**
         * 配置思考模型的输出格式。
         * <ul>
         *   <li>{@link ThinkingMode#THINK_TAG} - 思考内容嵌入 content 中的 &lt;think/&gt; 标签（MiniMax等）</li>
         *   <li>{@link ThinkingMode#REASONING_CONTENT} - 思考内容通过独立 thinking 通道返回（DeepSeek、Qwen3.6 等，
         *       需在模型侧开启 returnThinking）</li>
         * </ul>
         */
        public Builder thinkingMode(ThinkingMode thinkingMode) {
            this.thinkingMode = thinkingMode != null ? thinkingMode : ThinkingMode.DISABLED;
            return this;
        }

        /**
         * 配置 LLM 调用的最大重试次数。默认 3 次。设为 0 禁用重试。
         */
        public Builder maxRetries(int maxRetries) {
            this.maxRetries = maxRetries;
            return this;
        }

        /**
         * 配置上下文压缩策略，启用后 Agent 在每轮 LLM 调用前自动压缩历史消息。
         */
        public Builder contextPolicy(ContextPolicy contextPolicy) {
            this.contextPolicy = contextPolicy;
            return this;
        }

        /**
         * 配置沙箱，启用隔离执行。
         */
        public Builder sandbox(SandboxConfig sandboxConfig) {
            this.sandboxConfig = sandboxConfig;
            return this;
        }

        /**
         * 配置延迟加载工具，启用 ToolSearch 能力。
         * <p>注册表在 {@link #build()} 解析完模型后创建，因此可在 {@link #models(AgentModels)} 之前调用。
         *
         * @param config 搜索配置
         * @param tools  延迟加载的工具
         */
        public Builder deferredTools(ToolSearchConfig config, AgentTool... tools) {
            if (tools != null && tools.length > 0) {
                this.deferredToolSearchConfig = config;
                this.pendingDeferredTools.addAll(List.of(tools));
            }
            return this;
        }

        /**
         * 注册子 Agent。
         * <p>
         * 子 Agent 会被包装为 {@code call_{name}} 工具，主 Agent 的 LLM 根据描述自动决定是否委派。
         *
         * @param agentProvider 子 Agent 工厂（每次调用创建新实例，保证线程安全）
         */
        public Builder subAgent(Supplier<ReactAgent> agentProvider) {
            this.subAgentProviders.add(agentProvider);
            return this;
        }

        /**
         * 启用 LLM 调用审计。
         * 启用后，每次 LLM 调用的请求/响应记录到 {@code agentx_trace} 表。
         * 需要同时配置 {@link #dataSource(DataSource)} 才会入库。
         * 默认 true。
         */
        public Builder enableTrace(boolean enableTrace) {
            this.enableTrace = enableTrace;
            return this;
        }

        /**
         * 向后兼容：等同于 {@link #enableTrace(boolean)}。
         */
        public Builder requestLogging(boolean requestLogging) {
            this.enableTrace = requestLogging;
            return this;
        }

        /**
         * 配置 Agent 状态存储，启用用户主动中断的断点重连能力。
         * 未配置时默认使用 {@link InMemoryPauseStateStore}（单节点、进程内、TTL 7 天）。
         */
        public Builder stateStore(PauseStateStore stateStore) {
            this.stateStore = stateStore;
            return this;
        }

        public ReactAgent build() {
            AgentModels resolved = models != null
                    ? models
                    : AgentModels.resolve(chatModel, streamingChatModel);
            this.chatModel = resolved.chatModel();
            this.streamingChatModel = resolved.streamingChatModel();

            if (deferredToolSearchConfig != null && !pendingDeferredTools.isEmpty()) {
                this.deferredToolRegistry = DeferredToolRegistry.create(
                        deferredToolSearchConfig, List.copyOf(pendingDeferredTools), this.chatModel);
            }

            SessionMessageStore sessionMessageStore = null;
            ConversationStore conversationStore = null;

            TraceStore traceStore = null;
            if (dataSource != null) {
                sessionMessageStore = DataSourceStorageFactory.createSessionMessageStore(dataSource);
                conversationStore = DataSourceStorageFactory.createConversationStore(dataSource);
                traceStore = DataSourceStorageFactory.createTraceStore(dataSource);
            }

            LongTermMemoryManager longTermMemoryManager = null;
            if (longTermMemoryConfig != null) {
                longTermMemoryManager = new LongTermMemoryManager(longTermMemoryConfig, chatModel);
            }

            if (taskManager == null) {
                taskManager = new AgentTaskManager();
            }

            // askUser=true 时自动注册内置 AskUserTool 并加入 HITL 拦截名单
            if (askUser) {
                tools.addAll(AskUserTool.create());
                interceptToolNames.add("ask_user");
            }

            // 处理 SubAgent：调一次 supplier 提取 name/description，注册为工具
            final TraceStore subAgentTraceStore = this.enableTrace ? traceStore : null;
            for (Supplier<ReactAgent> provider : subAgentProviders) {
                ReactAgent prototype = provider.get();
                String agentName = prototype.getName();
                String agentDesc = prototype.getDescription();
                if (agentName == null || agentName.isBlank()) {
                    throw new IllegalArgumentException("SubAgent 必须设置 name");
                }
                Supplier<ReactAgent> wrappedProvider = () -> {
                    ReactAgent agent = provider.get();
                    agent.configureAsSubAgent(subAgentTraceStore);
                    return agent;
                };
                tools.add(SubAgentTool.create(agentName, agentDesc, wrappedProvider));
            }

            return new ReactAgent(this, sessionMessageStore, conversationStore,
                    longTermMemoryManager, traceStore,
                    stateStore != null ? stateStore : new InMemoryPauseStateStore());
        }
    }
}
