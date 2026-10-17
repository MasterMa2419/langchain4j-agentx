package com.agentx.ai.core.agent;

import com.agentx.ai.core.agent.internal.AgentTaskManager;
import com.agentx.ai.core.agent.internal.ChatExecutor;
import com.agentx.ai.core.chatmodels.AgentModels;
import com.agentx.ai.core.hook.AgentHook;
import com.agentx.ai.core.memory.store.ConversationStore;
import com.agentx.ai.core.memory.store.DataSourceStorageFactory;
import com.agentx.ai.core.memory.store.SessionMessageStore;
import com.agentx.ai.core.model.AgentResult;
import com.agentx.ai.core.model.AgentStreamEvent;
import com.agentx.ai.core.model.AgentTool;
import com.agentx.ai.core.model.RunnableParams;
import com.agentx.ai.core.model.ThinkingMode;
import com.agentx.ai.core.trace.TraceStore;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.StreamingChatModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import reactor.core.publisher.Flux;

import javax.sql.DataSource;
import java.util.List;

/**
 * ChatAgent - 普通对话 Agent（单轮 + 有界工具调用）。
 * <p>
 * 与 {@link ReactAgent}（ReAct 多轮循环）相对：不做 Reasoning + Acting 自主循环，
 * 至多执行 {@code maxToolRounds} 轮工具调用（默认 1 轮），随后一轮必然产出最终答案，
 * LLM 调用次数上限为 {@code maxToolRounds + 1}。适合问答、客服、翻译、摘要，
 * 以及"查一次再回答"这类不需要反复推理的浅层工具场景。
 * <p>
 * 工具支持：通过 {@link Builder#tools(AgentTool...)} 注册基础 Function Calling 工具，
 * 模型调用后由框架执行并回填结果。仅覆盖基础工具调用，不含 HITL 拦截、ToolSearch 延迟工具、
 * 沙箱隔离与 SubAgent 委派——这些能力请使用 {@link ReactAgent}。
 * <p>
 * 复用与 ReactAgent 相同的基础设施：
 * <ul>
 *   <li>会话记忆：{@code dataSource} 落库后按 {@code conversationId} 跨多轮续聊</li>
 *   <li>流式输出：{@code stream} / {@code streamForResult} token 级事件</li>
 *   <li>持久化 + Trace：{@code agentx_session} / {@code agentx_conversation} / {@code agentx_trace}</li>
 *   <li>Hook 生命周期：BeforeCall / BeforeReasoning / AfterReasoning / AfterCall / Error</li>
 *   <li>思考模型适配：{@link ThinkingMode#THINK_TAG} / {@link ThinkingMode#REASONING_CONTENT}</li>
 * </ul>
 * <p>
 * 不具备的能力（由 ReAct 引擎提供）：HITL 中断恢复、上下文压缩、
 * SubAgent 委派（作为父 Agent）、沙箱隔离、长期记忆。需要这些能力请使用 {@link ReactAgent}。
 * <p>
 * 2.0 起 AI 层基于 LangChain4j：推荐通过 {@link AgentModels} 一次构造同步 + 流式模型对，
 * 与 {@link ReactAgent} 的模型注入方式保持一致。
 *
 * @author bigchui
 */
public class ChatAgent implements Agent {

    private static final Logger log = LoggerFactory.getLogger(ChatAgent.class);

    private final String name;
    private final String description;
    private final ChatModel chatModel;
    private final StreamingChatModel streamingChatModel;
    private final String instructions;
    private final AgentTaskManager taskManager;
    private final SessionMessageStore sessionMessageStore;
    private final ConversationStore conversationStore;
    private final List<AgentTool> tools;
    private final int maxToolRounds;
    private boolean enableSession;
    private final List<AgentHook> hooks;
    private final ThinkingMode thinkingMode;
    private final int maxRetries;
    private final boolean enableTrace;
    private TraceStore traceStore;

    private ChatAgent(Builder builder, SessionMessageStore sessionMessageStore,
                      ConversationStore conversationStore, TraceStore traceStore) {
        this.name = builder.name;
        this.description = builder.description;
        this.chatModel = builder.chatModel;
        this.streamingChatModel = builder.streamingChatModel;
        this.instructions = builder.instructions;
        this.taskManager = builder.taskManager;
        this.sessionMessageStore = sessionMessageStore;
        this.conversationStore = conversationStore;
        this.tools = builder.tools != null ? List.copyOf(builder.tools) : List.of();
        this.maxToolRounds = Math.max(0, builder.maxToolRounds);
        this.enableSession = builder.enableSession;
        this.hooks = builder.hooks;
        this.thinkingMode = builder.thinkingMode;
        this.maxRetries = builder.maxRetries;
        this.enableTrace = builder.enableTrace;
        this.traceStore = traceStore;
    }

    /**
     * 创建 Builder
     */
    public static Builder builder() {
        return new Builder();
    }

    private ChatExecutor createExecutor() {
        return ChatExecutor.builder()
                .taskManager(taskManager)
                .instructions(instructions)
                .chatModel(chatModel)
                .streamingChatModel(streamingChatModel)
                .sessionMessageStore(sessionMessageStore)
                .conversationStore(conversationStore)
                .enableSession(enableSession)
                .enableTrace(enableTrace)
                .traceStore(traceStore)
                .tools(tools)
                .maxToolRounds(maxToolRounds)
                .hooks(hooks)
                .thinkingMode(thinkingMode)
                .maxRetries(maxRetries)
                .build();
    }

    // ==================== 调用 API ====================

    @Override
    public String call(String query) {
        return call(query, RunnableParams.empty());
    }

    @Override
    public String call(String query, RunnableParams params) {
        if (params == null) {
            params = RunnableParams.empty();
        }
        AgentResult result = createExecutor().call(query, params);
        return result.answer();
    }

    @Override
    public Flux<String> stream(String query) {
        return stream(query, RunnableParams.empty());
    }

    @Override
    public Flux<String> stream(String query, RunnableParams params) {
        if (params == null) {
            params = RunnableParams.empty();
        }
        return createExecutor().stream(query, params)
                .filter(e -> e instanceof AgentStreamEvent.Text)
                .map(e -> ((AgentStreamEvent.Text) e).content());
    }

    @Override
    public AgentResult callForResult(String query, RunnableParams params) {
        if (params == null) {
            params = RunnableParams.empty();
        }
        return createExecutor().call(query, params);
    }

    @Override
    public Flux<AgentStreamEvent> streamForResult(String query, RunnableParams params) {
        if (params == null) {
            params = RunnableParams.empty();
        }
        return createExecutor().stream(query, params);
    }

    // ==================== 任务控制 ====================

    @Override
    public boolean stopStream(String conversationId) {
        if (taskManager == null) {
            return false;
        }
        return taskManager.stopTask(conversationId);
    }

    @Override
    public boolean hasRunningTask(String conversationId) {
        if (taskManager == null) {
            return false;
        }
        return taskManager.hasRunningTask(conversationId);
    }

    /**
     * 获取当前运行中的任务数量
     */
    public int getRunningTaskCount() {
        if (taskManager == null) {
            return 0;
        }
        return taskManager.getTaskCount();
    }

    // ==================== SubAgent 模式 ====================

    /**
     * 返回注册的基础工具清单（SubAgent 校验依赖此约定：作为子代理时不得携带
     * {@code call_*} / {@code ask_user} 这类需要父引擎能力的工具）。
     */
    @Override
    public List<AgentTool> getTools() {
        return tools;
    }

    /**
     * 配置为 SubAgent 模式（由父 Agent 的 build() 在 wrap supplier 中调用）。
     * <p>
     * - 禁用 session：不记录 agentx_session，父 Agent 是唯一的会话持久化边界<br>
     * - 注入父 TraceStore：复用父 Agent 的 trace 审计（受父 enableTrace 控制，null 则不记录）
     */
    @Override
    public void configureAsSubAgent(TraceStore parentTraceStore) {
        this.enableSession = false;
        this.traceStore = parentTraceStore;
    }

    // ==================== Getters ====================

    @Override
    public String getName() {
        return name;
    }

    @Override
    public String getDescription() {
        return description;
    }

    public ChatModel getChatModel() {
        return chatModel;
    }

    public String getInstructions() {
        return instructions;
    }

    /**
     * 已注册的工具清单（只读）。
     */
    public List<AgentTool> getToolList() {
        return tools;
    }

    /**
     * 允许的工具执行轮数；0 表示不向 LLM 暴露工具。
     */
    public int getMaxToolRounds() {
        return maxToolRounds;
    }

    public AgentTaskManager getTaskManager() {
        return taskManager;
    }

    /**
     * Builder 模式构建 ChatAgent（普通单轮对话 Agent）
     */
    public static class Builder {
        private String name;
        private String description;
        private AgentModels models;
        private ChatModel chatModel;
        private StreamingChatModel streamingChatModel;
        private String instructions;
        private DataSource dataSource;
        private SessionMessageStore sessionMessageStore;
        private ConversationStore conversationStore;
        private boolean enableSession = true;
        private boolean enableTrace = true;
        private List<AgentHook> hooks;
        private ThinkingMode thinkingMode = ThinkingMode.DISABLED;
        private int maxRetries = 3;
        private AgentTaskManager taskManager;
        private List<AgentTool> tools;
        private int maxToolRounds = 1;

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder description(String description) {
            this.description = description;
            return this;
        }

        /**
         * 推荐入口：一次注入同步 + 流式模型对（对齐 {@link ReactAgent#builder()}）。
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
         * 可选：显式指定流式模型。两者都未提供时 {@link #build()} 会失败。
         */
        public Builder streamingChatModel(StreamingChatModel streamingChatModel) {
            this.streamingChatModel = streamingChatModel;
            return this;
        }

        /**
         * 系统指令（system prompt）。每次调用作为 SystemMessage 发出。
         */
        public Builder instructions(String instructions) {
            this.instructions = instructions;
            return this;
        }

        /**
         * 配置 DataSource，框架自动管理会话存储与追踪存储
         * （{@code agentx_session} / {@code agentx_conversation} / {@code agentx_trace}）。
         * 传入后同一 {@code conversationId} 即拥有跨调用会话记忆。
         *
         * @param dataSource 数据源
         */
        public Builder dataSource(DataSource dataSource) {
            this.dataSource = dataSource;
            return this;
        }

        /**
         * 注入自定义会话消息存储实现（如内存版、Redis 版）。
         * 未配置 {@link #dataSource(DataSource)} 时生效，优先级低于 DataSource。
         */
        public Builder sessionMessageStore(SessionMessageStore store) {
            this.sessionMessageStore = store;
            return this;
        }

        /**
         * 注入自定义会话窗口（调用边界）存储实现。未配置 DataSource 时生效。
         */
        public Builder conversationStore(ConversationStore store) {
            this.conversationStore = store;
            return this;
        }

        /**
         * 是否启用会话历史记录（agentx_session）。默认启用。
         *
         * @param enabled true 启用（默认），false 禁用
         */
        public Builder enableSession(boolean enabled) {
            this.enableSession = enabled;
            return this;
        }

        /**
         * 启用 LLM 调用审计。需要同时配置 {@link #dataSource(DataSource)} 才会入库。默认 true。
         */
        public Builder enableTrace(boolean enableTrace) {
            this.enableTrace = enableTrace;
            return this;
        }

        /**
         * 注册 Hook，在 Agent 生命周期关键节点被调用
         * （BeforeCall / BeforeReasoning / AfterReasoning / AfterCall / Error）。
         */
        public Builder hooks(List<AgentHook> hooks) {
            this.hooks = hooks;
            return this;
        }

        /**
         * 配置思考模型的输出格式（对齐 {@link ReactAgent#builder()}）。
         * <ul>
         *   <li>{@link ThinkingMode#THINK_TAG} - 思考内容嵌入 content 中的 &lt;think/&gt; 标签（MiniMax等）</li>
         *   <li>{@link ThinkingMode#REASONING_CONTENT} - 思考内容通过独立 thinking 通道返回（DeepSeek 等）</li>
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
         * 自定义任务管理器（默认内存实现，单实例内生效）。
         */
        public Builder taskManager(AgentTaskManager taskManager) {
            this.taskManager = taskManager;
            return this;
        }

        /**
         * 注册基础工具（Function Calling）：模型可调用，框架执行后回填结果并继续本轮回答。
         * <p>
         * 工具轮次受 {@link #maxToolRounds(int)} 限制，默认只允许 1 轮工具调用。
         * 不含 HITL 拦截、ToolSearch 与沙箱；需要这些能力请使用 {@link ReactAgent}。
         *
         * @param tools 工具列表（可变参数）
         */
        public Builder tools(AgentTool... tools) {
            this.tools = tools != null ? List.of(tools) : List.of();
            return this;
        }

        /**
         * 注册基础工具（Function Calling），语义同 {@link #tools(AgentTool...)}。
         *
         * @param tools 工具列表
         */
        public Builder tools(List<AgentTool> tools) {
            this.tools = tools;
            return this;
        }

        /**
         * 允许的工具执行轮数，默认 1。
         * <p>
         * 语义：模型可发起至多 N 轮工具调用；用尽后最后一轮不再携带工具规格，
         * 模型必然产出最终答案。因此单次对话的 LLM 调用次数 ≤ N + 1。
         * <ul>
         *   <li>1（默认）：一次工具批次 + 一次最终回答，适合"查一次再回答"</li>
         *   <li>N &gt; 1：允许链式调用（如先检索再取详情）</li>
         *   <li>0：完全不向 LLM 暴露工具，等价纯对话</li>
         * </ul>
         *
         * @param maxToolRounds 工具执行轮数（负数按 0 处理）
         */
        public Builder maxToolRounds(int maxToolRounds) {
            this.maxToolRounds = maxToolRounds;
            return this;
        }

        public ChatAgent build() {
            AgentModels resolved = models != null
                    ? models
                    : AgentModels.resolve(chatModel, streamingChatModel);
            this.chatModel = resolved.chatModel();
            this.streamingChatModel = resolved.streamingChatModel();

            SessionMessageStore sessionMessageStore = null;
            ConversationStore conversationStore = null;
            TraceStore traceStore = null;
            if (dataSource != null) {
                sessionMessageStore = DataSourceStorageFactory.createSessionMessageStore(dataSource);
                conversationStore = DataSourceStorageFactory.createConversationStore(dataSource);
                traceStore = DataSourceStorageFactory.createTraceStore(dataSource);
            } else {
                sessionMessageStore = this.sessionMessageStore;
                conversationStore = this.conversationStore;
            }

            if (taskManager == null) {
                taskManager = new AgentTaskManager();
            }

            return new ChatAgent(this, sessionMessageStore, conversationStore, traceStore);
        }
    }
}
