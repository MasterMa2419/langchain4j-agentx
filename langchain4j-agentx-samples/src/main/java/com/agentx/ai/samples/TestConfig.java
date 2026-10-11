package com.agentx.ai.samples;

import com.agentx.ai.core.agent.ReactAgent;
import com.agentx.ai.core.chatmodels.DeepSeekV4ChatModel;
import com.agentx.ai.core.model.AgentStreamEvent;
import com.agentx.ai.core.model.PauseState;
import com.agentx.ai.core.model.RunnableParams;
import com.agentx.ai.core.model.SubAgentSource;
import com.zaxxer.hikari.HikariDataSource;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.model.openai.OpenAiEmbeddingModel;
import dev.langchain4j.store.embedding.pgvector.PgVectorEmbeddingStore;
import dev.langchain4j.store.embedding.pgvector.MetadataStorageConfig;
import dev.langchain4j.store.embedding.pgvector.DefaultMetadataStorageConfig;
import org.springframework.jdbc.core.JdbcTemplate;

import javax.sql.DataSource;
import java.io.IOException;
import java.io.InputStream;
import java.time.Duration;
import java.util.Properties;
import java.util.UUID;

/**
 * 测试公共配置 — 所有测试类共享的工厂方法和常量。
 *
 * @author bigchui
 *
 */
public final class TestConfig {

    private static final Properties secrets = loadSecrets();

    // 便捷访问方法
    private static String s(String key) { return secrets.getProperty(key); }
    private static String s(String key, String defaultValue) { return secrets.getProperty(key, defaultValue); }
    private static int si(String key, int defaultValue) { return Integer.parseInt(s(key, String.valueOf(defaultValue))); }

    public static String secret(String key) {
        return s(key);
    }

    // ===== 公开常量（用于测试打印）=====
    public static final String CHAT_MODEL = s("dashscope.chat.model", "qwen-plus");
    public static final String EMBEDDING_MODEL = s("embedding.model", "text-embedding-v3");
    public static final String SKILLS_DIR = s("skills.dir", "");

    /** HTTP 响应超时（5 分钟），思考模型非流式调用需要较长等待 */
    private static final Duration HTTP_TIMEOUT = Duration.ofSeconds(300);

    private TestConfig() {
    }

    private static Properties loadSecrets() {
        Properties props = new Properties();
        try (InputStream is = TestConfig.class.getClassLoader().getResourceAsStream("secrets.properties")) {
            if (is != null) {
                props.load(is);
            } else {
                System.err.println("[TestConfig] 未找到 secrets.properties，请复制 secrets.properties.example 并填入真实值");
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to load secrets.properties", e);
        }
        return props;
    }

    // ---------- Factory Methods ----------

    public static ChatModel createChatModel() {
        return OpenAiChatModel.builder()
                .baseUrl(s("dashscope.base.url", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
                .apiKey(s("dashscope.api.key"))
                .modelName(s("dashscope.chat.model", "qwen-plus"))
                .temperature(0.7)
                .timeout(HTTP_TIMEOUT)
                .returnThinking(true)
                .build();
    }

    public static ChatModel createMiniMaxChatModel() {
        return OpenAiChatModel.builder()
                .baseUrl(s("minimax.base.url", "https://api.minimaxi.com/v1"))
                .apiKey(s("minimax.api.key"))
                .modelName(s("minimax.chat.model", "MiniMax-M2.7"))
                .temperature(0.7)
                .timeout(HTTP_TIMEOUT)
                .build();
    }

    public static ChatModel createMultimodalChatModel() {
        return OpenAiChatModel.builder()
                .baseUrl(s("dashscope.base.url", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
                .apiKey(s("dashscope.api.key"))
                .modelName(s("multimodal.model", "qwen-vl-plus"))
                .temperature(0.2)
                .timeout(HTTP_TIMEOUT)
                .build();
    }

    /**
     * DeepSeek 的 OpenAI 兼容实现（不提取思考内容）。
     */
    public static ChatModel createDeepSeekChatModel() {
        return OpenAiChatModel.builder()
                .baseUrl(s("deepseek.base.url", "https://api.deepseek.com/v1"))
                .apiKey(s("deepseek.api.key"))
                .modelName(s("deepseek.chat.model", "deepseek-chat"))
                .temperature(0.7)
                .timeout(HTTP_TIMEOUT)
                .build();
    }

    /**
     * 正确用法：DeepSeek V4 思考模型工厂（原生 reasoning_content 回传）。
     */
    public static ChatModel createDeepSeekV4ChatModel() {
        return DeepSeekV4ChatModel.builder()
                .apiKey(s("deepseek.api.key"))
                .modelName(s("deepseek.chat.model", "deepseek-v4"))
                .temperature(0.7)
                .build();
    }

    public static ChatModel createZhiPuChatModel() {
        // zhipu-ai 专用实现已停更，统一走 OpenAI 兼容接口
        return createZhiPuOpenAiChatModel();
    }

    /**
     * 创建 OpenAI 兼容风格的 ZhiPu ChatModel。
     * <p>
     * 通过 OpenAI 兼容接口访问 GLM 模型，支持 {@code reasoning_content} 字段提取
     * （{@code returnThinking(true)}），支持 REASONING_CONTENT 思考模式。
     */
    public static ChatModel createZhiPuOpenAiChatModel() {
        return OpenAiChatModel.builder()
                .baseUrl(s("zhipu.openai.base.url", "https://open.bigmodel.cn/api/coding/paas/v4"))
                .apiKey(s("zhipu.api.key"))
                .modelName(s("zhipu.chat.model", "glm-4.7"))
                .temperature(0.7)
                .timeout(HTTP_TIMEOUT)
                .returnThinking(true)
                .build();
    }

    public static EmbeddingModel createEmbeddingModel() {
        return OpenAiEmbeddingModel.builder()
                .baseUrl(s("dashscope.base.url", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
                .apiKey(s("dashscope.api.key"))
                .modelName(EMBEDDING_MODEL)
                .dimensions(1024)
                .build();
    }

    /**
     * 创建使用错误 API Key 的 ChatModel，用于测试 ErrorEvent。
     */
    public static ChatModel createBadApiKeyChatModel() {
        return DeepSeekV4ChatModel.builder()
                .apiKey("sk-bad-api-key-for-testing-error-event")
                .baseUrl(s("deepseek.base.url", "https://api.deepseek.com/v1"))
                .modelName(s("deepseek.chat.model", "deepseek-chat"))
                .temperature(0.7)
                .build();
    }

    public static DataSource createMySqlDataSource() {
        HikariDataSource ds = new HikariDataSource();
        ds.setJdbcUrl(s("mysql.url", "jdbc:mysql://localhost:3306/agent_test?useSSL=false&allowPublicKeyRetrieval=true"));
        ds.setUsername(s("mysql.user", "root"));
        ds.setPassword(s("mysql.password", "root"));
        ds.setDriverClassName("com.mysql.cj.jdbc.Driver");
        return ds;
    }

    public static DataSource createPgDataSource() {
        HikariDataSource ds = new HikariDataSource();
        ds.setJdbcUrl(String.format("jdbc:postgresql://%s:%d/%s",
                s("pg.host", "192.168.113.52"),
                si("pg.port", 5433),
                s("pg.database", "vector_store")));
        ds.setUsername(s("pg.user", "postgres"));
        ds.setPassword(s("pg.password", "postgres"));
        ds.setDriverClassName("org.postgresql.Driver");
        return ds;
    }

    public static PgVectorEmbeddingStore createPgVectorStore(DataSource pgDataSource, EmbeddingModel embeddingModel) {
        return createPgVectorStore(pgDataSource,
                s("pg.table.name", "agentx_long_term_memory"));
    }

    public static PgVectorEmbeddingStore createPgVectorStore(DataSource pgDataSource, String tableName) {
        return PgVectorEmbeddingStore.datasourceBuilder()
                .datasource(pgDataSource)
                .table(tableName)
                .dimension(1024)
                .useIndex(true)
                .createTable(true)
                .dropTableFirst(false)
                .build();
    }

    // ---------- Helper Methods ----------

    public static String randomConvId() {
        return "conv_" + UUID.randomUUID();
    }

    public static String randomUserId(String prefix) {
        return prefix + "_" + System.currentTimeMillis();
    }

    public static RunnableParams buildParams(String convId, String userId) {
        return RunnableParams.builder()
                .conversationId(convId)
                .userId(userId)
                .build();
    }

    /**
     * 流式调用 Agent 并实时打印输出。
     */
    public static void streamAndPrint(ReactAgent agent, String query) {
        streamAndPrint(agent, query, null);
    }

    public static void streamAndPrint(ReactAgent agent, String query, RunnableParams params) {
        System.out.println("Q: " + query);
        System.out.print("A: ");
        if (params != null) {
            agent.stream(query, params)
                    .doOnNext(chunk -> {
                        System.out.print(chunk);
                        System.out.flush();
                    })
                    .doOnError(err -> System.err.println("\nError: " + err.getMessage()))
                    .blockLast();
        } else {
            agent.stream(query)
                    .doOnNext(chunk -> {
                        System.out.print(chunk);
                        System.out.flush();
                    })
                    .doOnError(err -> System.err.println("\nError: " + err.getMessage()))
                    .blockLast();
        }
        System.out.println("\n");
    }

    public static void printTestHeader(String title) {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("  " + title);
        System.out.println("=".repeat(60) + "\n");
    }

    // ---------- 流式事件工具方法 ----------

    /**
     * 上一个事件是否为 Thinking，用于控制标记打印
     */
    private static boolean lastEventWasThinking = false;

    /**
     * 当前事件来源的 Agent 名称（null = 主 Agent），用于 SubAgent 分隔
     */
    private static String currentAgentName = null;

    /**
     * 从事件中提取 SubAgentSource。
     */
    private static SubAgentSource extractSource(AgentStreamEvent event) {
        return switch (event) {
            case AgentStreamEvent.Text e -> e.source();
            case AgentStreamEvent.Thinking e -> e.source();
            case AgentStreamEvent.ToolStart e -> e.source();
            case AgentStreamEvent.ToolEnd e -> e.source();
            case AgentStreamEvent.Complete e -> e.source();
            default -> null;
        };
    }

    /**
     * 格式化打印单个流式事件。
     * <p>
     * - SubAgent 来源切换时打印 ━━━ [agent名] ━━━ 分隔线
     * - Thinking 事件只在进入思考时打印一次标记，避免每个 chunk 重复前缀
     */
    public static void printEvent(AgentStreamEvent event) {
        // SubAgent 来源切换检测
        SubAgentSource source = extractSource(event);
        String agentName = source != null ? source.agentName() : null;
        if (!java.util.Objects.equals(agentName, currentAgentName)) {
            if (currentAgentName != null) System.out.println();
            currentAgentName = agentName;
            lastEventWasThinking = false;
            String label = agentName != null ? agentName : "main";
            System.out.println("\n━━━ [" + label + "] ━━━");
        }

        switch (event) {
            case AgentStreamEvent.Thinking t -> {
                if (!lastEventWasThinking) {
                    System.out.print("\n[Thinking] ");
                }
                System.out.print(t.content());
                lastEventWasThinking = true;
            }
            case AgentStreamEvent.Text t -> {
                if (lastEventWasThinking) {
                    System.out.println("\n---");
                    System.out.println("\n最终答案：");
                }
                System.out.print(t.content());
                lastEventWasThinking = false;
            }
            case AgentStreamEvent.ToolStart ts ->
                    System.out.println("\n[ToolStart] " + ts.toolName() + " id=" + ts.toolCallId());
            case AgentStreamEvent.ToolEnd te -> System.out.println("[ToolEnd] " + te.toolName() + " result="
                    + (te.result() != null && te.result().length() > 80
                    ? te.result().substring(0, 80) + "..."
                    : te.result()));
            case AgentStreamEvent.Error e ->
                    System.out.println("\n[Error:" + e.code().code() + "] " + e.message() + "\n  " + e.detail());
            case AgentStreamEvent.Complete c -> System.out.println("\n[Complete]：promptTokens：" + c.totalPromptTokens() + "，completionTokens：" + c.totalCompletionTokens());
            case AgentStreamEvent.Paused p -> System.out.println("[Paused]");
        }
    }

    /**
     * 收集流式事件，实时打印所有事件，返回 PauseState（如果暂停）。
     */
    public static PauseState collectStreamEvents(reactor.core.publisher.Flux<AgentStreamEvent> flux) {
        PauseState[] holder = new PauseState[1];
        flux.doOnNext(event -> {
            if (event instanceof AgentStreamEvent.Paused p) {
                holder[0] = p.state();
            } else {
                printEvent(event);
            }
        }).blockLast();
        return holder[0];
    }
}
