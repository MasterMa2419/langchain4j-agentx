package com.agentx.ai.samples.v1_1;

import com.agentx.ai.core.agent.ChatAgent;
import com.agentx.ai.core.chatmodels.AgentModels;
import com.agentx.ai.core.model.AgentStreamEvent;
import com.agentx.ai.core.model.RunnableParams;
import com.agentx.ai.samples.TestConfig;
import dev.langchain4j.model.chat.ChatModel;
import org.springframework.jdbc.core.JdbcTemplate;

import javax.sql.DataSource;
import java.util.List;
import java.util.Map;

/**
 * 普通 Agent（ChatAgent）单轮对话测试。
 * <p>
 * 验证点：
 * 1. 不注册工具、不走 ReAct 循环：一次 LLM 调用直接出答案
 * 2. 同一 conversationId 跨多次调用拥有会话记忆（传 DataSource 落库）
 * 3. agentx_conversation / agentx_session 正常落库，与 ReactAgent 复用同一套存储
 *
 * <p>
 * 适用场景：问答、客服、翻译、摘要等"输入 - 回复"对话；需要工具调用、
 * HITL、上下文压缩、SubAgent 时请改用 {@link com.agentx.ai.core.agent.ReactAgent}。
 *
 * @author bigchui
 */
public class ChatAgentSessionTest {

    private static ChatAgent buildAgent() {
        ChatModel chatModel = TestConfig.createChatModel();
        DataSource dataSource = TestConfig.createMySqlDataSource();
        return ChatAgent.builder()
                .models(AgentModels.of(chatModel))
                .dataSource(dataSource)
                .instructions("你是一个简洁友好的中文助手，回答控制在 100 字以内。")
                .maxRetries(3)
                .build();
    }

    /**
     * 验证 agentx_conversation 与 agentx_session 的累积情况（与 ReactAgent 同一套表）。
     */
    private static void verifyAccumulated(DataSource dataSource, String convId, int expectedTurns) {
        JdbcTemplate jdbc = new JdbcTemplate(dataSource);
        System.out.println("\n--- 落库验证 convId=" + convId + " ---");

        List<Map<String, Object>> convRows = jdbc.queryForList(
                "SELECT session_id, status, LEFT(question, 40) AS question "
                        + "FROM agentx_conversation WHERE conversation_id = ? ORDER BY created_at",
                convId);
        System.out.println("[agentx_conversation] 共 " + convRows.size() + " 行 (预期 " + expectedTurns + ")");
        convRows.forEach(row -> System.out.println("  " + row));

        List<Map<String, Object>> msgRows = jdbc.queryForList(
                "SELECT state_key, COUNT(*) AS cnt FROM agentx_session "
                        + "WHERE conversation_id = ? GROUP BY state_key",
                convId);
        System.out.println("[agentx_session] 状态键分布:");
        msgRows.forEach(row -> System.out.println("  " + row));
    }

    public static void main(String[] args) {
        System.out.println("===============================================");
        System.out.println("  ChatAgent (普通单轮对话) Test");
        System.out.println("===============================================");
        System.out.println("ChatModel: " + TestConfig.CHAT_MODEL);
        System.out.println("===============================================\n");

        ChatAgent agent = buildAgent();
        String convId = TestConfig.randomConvId();
        String userId = TestConfig.randomUserId("user_chat");
        RunnableParams params = TestConfig.buildParams(convId, userId);
        System.out.println("convId = " + convId + "\n");

        // Turn 1：普通同步调用（不传 params，无会话记忆）
        System.out.println("--- 单次同步调用 ---");
        String answer = agent.call("用一句话介绍你自己");
        System.out.println("A: " + answer + "\n");

        // Turn 2 / 3：带会话的流式调用，第二轮应记住第一轮内容
        TestConfig.streamAndPrint(agent, "简单介绍下北京", params);
        TestConfig.streamAndPrint(agent, "上面我让你介绍的是哪个城市？", params);

        // 流式事件级输出（Thinking / Text / Complete）
        System.out.println("--- 流式事件输出 ---");
        agent.streamForResult("你是普通 Agent 还是 ReAct Agent？", params)
                .doOnNext(TestConfig::printEvent)
                .doOnError(err -> System.err.println("Stream Error: " + err.getMessage()))
                .blockLast();

        verifyAccumulated(TestConfig.createMySqlDataSource(), convId, 4);
    }
}
