package com.agentx.ai.samples.v1_1;

import com.agentx.ai.core.agent.ChatAgent;
import com.agentx.ai.core.chatmodels.AgentModels;
import com.agentx.ai.core.model.AgentTool;
import com.agentx.ai.core.model.RunnableParams;
import com.agentx.ai.samples.TestConfig;
import dev.langchain4j.model.chat.ChatModel;
import org.springframework.jdbc.core.JdbcTemplate;

import javax.sql.DataSource;
import java.util.List;
import java.util.Map;

/**
 * 普通 Agent（ChatAgent）基础工具调用测试（有界工具轮次）。
 * <p>
 * 验证点：
 * 1. 注册工具后，模型发起 tool call → 框架执行 → 回填结果 → 收敛最终答案
 * 2. 默认 maxToolRounds=1：最多 2 次 LLM 调用，agentx_trace 两轮（工具轮 + 最终轮）
 * 3. 工具调用 AiMessage 与 ToolExecutionResultMessage 写入 agentx_session，可跨轮续聊
 *
 * <p>
 * 适用场景：问答、客服等"查一次再回答"的浅层工具场景；需要多轮推理、HITL、
 * ToolSearch/MCP、上下文压缩、沙箱、子代理委派时请改用
 * {@link com.agentx.ai.core.agent.ReactAgent}。
 *
 * @author bigchui
 */
public class ChatAgentToolTest {

    private static ChatAgent buildAgent(DataSource dataSource, int maxToolRounds) {
        ChatModel chatModel = TestConfig.createChatModel();
        List<AgentTool> tools = AgentTool.fromAnnotated(new SessionTools());
        return ChatAgent.builder()
                .models(AgentModels.of(chatModel))
                .dataSource(dataSource)
                .instructions("你是一个简洁友好的中文助手。需要事实信息时先调用工具，再基于工具结果回答。")
                .tools(tools)
                .maxToolRounds(maxToolRounds)
                .maxRetries(3)
                .build();
    }

    /**
     * 验证 trace 逐轮记录与 session 消息链是否包含工具调用。
     */
    private static void verify(DataSource dataSource, String convId) {
        JdbcTemplate jdbc = new JdbcTemplate(dataSource);
        System.out.println("\n--- 落库验证 convId=" + convId + " ---");

        List<Map<String, Object>> traces = jdbc.queryForList(
                "SELECT round, LEFT(output_data, 60) AS output_data "
                        + "FROM agentx_trace WHERE conversation_id = ? ORDER BY round",
                convId);
        System.out.println("[agentx_trace] 共 " + traces.size() + " 行（工具轮 + 最终轮）");
        traces.forEach(row -> System.out.println("  " + row));

        List<Map<String, Object>> toolMsgs = jdbc.queryForList(
                "SELECT state_key, COUNT(*) AS cnt FROM agentx_session "
                        + "WHERE conversation_id = ? AND state_data LIKE '%tool%' "
                        + "GROUP BY state_key",
                convId);
        System.out.println("[agentx_session] 含工具内容的消息分布:");
        toolMsgs.forEach(row -> System.out.println("  " + row));
    }

    public static void main(String[] args) {
        System.out.println("===============================================");
        System.out.println("  ChatAgent Tool Test (普通 Agent + 基础工具)");
        System.out.println("===============================================");
        System.out.println("ChatModel: " + TestConfig.CHAT_MODEL);
        System.out.println("===============================================\n");

        DataSource dataSource = TestConfig.createMySqlDataSource();
        // 默认 1 轮工具：一次工具批次 + 一次最终答案
        ChatAgent agent = buildAgent(dataSource, 1);

        String convId = TestConfig.randomConvId();
        String userId = TestConfig.randomUserId("user_chat_tool");
        RunnableParams params = TestConfig.buildParams(convId, userId);
        System.out.println("convId = " + convId + "\n");

        // 触发工具调用：模型应调用 getWeather("北京")
        TestConfig.streamAndPrint(agent, "北京今天天气怎么样？", params);

        // 会话记忆：第二轮应记得上一轮的对话（工具消息完整入链）
        TestConfig.streamAndPrint(agent, "那上海的天气呢？", params);

        verify(dataSource, convId);
    }
}
