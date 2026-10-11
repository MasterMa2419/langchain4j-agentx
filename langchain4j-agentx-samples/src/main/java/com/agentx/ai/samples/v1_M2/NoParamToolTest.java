package com.agentx.ai.samples.v1_M2;

import com.agentx.ai.core.agent.ReactAgent;
import com.agentx.ai.core.model.AgentStreamEvent;
import com.agentx.ai.core.model.RunnableParams;
import com.agentx.ai.samples.TestConfig;
import dev.langchain4j.mcp.McpToolProvider;
import dev.langchain4j.mcp.client.DefaultMcpClient;
import dev.langchain4j.mcp.client.McpClient;
import dev.langchain4j.mcp.client.transport.http.StreamableHttpMcpTransport;
import dev.langchain4j.model.chat.ChatModel;
import com.agentx.ai.core.model.AgentTool;
import dev.langchain4j.service.tool.ToolProviderResult;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;

/**
 * 无参数工具调用测试。
 *
 * <p>验证 ReactAgent 能否正确处理无参数的 @Tool 方法（如 getSystemInfo）。
 *
 * @author bigchui
 *
 */
public class NoParamToolTest {

    /**
     * 测试 1：流式 — 无参数工具调用。
     */
    public static void testStreamNoParamTool() {
        TestConfig.printTestHeader("测试 1：流式 — 无参数工具调用");

        ChatModel chatModel = TestConfig.createChatModel();
        List<AgentTool> tools = AgentTool.fromAnnotated(new TestTools());

        ReactAgent agent = ReactAgent.builder()
                .chatModel(chatModel)
                .tools(tools)
                .maxRounds(3)
                .build();

        String query = "获取当前系统信息";
        System.out.println("Q: " + query);
        System.out.println("--- Events Start ---");

        List<AgentStreamEvent> events = new ArrayList<>();
        agent.streamForResult(query, RunnableParams.empty())
                .doOnNext(event -> {
                    TestConfig.printEvent(event);
                })
                .doOnError(err -> System.err.println("Error: " + err.getMessage()))
                .blockLast();

        System.out.println("--- Events End ---");
    }

    /**
     * 测试 2：流式 — 调用streamable mcp无参数工具。
     */
    public static void testCallNoParamTool() {
        TestConfig.printTestHeader("测试 2：流式 — 调用streamable mcp无参数工具");

        McpClient streamableClient = new DefaultMcpClient.Builder()
                .key("streamable-client")
                .clientName("streamable-client")
                .clientVersion("1.0")
                .toolExecutionTimeout(Duration.ofSeconds(10))
                .transport(new StreamableHttpMcpTransport.Builder()
                        .url("http://127.0.0.1:8004/api/mcp")
                        .timeout(Duration.ofSeconds(10))
                        .build())
                .build();

        // McpToolProvider 提供工具清单与执行器，转成 AgentTool 交给引擎托管执行
        McpToolProvider toolProvider = McpToolProvider.builder()
                .mcpClients(List.of(streamableClient))
                .build();
        ToolProviderResult provided = toolProvider.provideTools(null);
        List<AgentTool> tools = new ArrayList<>();
        provided.tools().forEach((specification, executor) ->
                tools.add(new AgentTool(specification, executor)));

        AgentTool[] callbacks = tools.toArray(new AgentTool[0]);

        ChatModel chatModel = TestConfig.createZhiPuChatModel();

        ReactAgent agent = ReactAgent.builder()
                .chatModel(chatModel)
                .tools(callbacks)
                .maxRounds(3)
                .build();

        String query = "帮我查一下商品的总数量";
        System.out.println("Q: " + query);

        List<AgentStreamEvent> events = new ArrayList<>();
        agent.streamForResult(query, RunnableParams.empty())
                .doOnNext(event -> {
                    TestConfig.printEvent(event);
                })
                .doOnError(err -> System.err.println("Error: " + err.getMessage()))
                .blockLast();
        System.out.println("--- Events End ---");
    }

    public static void main(String[] args) {
        System.out.println("===============================================");
        System.out.println("       No-Param Tool Call Test");
        System.out.println("===============================================");

        int testNumber = 2;

        switch (testNumber) {
            case 1 -> testStreamNoParamTool();
            case 2 -> testCallNoParamTool();
            default -> System.out.println("无效的测试编号: " + testNumber);
        }

        System.out.println("\n===============================================");
        System.out.println("       Test Completed");
        System.out.println("===============================================");
    }
}
