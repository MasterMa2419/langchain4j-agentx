package com.agentx.ai.core.tools;

import com.agentx.ai.core.agent.ReactAgent;
import com.agentx.ai.core.model.AgentResult;
import com.agentx.ai.core.model.AgentStreamEvent;
import com.agentx.ai.core.model.AgentTool;
import com.agentx.ai.core.model.RunnableParams;
import com.agentx.ai.core.model.SubAgentSource;
import com.baomidou.mybatisplus.core.toolkit.IdWorker;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.service.tool.ToolExecutor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import reactor.core.publisher.Sinks;

import java.util.function.Supplier;

/**
 * SubAgent 工具 — 将 ReactAgent 包装为 {@link AgentTool}。
 *
 * <p>核心职责：
 * <ul>
 *   <li>接收主 Agent LLM 的委派请求</li>
 *   <li>通过 Supplier 创建独立的子 ReactAgent 实例</li>
 *   <li>透传 RunnableParams，保持调用上下文一致</li>
 *   <li>流式模式：子 Agent stream，事件打 source 标签转发到父 Sink</li>
 *   <li>非流式模式：子 Agent call，直接返回结果</li>
 * </ul>
 *
 * <p>子 Agent 约束（Phase 1）：
 * <ul>
 *   <li>只能使用普通工具，禁止嵌套 SubAgentTool</li>
 *   <li>禁止使用 AskUserTool</li>
 * </ul>
 *
 * @author bigchui
 */
public class SubAgentTool {

    private static final Logger log = LoggerFactory.getLogger(SubAgentTool.class);
    private static final ObjectMapper objectMapper = new ObjectMapper();

    /**
     * 创建 SubAgentTool 的 AgentTool。
     *
     * @param agentName     子 Agent 名称（工具名称自动生成为 call_{name}）
     * @param description   工具描述（LLM 根据此描述决定是否委派）
     * @param agentProvider 子 Agent 工厂（每次调用创建新实例）
     * @return AgentTool
     */
    public static AgentTool create(String agentName, String description,
                                   Supplier<ReactAgent> agentProvider) {
        String toolName = "call_" + agentName;
        String toolDescription = description + "。将任务委派给此子Agent处理，它将返回处理结果。";

        ToolSpecification spec = ToolSpecification.builder()
                .name(toolName)
                .description(toolDescription)
                .build();
        ToolExecutor executor = (request, memoryId) ->
                execute(agentName, agentProvider, extractMessage(request));
        return new AgentTool(spec, executor);
    }

    private static String extractMessage(ToolExecutionRequest request) {
        String args = request.arguments();
        if (args == null || args.isBlank()) {
            return "";
        }
        try {
            JsonNode node = objectMapper.readTree(args);
            if (node.has("message")) {
                return node.get("message").asText("");
            }
            return args;
        } catch (Exception e) {
            // 非 JSON 参数，按原文处理
            return args;
        }
    }

    /**
     * 执行子 Agent。
     * <p>
     * 通过 {@link ContextHolder} 判断父 Agent 的执行模式：
     * - sink 不为 null：流式模式，子 Agent 也流式执行，事件转发
     * - sink 为 null：非流式模式，子 Agent 也非流式执行
     */
    private static String execute(String agentName, Supplier<ReactAgent> agentProvider,
                                   String message) {
        ReactAgent subAgent = agentProvider.get();
        assertAllowed(subAgent, agentName);

        Sinks.Many<AgentStreamEvent> parentSink = ContextHolder.getSink();
        RunnableParams params = ContextHolder.getParams();

        if (params == null) {
            params = RunnableParams.empty();
        }

        if (parentSink != null) {
            return executeStreaming(agentName, subAgent, message, params, parentSink);
        } else {
            return executeBlocking(subAgent, message, params);
        }
    }

    /**
     * 流式执行：子 Agent stream，事件打 source 标签转发到父 Sink。
     */
    private static String executeStreaming(String agentName, ReactAgent subAgent, String message,
                                            RunnableParams params, Sinks.Many<AgentStreamEvent> parentSink) {
        String subAgentId = "sa_" + IdWorker.getIdStr();
        SubAgentSource source = new SubAgentSource(agentName, subAgentId);

        log.debug("SubAgent[{}] streaming execution started, subAgentId={}", agentName, subAgentId);

        StringBuilder result = new StringBuilder();
        subAgent.streamForResult(message, params)
                .map(event -> withSource(event, source))
                .doOnNext(parentSink::tryEmitNext)
                .filter(e -> e instanceof AgentStreamEvent.Text)
                .map(e -> ((AgentStreamEvent.Text) e).content())
                .doOnNext(result::append)
                .blockLast();

        log.debug("SubAgent[{}] streaming execution completed, subAgentId={}", agentName, subAgentId);
        return result.toString();
    }

    /**
     * 非流式执行：子 Agent call，直接返回结果。
     */
    private static String executeBlocking(ReactAgent subAgent, String message,
                                           RunnableParams params) {
        log.debug("SubAgent blocking execution started");

        AgentResult agentResult = subAgent.callForResult(message, params);
        if (agentResult instanceof AgentResult.Completed c) {
            return c.answer();
        }
        if (agentResult instanceof AgentResult.Failed f) {
            return "SubAgent 执行失败：" + f.error();
        }
        return "SubAgent 执行异常：未知结果类型";
    }

    /**
     * 给子 Agent 的事件打上 source 标签。
     */
    private static AgentStreamEvent withSource(AgentStreamEvent event, SubAgentSource source) {
        return switch (event) {
            case AgentStreamEvent.Thinking e -> new AgentStreamEvent.Thinking(e.content(), source);
            case AgentStreamEvent.Text e -> new AgentStreamEvent.Text(e.content(), source);
            case AgentStreamEvent.ToolStart e -> new AgentStreamEvent.ToolStart(e.toolName(), e.toolCallId(), e.arguments(), source);
            case AgentStreamEvent.ToolEnd e -> new AgentStreamEvent.ToolEnd(e.toolName(), e.toolCallId(), e.result(), source);
            case AgentStreamEvent.Error e -> new AgentStreamEvent.Error(e.code(), e.message(), e.detail(), source);
            case AgentStreamEvent.Complete e -> new AgentStreamEvent.Complete(
                    e.totalPromptTokens(), e.totalCompletionTokens(), source);
            default -> event;
        };
    }

    /**
     * 合规检查：验证子 Agent 配置符合约束。
     */
    private static void assertAllowed(ReactAgent agent, String agentName) {
        for (AgentTool tool : agent.getTools()) {
            String toolDefName = tool.getToolDefinition().name();
            if (toolDefName.startsWith("call_")) {
                throw new IllegalArgumentException(
                        "SubAgent[" + agentName + "] 不允许嵌套使用 SubAgentTool");
            }
            if ("AskUser".equals(toolDefName) || "ask_user".equals(toolDefName)) {
                throw new IllegalArgumentException(
                        "SubAgent[" + agentName + "] 不允许使用 AskUser 工具");
            }
        }
    }

    // ==================== 上下文传递 ====================

    /**
     * 线程局部变量，用于在工具执行链中传递父 Agent 的 Sink 和 Params。
     * <p>
     * 由 ToolCallExecutor 在工具执行前设置（流式路径），SubAgentTool 读取。
     */
    public static class ContextHolder {
        private static final ThreadLocal<Sinks.Many<AgentStreamEvent>> SINK = new ThreadLocal<>();
        private static final ThreadLocal<RunnableParams> PARAMS = new ThreadLocal<>();

        public static void set(Sinks.Many<AgentStreamEvent> sink, RunnableParams params) {
            SINK.set(sink);
            PARAMS.set(params);
        }

        public static Sinks.Many<AgentStreamEvent> getSink() {
            return SINK.get();
        }

        public static RunnableParams getParams() {
            return PARAMS.get();
        }

        public static void clear() {
            SINK.remove();
            PARAMS.remove();
        }
    }

    // ==================== 请求参数 ====================

    /**
     * SubAgent 工具的输入参数。
     */
    public record SubAgentRequest(
            String message
    ) {}
}
