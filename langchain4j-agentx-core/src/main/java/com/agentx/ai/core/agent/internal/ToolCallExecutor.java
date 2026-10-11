package com.agentx.ai.core.agent.internal;

import com.agentx.ai.core.hook.HookManager;
import com.agentx.ai.core.hook.AfterToolExecutionEvent;
import com.agentx.ai.core.hook.BeforeToolExecutionEvent;
import com.agentx.ai.core.model.AgentStreamEvent;
import com.agentx.ai.core.model.AgentTool;
import com.agentx.ai.core.model.PendingToolCall;
import com.agentx.ai.core.model.RunnableParams;
import com.agentx.ai.core.sandbox.SandboxToolContexts;
import com.agentx.ai.core.sandbox.ToolCallContext;
import com.agentx.ai.core.sandbox.UnavailableExecutionBackend;
import com.agentx.ai.core.stage.AgentRuntimeContext;
import com.agentx.ai.core.tools.SubAgentTool;
import com.alibaba.fastjson2.JSON;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import reactor.core.publisher.Sinks;
import reactor.core.scheduler.Schedulers;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * 工具调用执行器 — 负责工具执行、结果收集、消息组装。
 *
 * <ul>
 *   <li>单个工具执行（含参数替换和错误处理）</li>
 *   <li>异步批量工具执行（保证顺序）</li>
 *   <li>暂停/恢复时的工具结果解析</li>
 *   <li>工具调用消息组装（LC4j：每个工具结果一条 {@link ToolExecutionResultMessage}）</li>
 * </ul>
 *
 * <p>2.0 变更：工具运行时上下文（沙箱后端、SubAgent 父 Sink/Params）通过
 * {@link ToolCallContext} 与 {@link SubAgentTool.ContextHolder} ThreadLocal 传递，
 * 替代 1.x 的 Spring AI ToolContext。
 *
 * @author bigchui
 */
public class ToolCallExecutor {

    private static final Logger log = LoggerFactory.getLogger(ToolCallExecutor.class);

    private static final Set<String> APPROVAL_KEYWORDS = Set.of(
            "ok", "yes", "y", "好", "好的", "确认", "同意", "是", "是的", "approve", "confirm");

    private final Map<String, AgentTool> toolMap;
    private final ObjectMapper objectMapper;
    private final String askUserToolName;
    private final HookManager hookManager;

    public ToolCallExecutor(Map<String, AgentTool> toolMap, ObjectMapper objectMapper,
                            String askUserToolName, HookManager hookManager) {
        this.toolMap = toolMap;
        this.objectMapper = objectMapper;
        this.askUserToolName = askUserToolName;
        this.hookManager = hookManager;
    }

    /**
     * 执行单个工具调用（非流式路径）。
     */
    public ToolExecutionResult executeSingleTool(ToolExecutionRequest toolCall, RunnableParams params) {
        return executeSingleTool(toolCall, params, null, null);
    }

    /**
     * 执行单个工具调用。
     *
     * @param toolCall   工具调用信息
     * @param params     调用参数
     * @param sink       事件 Sink（流式路径不为 null，非流式路径为 null）
     * @param runtimeCtx 当前调用的运行时上下文（null 时不触发 Hook）
     */
    public ToolExecutionResult executeSingleTool(ToolExecutionRequest toolCall, RunnableParams params,
                                                  Sinks.Many<AgentStreamEvent> sink,
                                                  AgentRuntimeContext runtimeCtx) {
        String toolName = toolCall.name();
        String argsJson = toolCall.arguments();

        // 无参数工具：LLM 可能返回 null 或空字符串，默认为空 JSON 对象
        if (argsJson == null || argsJson.isBlank()) {
            argsJson = "{}";
        }

        AgentTool tool = toolMap.get(toolName);

        if (tool == null) {
            return errorResult(toolName, "不存在名为 '" + toolName + "' 的工具");
        }

        // 先按工具的 inputSchema 过滤，只注入该工具真实声明的字段（避免 MCP 服务端严格校验报错）
        argsJson = replaceToolParams(tool, argsJson, params);

        log.debug("Executing tool: {} with args: {}", toolName, argsJson);

        Object result;
        try {
            // 绑定工具执行上下文（沙箱后端 + SubAgent 父上下文），执行完清理
            ToolCallContext toolContext = buildToolContext(runtimeCtx);
            Sinks.Many<AgentStreamEvent> subAgentSink = sink;
            RunnableParams subAgentParams = params;
            if (sink != null) {
                SubAgentTool.ContextHolder.set(sink, params);
            }

            String effectiveArgs = argsJson;
            try {
                if (runtimeCtx != null && !hookManager.isEmpty()) {
                    BeforeToolExecutionEvent event = new BeforeToolExecutionEvent(
                            runtimeCtx, toolName, toolCall.id(), argsJson);
                    BeforeToolExecutionEvent processed = hookManager.fireEvent(event);
                    effectiveArgs = processed.getArguments();
                }

                // Hook 处理后、工具执行前发射 ToolStart（携带 post-hook 入参，与工具真实执行一致）
                if (sink != null) {
                    sink.tryEmitNext(new AgentStreamEvent.ToolStart(toolName, toolCall.id(), effectiveArgs));
                }

                ToolCallContext.set(toolContext);
                result = tool.getToolExecutor().execute(
                        ToolExecutionRequest.builder()
                                .id(toolCall.id())
                                .name(toolName)
                                .arguments(effectiveArgs)
                                .build(),
                        null);
            } finally {
                ToolCallContext.clear();
                if (sink != null) {
                    SubAgentTool.ContextHolder.clear();
                }
            }
        } catch (Exception e) {
            log.error("Tool '{}' execution failed: {}", toolName, e.getMessage(), e);
            String hint = buildErrorHint(toolName, e);
            return errorResult(toolName, hint);
        }

        String rawResult = result != null ? result.toString() : "{}";
        return new ToolExecutionResult(rawResult);
    }

    /**
     * 执行非拦截的工具调用（拦截的由 resume 处理）。
     */
    public void executeNonPendingTools(List<ToolExecutionRequest> allToolCalls,
                                       List<PendingToolCall> pending,
                                       List<ChatMessage> messages,
                                       RunnableParams params,
                                       AgentRuntimeContext runtimeCtx) {
        Set<String> pendingIds = new HashSet<>();
        for (PendingToolCall ptc : pending) {
            pendingIds.add(ptc.id());
        }

        for (ToolExecutionRequest tc : allToolCalls) {
            if (!pendingIds.contains(tc.id())) {
                ToolExecutionResult result = executeSingleTool(tc, params, null, runtimeCtx);
                List<ChatMessage> collected = collectToolCallMessages(tc, result);
                messages.addAll(collected);
                if (runtimeCtx != null) {
                    runtimeCtx.appendOriginalMessages(collected);
                }
            }
        }
    }

    /**
     * 异步执行工具调用（保证顺序）。
     * 多个工具并发执行，但结果按原始 toolCalls 顺序添加到 messages。
     */
    public void executeToolCallsAsync(Sinks.Many<AgentStreamEvent> sink,
                                      List<ToolExecutionRequest> toolCalls,
                                      List<ChatMessage> messages,
                                      RunnableParams params,
                                      AgentRuntimeContext runtimeCtx,
                                      Runnable onComplete) {
        int total = toolCalls.size();
        AtomicInteger completedCount = new AtomicInteger(0);
        List<List<ChatMessage>> results = new ArrayList<>(total);
        for (int i = 0; i < total; i++) results.add(null);
        List<ToolExecDetail> execDetails = new ArrayList<>(total);
        for (int i = 0; i < total; i++) execDetails.add(null);

        for (int i = 0; i < toolCalls.size(); i++) {
            final int index = i;
            ToolExecutionRequest tc = toolCalls.get(i);

            Schedulers.boundedElastic().schedule(() -> {
                try {
                    ToolExecutionResult toolResult = executeSingleTool(tc, params, sink, runtimeCtx);
                    results.set(index, collectToolCallMessages(tc, toolResult));
                    execDetails.set(index, new ToolExecDetail(tc, toolResult.rawResult(), null));
                } catch (Exception ex) {
                    log.error("Unexpected error in tool execution: {} - {}", tc.name(), ex.getMessage());
                    results.set(index, collectToolCallErrorMessages(tc, ex));
                    execDetails.set(index, new ToolExecDetail(tc, null, ex));
                } finally {
                    int completed = completedCount.incrementAndGet();
                    if (completed >= total) {
                        appendResultsInOrder(results, messages);
                        if (runtimeCtx != null) {
                            for (List<ChatMessage> result : results) {
                                runtimeCtx.appendOriginalMessages(result);
                            }
                        }

                        for (ToolExecDetail detail : execDetails) {
                            if (detail.error == null) {
                                sink.tryEmitNext(new AgentStreamEvent.ToolEnd(
                                        detail.toolCall.name(), detail.toolCall.id(), detail.rawResult));

                                if (runtimeCtx != null && !hookManager.isEmpty()) {
                                    hookManager.fireEvent(new AfterToolExecutionEvent(
                                            runtimeCtx,
                                            detail.toolCall.name(),
                                            detail.toolCall.id(),
                                            detail.toolCall.arguments(),
                                            detail.rawResult,
                                            true,
                                            0));
                                }
                            }
                        }

                        onComplete.run();
                    }
                }
            });
        }
    }

    /**
     * 解析 USER_INTERRUPT 中断恢复时的工具结果。
     *
     * <p>与 {@link #resolveResumeToolResult}（HITL 路径）不同：
     * <ul>
     *   <li>已知工具：直接重新执行，拿真实工具结果</li>
     *   <li>外部工具（MCP 等）：注入占位结果，引导 LLM 重新调用</li>
     * </ul>
     */
    public String resolveInterruptToolResult(PendingToolCall ptc,
                                             ToolExecutionRequest toolCall,
                                             RunnableParams params) {
        if (toolMap.containsKey(ptc.name())) {
            ToolExecutionResult result = executeSingleTool(toolCall, params);
            return result.rawResult();
        }
        return "[工具执行被用户中断，请重新调用 " + ptc.name() + " 工具]";
    }

    /**
     * 解析恢复执行时工具调用的结果。
     */
    public String resolveResumeToolResult(PendingToolCall ptc,
                                          ToolExecutionRequest toolCall,
                                          Map<String, String> toolResults,
                                          RunnableParams params) {
        // 用户输入工具：用户回答即工具结果，不需要执行
        if (askUserToolName != null && askUserToolName.equals(ptc.name())) {
            return toolResults.getOrDefault(ptc.id(), "");
        }
        String userResponse = toolResults.getOrDefault(ptc.id(), "");
        // 用户拒绝：不执行工具
        if (!isApproved(userResponse)) {
            return ptc.name() + " 工具被用户拒绝执行：" + userResponse;
        }
        // 用户确认：实际执行
        if (toolMap.containsKey(ptc.name())) {
            ToolExecutionResult result = executeSingleTool(toolCall, params);
            return result.rawResult();
        }
        return userResponse;
    }

    // ==================== 消息组装 ====================

    public void addToolCallMessages(ToolExecutionRequest toolCall,
                                    ToolExecutionResult result,
                                    List<ChatMessage> messages) {
        addNormalToolMessage(toolCall, result.rawResult(), messages);
    }

    public void addNormalToolMessage(ToolExecutionRequest toolCall,
                                     String rawResult, List<ChatMessage> messages) {
        messages.add(ToolExecutionResultMessage.from(
                toolCall.id(), toolCall.name(), rawResult));
    }

    public List<ChatMessage> collectToolCallMessages(ToolExecutionRequest toolCall,
                                                     ToolExecutionResult result) {
        List<ChatMessage> collected = new ArrayList<>();
        addToolCallMessages(toolCall, result, collected);
        return collected;
    }

    public List<ChatMessage> collectToolCallErrorMessages(ToolExecutionRequest toolCall,
                                                          Exception ex) {
        log.error("Unexpected error in tool execution: {} - {}", toolCall.name(), ex.getMessage());
        ToolExecutionResult errorResult = errorResult(toolCall.name(), "内部错误：" + ex.getMessage());
        List<ChatMessage> collected = new ArrayList<>();
        addNormalToolMessage(toolCall, errorResult.rawResult(), collected);
        return collected;
    }

    private void appendResultsInOrder(List<List<ChatMessage>> results, List<ChatMessage> messages) {
        for (List<ChatMessage> result : results) {
            messages.addAll(result);
        }
    }

    // ==================== 参数替换 ====================

    /**
     * 把 RunnableParams.toolParams 合并进 LLM 生成的工具调用 args，**仅注入工具 schema 真实声明的字段**。
     * <p>
     * 三种情况都覆盖：
     * <ul>
     *   <li>LLM 漏掉 {@code userId} → 增量补上（避免 LLM 偶发漏字段导致工具收到 null）</li>
     *   <li>LLM 乱填 {@code "userId":"xxx"} → 覆盖为真值（框架参数不该信 LLM）</li>
     *   <li>LLM 传了旧版占位值 {@code "userId":"default"} → 同样覆盖为真值</li>
     * </ul>
     * <p>
     * <b>白名单过滤</b>：MCP 等远程工具服务端常严格校验多余字段，因此从
     * {@link ToolSpecification} 的 {@link JsonObjectSchema#properties()} 读出合法参数名清单，
     * 只注入该工具声明的字段，未声明的跳过。
     */
    private String replaceToolParams(AgentTool tool, String argsJson, RunnableParams params) {
        if (params == null || params.getToolParams() == null || params.getToolParams().isEmpty()) {
            return argsJson;
        }
        if (argsJson == null || argsJson.isBlank()) {
            return argsJson;
        }

        Set<String> accepted = getAcceptedParamNames(tool);
        if (accepted.isEmpty()) {
            // 拿不到 schema（异常或无 properties）→ 保守起见不注入，避免污染 MCP 严格校验工具
            log.debug("工具 {} 的 schema 无 properties，跳过 toolParams 注入", tool.getName());
            return argsJson;
        }

        try {
            Map<String, Object> args = objectMapper.readValue(argsJson,
                    new TypeReference<Map<String, Object>>() {});
            for (Map.Entry<String, Object> entry : params.getToolParams().entrySet()) {
                if (accepted.contains(entry.getKey())) {
                    args.put(entry.getKey(), entry.getValue());
                } else {
                    log.debug("工具 {} 不接受参数 {}，跳过注入", tool.getName(), entry.getKey());
                }
            }
            return objectMapper.writeValueAsString(args);
        } catch (Exception e) {
            log.error("替换工具参数失败（argsJson 不是合法 JSON，原样返回）: {}", argsJson, e);
            return argsJson;
        }
    }

    /**
     * 从 ToolSpecification 的 parameters 读出合法参数名集合。
     */
    private Set<String> getAcceptedParamNames(AgentTool tool) {
        try {
            JsonObjectSchema schema = tool.getToolDefinition().parameters();
            if (schema == null || schema.properties() == null || schema.properties().isEmpty()) {
                return Set.of();
            }
            Set<String> names = new HashSet<>(schema.properties().size());
            for (String k : schema.properties().keySet()) {
                if (k != null) {
                    names.add(k);
                }
            }
            return names;
        } catch (Exception e) {
            log.warn("解析工具 {} 参数 schema 失败，跳过 toolParams 注入: {}",
                    tool.getName(), e.getMessage());
            return Set.of();
        }
    }

    // ==================== 辅助方法 ====================

    private ToolCallContext buildToolContext(AgentRuntimeContext runtimeCtx) {
        if (runtimeCtx == null) {
            return ToolCallContext.empty();
        }
        if (runtimeCtx.getExecutionBackend() != null) {
            return ToolCallContext.of(runtimeCtx.getExecutionBackend());
        }
        if (runtimeCtx.isSandboxFailed()) {
            // 严格模式下沙箱获取失败：注入占位后端，拒绝降级到宿主机执行
            return ToolCallContext.of(UnavailableExecutionBackend.INSTANCE);
        }
        return ToolCallContext.empty();
    }

    private String buildErrorHint(String toolName, Exception e) {
        if (e.getCause() instanceof JsonProcessingException
                || e instanceof JsonProcessingException) {
            return toolName + " 工具调用失败：参数 JSON 解析错误，可能是参数过长被截断。"
                    + "请尝试缩短参数或拆分为多步操作。";
        }
        return toolName + " 工具调用失败：" + e.getMessage();
    }

    private ToolExecutionResult errorResult(String toolName, String errorMessage) {
        try {
            Map<String, String> errorMap = new HashMap<>();
            errorMap.put("error", errorMessage);
            errorMap.put("tool", toolName);
            return new ToolExecutionResult(objectMapper.writeValueAsString(errorMap));
        } catch (JsonProcessingException ex) {
            return new ToolExecutionResult("{\"error\":\"" + errorMessage + "\"}");
        }
    }

    private boolean isApproved(String userResponse) {
        if (userResponse == null || userResponse.isBlank()) {
            return false;
        }
        return APPROVAL_KEYWORDS.contains(userResponse.trim().toLowerCase());
    }

    /**
     * 校验并修复工具调用参数。
     * 如果某个 tool call 的 arguments 不是合法 JSON，替换为 {@code {}}，避免后续 API 调用 400。
     * 全部合法时原样返回传入列表。
     */
    public List<ToolExecutionRequest> sanitizeToolCalls(List<ToolExecutionRequest> toolCalls) {
        boolean needsFix = false;
        for (ToolExecutionRequest tc : toolCalls) {
            String args = tc.arguments();
            if (args != null && !args.isBlank() && !isValidJson(args)) {
                needsFix = true;
                break;
            }
        }
        if (!needsFix) {
            return toolCalls;
        }

        List<ToolExecutionRequest> fixed = new ArrayList<>(toolCalls.size());
        for (ToolExecutionRequest tc : toolCalls) {
            String args = tc.arguments();
            if (args == null || args.isBlank() || isValidJson(args)) {
                fixed.add(tc);
            } else {
                log.warn("工具 '{}' 的 arguments 不是合法 JSON，已替换为空对象: {}", tc.name(), args);
                fixed.add(ToolExecutionRequest.builder()
                        .id(tc.id())
                        .name(tc.name())
                        .arguments("{}")
                        .build());
            }
        }
        return fixed;
    }

    private boolean isValidJson(String json) {
        try {
            JSON.parse(json);
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    // ==================== 内部记录 ====================

    record ToolExecutionResult(String rawResult) {
    }

    record ToolExecDetail(ToolExecutionRequest toolCall, String rawResult, Exception error) {
    }
}
