package com.agentx.ai.core.agent.internal;

import dev.langchain4j.agent.tool.ToolExecutionRequest;

import java.util.ArrayList;
import java.util.List;

/**
 * 流式轮次执行状态。
 * 从 AgentLoopExecutor 中提取为独立类，供 ThinkingModeProcessor 等协作类共享。
 *
 * <p>2.0 变更：LC4j 流结束时由最终 AiMessage 提供完整 toolExecutionRequests，
 * 无需逐分片合并；advisorContext 字段随 Advisor 机制移除而删除，
 * requestJson 由引擎在发起调用前直接序列化。
 */
public class RoundState {
    RoundMode mode = RoundMode.TEXT;
    final List<ToolExecutionRequest> toolCalls = new ArrayList<>();
    final StringBuilder textBuffer = new StringBuilder();
    final StringBuilder reasoningBuffer = new StringBuilder();
    boolean inThink = false;
    long promptTokens = -1;
    long completionTokens = -1;
    String finishReason;
    String requestJson;
}
