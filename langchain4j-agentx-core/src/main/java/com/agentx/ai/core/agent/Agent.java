package com.agentx.ai.core.agent;

import com.agentx.ai.core.model.AgentResult;
import com.agentx.ai.core.model.AgentStreamEvent;
import com.agentx.ai.core.model.AgentTool;
import com.agentx.ai.core.model.RunnableParams;
import com.agentx.ai.core.trace.TraceStore;
import reactor.core.publisher.Flux;

import java.util.List;

/**
 * Agent 统一抽象 — 框架内所有 Agent 类型的公共接口。
 * <p>
 * 当前提供两种实现：
 * <ul>
 *   <li>{@link ReactAgent} - ReAct 多轮循环 Agent：Reasoning + Acting 驱动，
 *       支持工具调用、HITL、中断恢复、上下文压缩、SubAgent 等完整能力，适合复杂任务</li>
 *   <li>{@link ChatAgent} - 单轮对话 Agent：一次 LLM 调用直接出答案，
 *       复用会话记忆 / 流式输出 / 持久化 / Hook，适合问答、客服等不需要工具循环的场景</li>
 * </ul>
 * <p>
 * 业务代码、RAG 工具、前端展示层等面向本接口编程，即可在同一栈内同时驱动两种 Agent。
 *
 * @author bigchui
 */
public interface Agent {

    /**
     * Agent 名称。作为 SubAgent 注册时必填；普通使用可为 null。
     */
    String getName();

    /**
     * Agent 描述。作为 SubAgent 注册时用于生成工具描述；普通使用可为 null。
     */
    String getDescription();

    /**
     * 同步调用 Agent。
     *
     * @param query 用户消息
     * @return Agent 响应文本
     */
    String call(String query);

    /**
     * 同步调用 Agent（带参数）。
     *
     * @param query  用户消息
     * @param params 调用参数（conversationId / userId / customParams / toolParams / outputType）
     * @return Agent 响应文本
     */
    String call(String query, RunnableParams params);

    /**
     * 同步调用 Agent，返回完整结果（{@link AgentResult.Completed} 等）。
     *
     * @param query  用户消息
     * @param params 调用参数
     * @return AgentResult
     */
    AgentResult callForResult(String query, RunnableParams params);

    /**
     * 流式调用 Agent。
     *
     * @param query 用户消息
     * @return 文本流
     */
    Flux<String> stream(String query);

    /**
     * 流式调用 Agent（带参数）。
     *
     * @param query  用户消息
     * @param params 调用参数
     * @return 文本流（过滤掉暂停等非文本事件）
     */
    Flux<String> stream(String query, RunnableParams params);

    /**
     * 流式调用 Agent，返回完整事件流。
     *
     * @param query  用户消息
     * @param params 调用参数
     * @return AgentStreamEvent 流
     */
    Flux<AgentStreamEvent> streamForResult(String query, RunnableParams params);

    /**
     * 本 Agent 的能力清单（工具定义）。
     * <p>用于 SubAgent 委派前的合规校验（禁止嵌套 SubAgent / AskUser）与运行时内省。
     * 不支持工具的 Agent 实现（如 {@link ChatAgent}）返回空列表。
     */
    List<AgentTool> getTools();

    /**
     * 框架内部使用：本 Agent 被注册为 SubAgent 时，由父 Agent 在委派前调用。
     * <p>实现类应关闭自身会话持久化（避免子代理写父会话边界之外的 agentx_session），
     * 并复用父 Agent 的 TraceStore（受父 enableTrace 控制，null 则不记录）。
     *
     * @param parentTraceStore 父 Agent 的 TraceStore，父未开启 trace 时为 null
     */
    default void configureAsSubAgent(TraceStore parentTraceStore) {
        // 默认无操作
    }

    /**
     * 停止指定会话的流式任务。
     *
     * @param conversationId 会话 ID
     * @return 是否成功停止
     */
    default boolean stopStream(String conversationId) {
        return false;
    }

    /**
     * 检查指定会话是否有正在运行的任务。
     *
     * @param conversationId 会话 ID
     * @return 是否有正在运行的任务
     */
    default boolean hasRunningTask(String conversationId) {
        return false;
    }
}
