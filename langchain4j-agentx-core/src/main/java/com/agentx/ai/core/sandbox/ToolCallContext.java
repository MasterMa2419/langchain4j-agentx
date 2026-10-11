package com.agentx.ai.core.sandbox;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 工具执行上下文 — 替代 1.x 中 Spring AI ToolContext 的框架内上下文传递机制。
 *
 * <p>引擎在每次工具执行前通过 {@link #set} 注入当前调用的运行时数据，
 * 工具通过 {@link #executionBackend()} 等方法读取；执行结束后引擎调用 {@link #clear()}。
 *
 * <p>当前传递的数据：
 * <ul>
 *   <li>{@code executionBackend} — 沙箱执行后端（{@link ExecutionBackend}），
 *       由 ToolCallExecutor 写入，{@code BashTool} / {@code FileSystemTools} / {@code GrepTool} 读取</li>
 * </ul>
 *
 * <p>注意：工具执行统一在 {@code boundedElastic} 调度器上执行（同步路径在调用线程），
 * ThreadLocal 的 set/clear 与单次工具执行在同一调用栈上完成，无跨线程传递需求。
 *
 * @author bigchui
 */
public final class ToolCallContext {

    private static final Logger log = LoggerFactory.getLogger(ToolCallContext.class);

    /** ThreadLocal 上下文槽 */
    private static final ThreadLocal<ToolCallContext> CONTEXT = new ThreadLocal<>();

    private final ExecutionBackend executionBackend;

    private ToolCallContext(ExecutionBackend executionBackend) {
        this.executionBackend = executionBackend;
    }

    /** 创建不含沙箱后端的空上下文 */
    public static ToolCallContext empty() {
        return new ToolCallContext(null);
    }

    /** 创建含沙箱后端的上下文 */
    public static ToolCallContext of(ExecutionBackend executionBackend) {
        return new ToolCallContext(executionBackend);
    }

    /** 绑定到当前线程。引擎在工具执行前调用。 */
    public static void set(ToolCallContext context) {
        CONTEXT.set(context);
    }

    /** 获取当前线程绑定的上下文，未绑定时返回空上下文。 */
    public static ToolCallContext current() {
        ToolCallContext ctx = CONTEXT.get();
        return ctx != null ? ctx : empty();
    }

    /** 清除绑定。引擎在工具执行后 finally 中调用。 */
    public static void clear() {
        CONTEXT.remove();
    }

    /**
     * 提取沙箱执行后端。
     *
     * @return 后端实例；不存在或类型不匹配时返回 {@code null}
     */
    public static ExecutionBackend executionBackend() {
        return current().executionBackend;
    }

    /**
     * 便捷方法：提取当前线程绑定的沙箱执行后端（替代 1.x SandboxToolContexts.extract(toolContext)）。
     */
    public static ExecutionBackend extractExecutionBackend() {
        return executionBackend();
    }
}
