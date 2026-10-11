package com.agentx.ai.core.sandbox;

/**
 * 从工具执行上下文提取 {@link ExecutionBackend} 的静态工具方法。
 *
 * <p>所有沙箱感知工具（{@code BashTool}、{@code FileSystemTools}、{@code GrepTool}）通过本类
 * 统一获取执行后端，避免重复的类型检查代码。
 *
 * <p>2.0 起上下文通过 {@link ToolCallContext}（ThreadLocal）传递，
 * 由引擎在工具执行前绑定；1.x 的 Spring AI ToolContext 机制已移除。
 *
 * @author bigchui
 */
public final class SandboxToolContexts {

    private SandboxToolContexts() {
    }

    /**
     * 提取当前线程绑定的 ExecutionBackend。
     *
     * @return 后端实例；不存在时返回 {@code null}
     */
    public static ExecutionBackend extract() {
        return ToolCallContext.executionBackend();
    }
}
