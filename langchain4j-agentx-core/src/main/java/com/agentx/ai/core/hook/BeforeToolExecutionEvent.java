package com.agentx.ai.core.hook;

import com.agentx.ai.core.stage.AgentRuntimeContext;

/**
 * 单个工具执行前事件（工具执行器调用前）。
 *
 * <p>可修改：arguments 可替换（如沙箱路径重写）。
 * 2.0 起工具运行时上下文（沙箱后端等）通过 {@link com.agentx.ai.core.sandbox.ToolCallContext}
 * ThreadLocal 传递，不再经由本事件改写；Hook 若需干预沙箱后端，
 * 可通过 {@link AgentRuntimeContext} 操作。
 *
 * @author bigchui
 */
public final class BeforeToolExecutionEvent implements HookEvent {

    private final AgentRuntimeContext runtimeContext;
    private final String toolName;
    private final String toolCallId;
    private String arguments;

    public BeforeToolExecutionEvent(AgentRuntimeContext runtimeContext, String toolName, String toolCallId,
                                     String arguments) {
        this.runtimeContext = runtimeContext;
        this.toolName = toolName;
        this.toolCallId = toolCallId;
        this.arguments = arguments;
    }

    public AgentRuntimeContext getRuntimeContext() {
        return runtimeContext;
    }

    public String getToolName() {
        return toolName;
    }

    public String getToolCallId() {
        return toolCallId;
    }

    public String getArguments() {
        return arguments;
    }

    public void setArguments(String arguments) {
        this.arguments = arguments;
    }
}
