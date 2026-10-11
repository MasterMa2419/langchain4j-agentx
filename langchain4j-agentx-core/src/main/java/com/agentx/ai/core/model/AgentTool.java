package com.agentx.ai.core.model;

import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.agent.tool.ToolSpecifications;
import dev.langchain4j.service.tool.DefaultToolExecutor;
import dev.langchain4j.service.tool.ToolExecutor;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Agent 工具统一抽象：工具定义（{@link ToolSpecification}）与工具执行器（{@link ToolExecutor}）的组合。
 *
 * <p>对应 1.x 的 Spring AI {@code ToolCallback}。框架要求工具执行权归引擎所有，
 * 因此以「定义 + 执行器」的显式组合替代模型侧自动执行。
 *
 * <p>典型构造方式：
 * <pre>{@code
 * // 1. 注解扫描（@Tool 方法 + 反射执行器）
 * AgentTool.fromAnnotated(new MyTools());
 *
 * // 2. 编程式
 * new AgentTool(
 *     ToolSpecification.builder().name("t").description("d").build(),
 *     (request, memoryId) -> "result");
 * }</pre>
 *
 * @author bigchui
 */
public class AgentTool {

    private final ToolSpecification toolDefinition;
    private final ToolExecutor toolExecutor;

    public AgentTool(ToolSpecification toolDefinition, ToolExecutor toolExecutor) {
        this.toolDefinition = Objects.requireNonNull(toolDefinition, "toolDefinition must not be null");
        this.toolExecutor = Objects.requireNonNull(toolExecutor, "toolExecutor must not be null");
    }

    public ToolSpecification getToolDefinition() {
        return toolDefinition;
    }

    public ToolExecutor getToolExecutor() {
        return toolExecutor;
    }

    public String getName() {
        return toolDefinition.name();
    }

    /**
     * 扫描对象上所有 {@code @Tool} 注解方法，生成对应的 AgentTool 列表。
     * 每个方法生成一个 {@link ToolSpecification}（由 {@link ToolSpecifications} 反射构建）
     * 和一个 {@link DefaultToolExecutor}（反射调用）。
     */
    public static List<AgentTool> fromAnnotated(Object toolObject) {
        Objects.requireNonNull(toolObject, "toolObject must not be null");
        List<ToolSpecification> specifications = ToolSpecifications.toolSpecificationsFrom(toolObject);
        List<AgentTool> result = new ArrayList<>(specifications.size());
        for (Method method : findAnnotatedMethods(toolObject)) {
            ToolSpecification spec = ToolSpecifications.toolSpecificationFrom(method);
            ToolExecutor executor = new DefaultToolExecutor(toolObject, method);
            result.add(new AgentTool(spec, executor));
        }
        return result;
    }

    private static List<Method> findAnnotatedMethods(Object toolObject) {
        List<Method> methods = new ArrayList<>();
        for (Class<?> type = toolObject.getClass(); type != null && type != Object.class; type = type.getSuperclass()) {
            for (Method method : type.getDeclaredMethods()) {
                if (method.isAnnotationPresent(dev.langchain4j.agent.tool.Tool.class)) {
                    methods.add(method);
                }
            }
        }
        return methods;
    }

    @Override
    public String toString() {
        return "AgentTool{" + toolDefinition.name() + "}";
    }
}
