package com.agentx.ai.core.utils;

import com.agentx.ai.core.model.AgentTool;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * 工具合并工具类，支持按名称去重。
 *
 * @author bigchui
 */
public final class ToolMergeUtil {

    private ToolMergeUtil() {
    }

    /**
     * 合并多个工具数组，按工具名称去重（相同名称只保留第一个）。
     *
     * @param toolArrays 工具数组
     * @return 合并并去重后的工具数组
     */
    @SafeVarargs
    public static AgentTool[] mergeTools(AgentTool[]... toolArrays) {
        List<AgentTool> result = new ArrayList<>();
        Set<String> seen = new LinkedHashSet<>();
        if (toolArrays != null) {
            for (AgentTool[] array : toolArrays) {
                if (array != null) {
                    for (AgentTool tool : array) {
                        String name = tool.getToolDefinition().name();
                        if (seen.add(name)) {
                            result.add(tool);
                        }
                    }
                }
            }
        }
        return result.toArray(new AgentTool[0]);
    }

    /**
     * 合并多个工具集合，按工具名称去重（相同名称只保留第一个）。
     *
     * @param toolLists 工具集合
     * @return 合并并去重后的工具数组
     */
    @SafeVarargs
    public static AgentTool[] mergeTools(List<AgentTool>... toolLists) {
        List<AgentTool> result = new ArrayList<>();
        Set<String> seen = new LinkedHashSet<>();
        if (toolLists != null) {
            for (List<AgentTool> list : toolLists) {
                if (list != null) {
                    for (AgentTool tool : list) {
                        String name = tool.getToolDefinition().name();
                        if (seen.add(name)) {
                            result.add(tool);
                        }
                    }
                }
            }
        }
        return result.toArray(new AgentTool[0]);
    }
}
