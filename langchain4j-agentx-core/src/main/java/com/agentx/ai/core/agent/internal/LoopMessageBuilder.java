package com.agentx.ai.core.agent.internal;

import com.agentx.ai.core.memory.store.SessionMessageStore;
import com.agentx.ai.core.memory.util.MemoryInjector;
import com.agentx.ai.core.model.OutputType;
import com.agentx.ai.core.model.RunnableParams;
import com.agentx.ai.core.model.ThinkingMode;
import com.agentx.ai.core.prompt.PromptConstants;
import com.agentx.ai.core.tools.toolsearch.DeferredToolRegistry;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * 消息构建器 — 构建初始消息列表并记录新消息边界。
 * 历史从 SessionMessageStore 加载完整消息链（含工具调用与工具结果）。
 *
 * <p>2.0 变更：结构化输出改为直接在 user 消息中注入 JSON 格式指令
 * （1.x 用 Spring AI BeanOutputConverter 生成 JSON schema 指令），
 * 输出仍由 JsonRepairUtil 修复后由调用方反序列化。
 *
 * @author bigchui
 */
public class LoopMessageBuilder {

    private static final Logger log = LoggerFactory.getLogger(LoopMessageBuilder.class);

    private final String instructions;
    private final SessionMessageStore sessionMessageStore;
    private final MemoryInjector memoryInjector;
    private final ThinkingMode thinkingMode;
    private final DeferredToolRegistry deferredToolRegistry;
    private final boolean todoWriteEnabled;
    private final boolean enableSession;

    public LoopMessageBuilder(String instructions, SessionMessageStore sessionMessageStore,
                              MemoryInjector memoryInjector, ThinkingMode thinkingMode,
                              DeferredToolRegistry deferredToolRegistry,
                              boolean todoWriteEnabled, boolean enableSession) {
        this.instructions = instructions;
        this.sessionMessageStore = sessionMessageStore;
        this.memoryInjector = memoryInjector;
        this.thinkingMode = thinkingMode;
        this.deferredToolRegistry = deferredToolRegistry;
        this.todoWriteEnabled = todoWriteEnabled;
        this.enableSession = enableSession;
    }

    /**
     * 构建初始消息列表，返回消息列表与新消息起点索引。
     * newMsgStartIndex 标记本次调用新增消息的边界，终态批量落库时据此切片。
     */
    public BuiltMessages buildInitialMessages(String query, RunnableParams params) {
        List<ChatMessage> messages = new ArrayList<>();

        // 1. 系统提示词
        String systemPrompt = "";
        if (instructions != null && !instructions.isBlank()) {
            systemPrompt = instructions;
        }
        systemPrompt = appendSection(systemPrompt, memoryInjector.buildMemorySection(params, query));

        String customParamSection = buildCustomParamSection(params);
        if (!customParamSection.isEmpty()) {
            systemPrompt = systemPrompt + customParamSection;
        }
        if (deferredToolRegistry != null) {
            systemPrompt = appendSection(systemPrompt, PromptConstants.TOOL_SEARCH_GUIDANCE);
        }
        if (todoWriteEnabled) {
            systemPrompt = appendSection(systemPrompt, PromptConstants.TODO_WRITE_GUIDANCE);
        }
        if (!systemPrompt.isEmpty()) {
            messages.add(SystemMessage.from(systemPrompt));
        }

        // 2. 历史消息链（优先 working_messages 压缩视图，回退 original_messages）
        String conversationId = params != null ? params.getConversationId() : null;
        if (enableSession && sessionMessageStore != null && conversationId != null) {
            List<ChatMessage> history = sessionMessageStore.getMessages(conversationId, "working_messages");
            if (history.isEmpty()) {
                history = sessionMessageStore.getMessages(conversationId, "original_messages");
            }
            for (ChatMessage msg : history) {
                if (!(msg instanceof SystemMessage)) {
                    messages.add(msg);
                }
            }
        }

        // 4. 边界标记：此前的消息为历史/背景，此后为本轮新增
        int newMsgStartIndex = messages.size();

        // 5. 当前用户问题
        String userContent = query;
        if (params != null && params.getOutputType() != null) {
            userContent = userContent + "\n" + buildJsonFormatInstruction(params.getOutputType());
        }
        if (thinkingMode == ThinkingMode.DISABLED) {
            userContent = userContent + "\n<no_think>";
        }
        messages.add(UserMessage.from(userContent));

        return new BuiltMessages(messages, newMsgStartIndex);
    }

    /**
     * 构建 JSON 输出格式指令。
     * 形如：
     * <pre>
     * 你的回答必须只包含一个合法 JSON 对象，不要包含任何其他文本或 markdown 代码块标记。
     * JSON 结构如下（字段名必须一致）：
     * {
     *   "plan" : "",
     *   "steps" : []
     * }
     * </pre>
     */
    static String buildJsonFormatInstruction(OutputType outputType) {
        Type type = outputType.getType();
        String shape = describeTypeShape(type);
        return "你的回答必须只包含一个合法 JSON，不要包含任何其他文本或 markdown 代码块标记。\n"
                + "JSON 结构如下（字段名必须一致，值为占位说明，请填入真实内容）：\n" + shape;
    }

    /**
     * 反射描述类型形状：顶层对象给字段清单，List 给元素对象字段清单。
     * 仅支持简单 POJO（getter/isX 或 public 字段）与 List<T>。
     */
    private static String describeTypeShape(Type type) {
        Class<?> raw;
        boolean isList = false;
        if (type instanceof ParameterizedType pt && pt.getRawType() == List.class) {
            isList = true;
            Type arg = pt.getActualTypeArguments()[0];
            raw = arg instanceof Class<?> c ? c : Object.class;
        } else if (type instanceof Class<?> c) {
            raw = c;
        } else {
            return "\"<json>\"";
        }

        StringBuilder sb = new StringBuilder();
        if (isList) {
            sb.append("[\n  {\n");
        } else {
            sb.append("{\n");
        }
        List<String> lines = new ArrayList<>();
        for (var field : raw.getDeclaredFields()) {
            if (java.lang.reflect.Modifier.isStatic(field.getModifiers())) {
                continue;
            }
            lines.add("    \"" + field.getName() + "\": \"\"");
        }
        if (lines.isEmpty()) {
            sb.append("    \"...\": \"\"");
        } else {
            sb.append(String.join(",\n", lines));
        }
        if (isList) {
            sb.append("\n  }\n]");
        } else {
            sb.append("\n}");
        }
        return sb.toString();
    }

    private String buildCustomParamSection(RunnableParams params) {
        if (params == null || params.getCustomParams() == null || params.getCustomParams().isEmpty()) {
            return "";
        }
        StringBuilder sb = new StringBuilder();
        sb.append("\n\n## 系统参数（LLM 可见）\n");
        for (Map.Entry<String, Object> entry : params.getCustomParams().entrySet()) {
            sb.append(entry.getKey()).append(": ").append(entry.getValue()).append("\n");
        }
        return sb.toString();
    }

    private static String appendSection(String base, String section) {
        if (section == null || section.isEmpty()) {
            return base;
        }
        return base.isEmpty() ? section : base + "\n\n" + section;
    }
}
