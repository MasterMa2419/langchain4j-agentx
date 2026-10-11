package com.agentx.ai.core.tools;

import com.agentx.ai.core.model.AgentTool;
import dev.langchain4j.agent.tool.P;
import dev.langchain4j.agent.tool.Tool;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.service.tool.ToolExecutor;
import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Engine;
import org.graalvm.polyglot.PolyglotException;
import org.graalvm.polyglot.Value;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

/**
 * Python 执行工具 - 使用 GraalVM polyglot 执行 Python 代码
 *
 * 该工具允许 Agent 执行 Python 代码片段并获取结果。
 * 使用 GraalVM 的 polyglot API 在沙箱环境中运行 Python 代码。
 *
 * 使用方式：
 * {@code
 * PythonTool pythonTool = new PythonTool();
 *
 * ReactAgent.builder()
 *     .tools(PythonTool.create())
 *     .build();
 * }
 *
 * @author bigchui
 * 
 */
public class PythonTool {

    private static final Logger log = LoggerFactory.getLogger(PythonTool.class);

    public static final String DESCRIPTION = """
            Executes Python code in a lightweight sandbox (GraalVM Python).

            **IMPORTANT - When to use THIS vs bash + python:**

            USE python TOOL for:
            - Simple calculations: 2 + 2, math.sqrt(16)
            - String manipulation: "hello".upper(), "a b c".split()
            - JSON processing: import json; json.loads(s)
            - Basic algorithms without external packages

            USE bash + python for:
            - Third-party packages: pptx, numpy, pandas, requests
            - File I/O operations
            - Complex data processing
            - Machine learning / data science

            **LIMITATIONS:**
            - Standard library only (no pip packages)
            - Sandbox environment (limited file access)

            Examples:
            - code = "2 + 2" returns "4"
            - code = "'hello'.upper()" returns "HELLO"
            - code = "import json; json.dumps({'x': 1})" returns '{"x": 1}'
            """;

    private Engine engine;

    /**
     * 构造函数 - 创建共享引擎以提升性能
     */
    public PythonTool() {
        try {
            this.engine = Engine.newBuilder()
                    .option("engine.WarnInterpreterOnly", "false")
                    .build();
        } catch (Exception e) {
            log.warn("Failed to initialize Python engine, Python tool will not be available: {}", e.getMessage());
            this.engine = null;
        }
    }

    /**
     * 创建 Python 工具的 AgentTool 列表（默认配置）
     *
     * 这是一个便捷方法，使用默认描述创建工具实例。
     *
     * @return AgentTool 列表，包含 python 工具
     */
    public static List<AgentTool> create() {
        return List.of(createAgentTool(DESCRIPTION));
    }

    /**
     * 创建 Python 工具的 AgentTool
     *
     * @param description 工具描述
     * @return AgentTool 实例
     */
    public static AgentTool createAgentTool(String description) {
        PythonTool tool = new PythonTool();
        ToolSpecification spec = ToolSpecification.builder()
                .name("python")
                .description(description)
                .build();
        ToolExecutor executor = (request, memoryId) -> tool.executePython(request.arguments());
        return new AgentTool(spec, executor);
    }

    /**
     * 执行 Python 代码（使用 @Tool 注解，可被 LLM 直接调用）
     *
     * @param code Python 代码
     * @return 执行结果
     */
    @Tool(name = "python", value =
            "Executes Python code and returns the result.\n\n" +
            "**LIMITATIONS:** Uses GraalVM Python (not CPython). Third-party packages " +
            "like numpy, pandas, pptx may not work. For complex tasks, use bash tool " +
            "to call system Python instead.\n\n" +
            "**Best for:** Simple calculations, string manipulation, JSON parsing.\n" +
            "**NOT for:** File operations, ML packages, packages with C extensions.\n\n" +
            "Examples:\n" +
            "- code = \"2 + 2\" returns \"4\"\n" +
            "- code = \"'Hello, ' + 'World'\" returns \"Hello, World\"")
    public String executePython(String toolInput) { // @formatter:on

        String code = extractCode(toolInput);
        return apply(code);
    }

    private static String extractCode(String toolInput) {
        // 参数为裸代码或 {"code": "..."} JSON 均兼容
        if (toolInput == null) {
            return null;
        }
        String trimmed = toolInput.trim();
        if (trimmed.startsWith("{") && trimmed.endsWith("}")) {
            try {
                com.fasterxml.jackson.databind.JsonNode node =
                        new com.fasterxml.jackson.databind.ObjectMapper().readTree(trimmed);
                if (node.has("code")) {
                    return node.get("code").asText();
                }
            } catch (Exception ignored) {
                // 非 JSON，按裸代码处理
            }
        }
        return toolInput;
    }

    public String apply(String code) {
        if (code == null || code.trim().isEmpty()) {
            return "Error: Python code cannot be empty";
        }

        // 清理代码：处理常见的 JSON 转义问题
        String cleanedCode = cleanCode(code);

        try (Context context = Context.newBuilder("python")
                .engine(engine)
                .allowAllAccess(false)        // 安全：默认限制访问权限
                .allowIO(true)                // 允许文件 I/O（pptx 等包需要）
                .allowNativeAccess(false)     // 安全：禁用本地访问
                .allowCreateProcess(false)    // 安全：禁用进程创建
                .allowHostAccess(true)        // 允许访问宿主对象
                .build()) {

            log.debug("Executing Python code: {}", cleanedCode);

            // 执行 Python 代码
            Value result = context.eval("python", cleanedCode);

            // 将结果转换为字符串
            if (result.isNull()) {
                return "Execution completed with no return value";
            }

            // 处理不同的结果类型
            if (result.isString()) {
                return result.asString();
            }
            else if (result.isNumber()) {
                return String.valueOf(result.as(Object.class));
            }
            else if (result.isBoolean()) {
                return String.valueOf(result.asBoolean());
            }
            else if (result.hasArrayElements()) {
                // 将数组/列表转换为字符串表示
                StringBuilder sb = new StringBuilder("[");
                long size = result.getArraySize();
                for (long i = 0; i < size; i++) {
                    if (i > 0) {
                        sb.append(", ");
                    }
                    Value element = result.getArrayElement(i);
                    sb.append(element.toString());
                }
                sb.append("]");
                return sb.toString();
            }
            else {
                // 其他类型使用 toString()
                return result.toString();
            }
        }
        catch (PolyglotException e) {
            log.error("Error executing Python code", e);
            return "Error executing Python code: " + e.getMessage();
        }
        catch (Exception e) {
            log.error("Unexpected error executing Python code", e);
            return "Unexpected error: " + e.getMessage();
        }
    }

    /**
     * 清理 Python 代码，处理常见的 JSON 转义问题。
     *
     * LLM 生成的代码中可能包含转义字符，例如：
     * - \\n 应该被转换为真正的换行符
     * - \\t 应该被转换为制表符
     * - \\r 应该被转换为回车符
     * - \\" 应该被转换为双引号
     * - \\\\ 应该被转换为单个反斜杠
     *
     * @param code 原始代码
     * @return 清理后的代码
     */
    private String cleanCode(String code) {
        if (code == null) {
            return null;
        }

        // 处理常见的转义序列
        String cleaned = code
                // 处理双反斜杠转义（JSON 解析后可能留下 \\n 这种模式）
                .replace("\\\\n", "\n")
                .replace("\\\\t", "\t")
                .replace("\\\\r", "\r")
                .replace("\\\\\"", "\"")
                .replace("\\\\'", "'")
                .replace("\\\\\\", "\\");

        return cleaned;
    }

    /**
     * Python 工具请求参数（保留以兼容旧调用方）
     */
    public static class PythonRequest {

        public String code;

        public PythonRequest() {
        }

        public PythonRequest(String code) {
            this.code = code;
        }
    }
}
