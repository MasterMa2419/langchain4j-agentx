package com.agentx.ai.core.tools;

import com.agentx.ai.core.model.AgentTool;
import com.agentx.ai.core.sandbox.ExecutionBackend;
import com.agentx.ai.core.sandbox.ToolCallContext;
import com.agentx.ai.core.sandbox.UnavailableExecutionBackend;
import com.fasterxml.jackson.databind.ObjectMapper;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.service.tool.ToolExecutor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Bash Tool - Shell 命令执行工具
 *
 * <p>沙箱模式：执行后端从 {@link ToolCallContext} 读取（由引擎在工具执行前绑定），
 * 直接委托 {@link ExecutionBackend#executeCommand}；
 * 非沙箱模式：走 {@link ShellSessionManager} 持久化会话。
 *
 * @author bigchui
 */
public class BashTool {

    private static final Logger log = LoggerFactory.getLogger(BashTool.class);
    private static final ObjectMapper objectMapper = new ObjectMapper();
    private static final String DEFAULT_SESSION_ID = "default";

    private final ShellSessionManager sessionManager;
    private final long defaultTimeoutMs;
    private final int maxLines;
    private final int maxBytes;

    private BashTool(Builder builder) {
        this.sessionManager = builder.sessionManager != null
                ? builder.sessionManager
                : ShellSessionManager.builder()
                  .maxLines(builder.maxLines)
                  .maxBytes(builder.maxBytes)
                  .timeoutMs(builder.timeoutMs)
                  .build();
        this.defaultTimeoutMs = builder.timeoutMs;
        this.maxLines = builder.maxLines;
        this.maxBytes = builder.maxBytes;
    }

    /**
     * 创建 Bash 工具（默认配置）。
     * <p>
     * 工具描述会根据当前操作系统动态生成，确保 LLM 生成正确的命令。
     *
     * @return AgentTool 列表，包含 bash 工具
     */
    public static List<AgentTool> create() {
        BashTool tool = builder().build();
        return List.of(createAgentTool(tool, buildDynamicDescription()));
    }

    /**
     * 构建 AgentTool：动态描述写入 ToolSpecification，执行器带沙箱分支。
     */
    private static AgentTool createAgentTool(BashTool tool, String dynamicDescription) {
        ToolSpecification spec = ToolSpecification.builder()
                .name("bash")
                .description(dynamicDescription)
                .build();
        ToolExecutor executor = (ToolExecutionRequest request, Object memoryId) ->
                tool.call(request.arguments());
        return new AgentTool(spec, executor);
    }

    /**
     * 构建动态工具描述（根据当前操作系统）
     */
    private static String buildDynamicDescription() {
        boolean isWindows = System.getProperty("os.name").toLowerCase().contains("win");
        String osType = isWindows ? "Windows (cmd.exe)" : "Unix/Linux/Mac (bash)";

        String osNotes;
        if (isWindows) {
            osNotes = """
                    **Windows 命令注意事项:**
                    - 用 'type' 代替 'cat'
                    - 用 'dir' 代替 'ls'
                    - 用 'copy' 或 'type nul > file' 代替 'touch'
                    - 用 'mkdir dirname'（不要加 -p！Windows 的 mkdir 自动创建父目录）
                    - 绝对不要用 'mkdir -p'，这会创建一个名为 '-p' 的文件夹！
                    - 用 'del' 代替 'rm'
                    - 用 'move' 代替 'mv'
                    - 用 'findstr' 代替 'grep'
                    - 路径使用反斜杠: C:\\\\Users\\\\...
                    - 用 '&&' 或 '&' 连接多条命令

                    **Python 注意事项（重要）:**
                    - 复杂代码或第三方库：先写入 .py 文件，再执行 python script.py
                    - 正确写法: python -c "open('s.py','w',encoding='utf-8').write('''代码''')" && python s.py
                    - 避免: echo '中文' > file.py（编码错误！）""";
        } else {
            osNotes = """
                    **Unix/Linux/Mac 命令注意事项:**
                    - 使用标准命令: ls, cat, grep, find, touch, mkdir, rm 等
                    - 'mkdir -p' 可自动创建父目录
                    - 路径使用正斜杠: /home/user/...
                    - 用 '&&' 或 ';' 连接多条命令""";
        }

        return """
                在持久化 Shell 会话中执行命令。

                **当前操作系统: %s**

                %s

                适用场景:
                - 执行 Shell 命令、系统命令、git 操作
                - 运行 Python 脚本: python script.py
                - 安装包、构建工具等

                Shell 会话特性:
                - 工作目录持久化（用 'cd' 切换）
                - 环境变量持久化
                - 设置 restart=true 可清除会话状态

                ⚠️ 工具优先级（重要）:
                本工具是最后的手段，请优先使用专用工具:
                - 读取文件 → read_file（不要用 cat/head/tail）
                - 编辑文件 → edit_file（不要用 sed/awk）
                - 写入文件 → write_file（不要用 echo/cat 重定向）
                - 搜索文件 → glob_files（不要用 find/ls）
                - 搜索内容 → grep（不要用 grep/rg 命令）""".formatted(osType, osNotes);
    }

    /**
     * 工具入口：解析参数 JSON，按沙箱/本地分支执行。
     */
    String call(String toolInput) {
        ExecutionBackend eb = ToolCallContext.extractExecutionBackend();
        if (eb != null && eb.isSandboxed()) {
            return executeInSandbox(eb, toolInput);
        }
        Map<String, Object> input = parseInput(toolInput);
        String command = (String) input.get("command");
        Boolean restart = input.get("restart") instanceof Boolean b ? b : null;
        Long timeoutMs = input.get("timeoutMs") instanceof Number n ? n.longValue() : null;
        return executeShellCommand(command, restart, timeoutMs);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> parseInput(String toolInput) {
        try {
            if (toolInput == null || toolInput.isBlank()) {
                return Map.of();
            }
            return objectMapper.readValue(toolInput, Map.class);
        } catch (Exception e) {
            return Map.of();
        }
    }

    /**
     * 创建 Builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * 执行 Shell 命令（持久化会话）。
     */
    public String executeShellCommand(
            String command,
            Boolean restart,
            Long timeoutMs) {

        log.debug("BashTool called with command: {}, restart: {}", command, restart);

        // 使用默认会话 ID
        String sessionId = DEFAULT_SESSION_ID;

        // 处理会话重启
        if (restart != null && restart) {
            log.debug("Restarting shell session: {}", sessionId);
            sessionManager.removeSession(sessionId);
        }

        // 执行命令
        try {
            ShellSessionManager.CommandResult result = sessionManager.executeCommand(
                    sessionId,
                    command,
                    null  // 工作目录由会话管理器自动处理
            );

            // 格式化输出
            return formatResult(result);
        } catch (Exception e) {
            log.error("Error executing command: {}", e.getMessage(), e);
            return "Error executing command: " + e.getMessage();
        }
    }

    /**
     * 简化版执行方法（直接调用）
     */
    public String executeShellCommand(String command) {
        return executeShellCommand(command, null, null);
    }

    /**
     * 格式化命令执行结果
     */
    private String formatResult(ShellSessionManager.CommandResult result) {
        List<String> parts = new ArrayList<>();

        if (result.output() != null && !result.output().isEmpty()) {
            parts.add(result.output());
        }
        if (result.error() != null && !result.error().isEmpty()) {
            parts.add("STDERR:\n" + result.error());
        }
        if (result.exitCode() != 0) {
            parts.add("[Exit code: " + result.exitCode() + "]");
        }

        return String.join("\n", parts);
    }

    /**
     * 获取会话管理器
     */
    public ShellSessionManager getSessionManager() {
        return sessionManager;
    }

    @SuppressWarnings("unchecked")
    private String executeInSandbox(ExecutionBackend eb, String toolInput) {
        try {
            Map<String, Object> input = objectMapper.readValue(toolInput == null || toolInput.isBlank() ? "{}" : toolInput, Map.class);
            String command = (String) input.get("command");
            if (command == null || command.isBlank()) {
                return "Error: command is required";
            }
            long timeout = defaultTimeoutMs;
            Object timeoutObj = input.get("timeoutMs");
            if (timeoutObj instanceof Number n) {
                timeout = n.longValue();
            }
            ShellSessionManager.CommandResult result = eb.executeCommand(command, timeout);
            return formatResult(result);
        } catch (Exception e) {
            log.error("Sandbox bash execution error: {}", e.getMessage(), e);
            return "Error executing command: " + e.getMessage();
        }
    }

    /**
     * Builder 模式
     */
    public static class Builder {
        private ShellSessionManager sessionManager;
        private long timeoutMs = 120000;  // 默认 2 分钟
        private int maxLines = 10000;     // 默认最大 10000 行
        private int maxBytes = 100000;    // 默认最大 100KB

        public Builder sessionManager(ShellSessionManager sessionManager) {
            this.sessionManager = sessionManager;
            return this;
        }

        public Builder timeoutMs(long timeoutMs) {
            this.timeoutMs = timeoutMs;
            return this;
        }

        public Builder maxLines(int maxLines) {
            this.maxLines = maxLines;
            return this;
        }

        public Builder maxBytes(int maxBytes) {
            this.maxBytes = maxBytes;
            return this;
        }

        public BashTool build() {
            return new BashTool(this);
        }
    }
}
