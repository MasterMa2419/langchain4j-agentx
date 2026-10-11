package com.agentx.ai.core.context.compress.strategy;

import com.agentx.ai.core.context.ContextPolicy;
import com.agentx.ai.core.context.TokenEstimator;
import com.agentx.ai.core.context.compress.CompressionContext;
import com.agentx.ai.core.context.compress.CompressionStrategy;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * L1 历史工具调用列表（不调 LLM）。
 * 在历史轮次区域内，若存在连续 N 条（AiMessage+tool_calls 或 ToolExecutionResultMessage）≥ minConsecutiveToolMessages
 * 且总 token ≥ minCompressionTokens 的工具段，将该段替换为单条字符串模板消息。
 * 原文按段整体 offload，供 context_reload 工具回溯。
 *
 * <p>2.0 语义说明：Spring AI 中一次工具交互是「assistant(toolCalls) + ToolResponseMessage(合并 N 个结果)」
 * 两条消息；LC4j 中是「AiMessage(toolExecutionRequests) + N 条 ToolExecutionResultMessage」。
 * 因此本策略的连续工具段判定按单条工具结果消息模型处理，L1 模板的输出语义与 1.x 保持一致。
 *
 * @author bigchui
 */
public class HistoricalToolListStrategy implements CompressionStrategy {

    private static final Logger log = LoggerFactory.getLogger(HistoricalToolListStrategy.class);

    @Override
    public boolean tryCompress(CompressionContext ctx) {
        ContextPolicy policy = ctx.policy();
        int scanEnd = ctx.historicalScanEnd(policy.lastKeep());
        if (scanEnd <= 0) {
            return false;
        }

        List<ChatMessage> messages = ctx.messages();
        List<int[]> ranges = findConsecutiveToolRuns(messages, 0, scanEnd, policy);
        if (ranges.isEmpty()) {
            return false;
        }

        int compressedSegments = 0;
        for (int i = ranges.size() - 1; i >= 0; i--) {
            int[] range = ranges.get(i);
            int start = range[0];
            int end = range[1];
            List<ChatMessage> segment = new ArrayList<>(messages.subList(start, end));

            String uuid = offloadSegment(ctx, segment);
            String summary = buildToolListSummary(segment, uuid, policy);

            for (int j = end - 1; j >= start; j--) {
                messages.remove(j);
            }
            messages.add(start, AiMessage.from(summary));
            compressedSegments++;
        }

        log.info("[L1] Compressed historical tool segments: segments={}, scanEnd={}",
                compressedSegments, scanEnd);
        return true;
    }

    private List<int[]> findConsecutiveToolRuns(List<ChatMessage> messages, int from, int to, ContextPolicy policy) {
        List<int[]> ranges = new ArrayList<>();
        int runStart = -1;
        int runLen = 0;
        int runTokens = 0;

        for (int i = from; i < to; i++) {
            ChatMessage msg = messages.get(i);
            if (isToolMessage(msg)) {
                if (runStart < 0) {
                    runStart = i;
                }
                runLen++;
                runTokens += TokenEstimator.estimateTokens(msg);
            } else {
                if (matchesThreshold(runLen, runTokens, policy)) {
                    ranges.add(new int[]{runStart, i});
                }
                runStart = -1;
                runLen = 0;
                runTokens = 0;
            }
        }
        if (matchesThreshold(runLen, runTokens, policy)) {
            ranges.add(new int[]{runStart, to});
        }
        return ranges;
    }

    private boolean matchesThreshold(int len, int tokens, ContextPolicy policy) {
        return len >= policy.minConsecutiveToolMessages()
                && tokens >= policy.minCompressionTokens();
    }

    private boolean isToolMessage(ChatMessage msg) {
        if (msg instanceof ToolExecutionResultMessage) {
            return true;
        }
        if (msg instanceof AiMessage am) {
            return am.toolExecutionRequests() != null && !am.toolExecutionRequests().isEmpty();
        }
        return false;
    }

    private String offloadSegment(CompressionContext ctx, List<ChatMessage> segment) {
        return ctx.hasOffloadStore()
                ? ctx.offloadStore().offload(ctx.conversationId(), ctx.sessionId(), segment)
                : null;
    }

    private String buildToolListSummary(List<ChatMessage> segment, String uuid, ContextPolicy policy) {
        List<String> entries = collectToolEntries(segment, policy);

        StringBuilder sb = new StringBuilder();
        sb.append("[历史工具调用记录]\n");
        sb.append("本轮对话之前，已执行过以下工具调用：\n");
        for (String entry : entries) {
            sb.append("- ").append(entry).append("\n");
        }
        if (uuid != null) {
            sb.append("（该段完整参数与结果已 offload，uuid=")
                    .append(uuid)
                    .append("；如需原文可调用 context_reload(uuid) 工具取回）");
        } else {
            sb.append("（当前未启用 offload 存储，仅保留工具调用摘要）");
        }
        return sb.toString();
    }

    private List<String> collectToolEntries(List<ChatMessage> segment, ContextPolicy policy) {
        List<String> entries = new ArrayList<>();
        for (ChatMessage msg : segment) {
            if (msg instanceof AiMessage am && am.toolExecutionRequests() != null) {
                for (ToolExecutionRequest tc : am.toolExecutionRequests()) {
                    entries.add(formatToolCall(tc, policy));
                }
            } else if (msg instanceof ToolExecutionResultMessage trm) {
                entries.add(formatResponseOnly(trm));
            }
        }
        return deduplicatePreservingOrder(entries);
    }

    private String formatToolCall(ToolExecutionRequest tc, ContextPolicy policy) {
        String name = tc.name() != null ? tc.name() : "unknown";
        String args = truncate(tc.arguments(), policy.toolArgPreviewChars());
        return name + "(" + args + ")";
    }

    private String formatResponseOnly(ToolExecutionResultMessage resp) {
        String name = resp.toolName() != null && !resp.toolName().isBlank() ? resp.toolName() : "unknown";
        return name + "(response only)";
    }

    private List<String> deduplicatePreservingOrder(List<String> entries) {
        List<String> deduped = new ArrayList<>();
        Set<String> seen = new HashSet<>();
        for (String entry : entries) {
            if (seen.add(entry)) {
                deduped.add(entry);
            }
        }
        return deduped;
    }

    private String truncate(String s, int max) {
        if (s == null) return "";
        if (s.length() <= max) return s;
        return s.substring(0, max) + "...";
    }
}
