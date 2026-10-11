package com.agentx.ai.core.context.compress.strategy;

import com.agentx.ai.core.context.ContextPolicy;
import com.agentx.ai.core.context.TokenEstimator;
import com.agentx.ai.core.context.compress.CompressionContext;
import com.agentx.ai.core.context.compress.CompressionStrategy;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 大消息 offload 策略基类（L2/L3 共享逻辑）。
 * 子类只需实现 {@link #scanEnd(CompressionContext)} 决定扫描上界。
 *
 * @author bigchui
 */
public abstract class AbstractLargeMsgOffloadStrategy implements CompressionStrategy {

    private static final Logger log = LoggerFactory.getLogger(AbstractLargeMsgOffloadStrategy.class);

    /**
     * 历史轮次扫描下界（inclusive），子类可覆盖。
     */
    protected int scanStart(CompressionContext ctx) {
        return 0;
    }

    /**
     * 历史轮次扫描上界（exclusive），子类决定是否受 lastKeep 保护。
     */
    protected abstract int scanEnd(CompressionContext ctx);

    @Override
    public boolean tryCompress(CompressionContext ctx) {
        ContextPolicy policy = ctx.policy();
        int from = scanStart(ctx);
        int to = scanEnd(ctx);
        if (to <= from) {
            return false;
        }

        var messages = ctx.messages();
        int compressedCount = 0;

        for (int i = from; i < to; i++) {
            ChatMessage original = messages.get(i);
            if (!shouldOffload(original, policy.largePayloadTokens())) {
                continue;
            }

            String uuid = ctx.hasOffloadStore()
                    ? ctx.offloadStore().offload(ctx.conversationId(), ctx.sessionId(), original)
                    : null;

            ChatMessage replaced = replaceWithPreview(original, uuid, policy);
            messages.set(i, replaced);
            compressedCount++;

            log.info("[{}] Offloaded large message: index={}, originalTokens={}, uuid={}",
                    name(), i, TokenEstimator.estimateTokens(original), uuid);
        }
        return compressedCount > 0;
    }

    private boolean shouldOffload(ChatMessage msg, int thresholdTokens) {
        if (msg instanceof AiMessage am
                && am.toolExecutionRequests() != null && !am.toolExecutionRequests().isEmpty()) {
            return false;
        }
        return TokenEstimator.estimateTokens(msg) > thresholdTokens;
    }

    /**
     * 用预览 + offload 引用替换原消息，保留原消息的 role 与结构。
     * - ToolExecutionResultMessage：保留 id/toolName，重建消息（仅替换文本）
     * - AiMessage 纯文本：重建 AiMessage
     * - UserMessage / SystemMessage：重建对应消息
     */
    private ChatMessage replaceWithPreview(ChatMessage original, String uuid, ContextPolicy policy) {
        String fullText = extractText(original);
        int keep = policy.offloadPreviewChars();
        String preview;
        if (fullText.length() <= keep) {
            preview = fullText;
        } else {
            preview = fullText.substring(0, keep)
                    + "...[内容已 offload"
                    + (uuid != null ? ", uuid=" + uuid : "")
                    + "，可调用 context_reload 工具取回]";
        }

        if (original instanceof ToolExecutionResultMessage trm) {
            return ToolExecutionResultMessage.from(trm.id(), trm.toolName(), preview);
        }
        if (original instanceof AiMessage) {
            return AiMessage.from(preview);
        }
        if (original instanceof SystemMessage) {
            return SystemMessage.from(preview);
        }
        return UserMessage.from(preview);
    }

    private String extractText(ChatMessage msg) {
        if (msg instanceof AiMessage am) {
            return am.text() != null ? am.text() : "";
        }
        if (msg instanceof ToolExecutionResultMessage trm) {
            return trm.text() != null ? trm.text() : "";
        }
        if (msg instanceof UserMessage um) {
            return um.singleText();
        }
        if (msg instanceof SystemMessage sm) {
            return sm.text();
        }
        return "";
    }
}
