package com.agentx.ai.core.agent.internal;

import com.agentx.ai.core.model.AgentStreamEvent;
import com.agentx.ai.core.model.ThinkingMode;
import com.agentx.ai.core.stage.ThinkTagParser;
import reactor.core.publisher.Sinks;

/**
 * ThinkingMode 处理器 — 封装思考模式的分支逻辑。
 *
 * <p>2.0 双通道设计：
 * <ul>
 *   <li><b>独立通道</b>（{@code reasoning_content} / thinking 字段）：
 *       LC4j 通过 {@code onPartialThinking} 回调传递，引擎把 {@code LlmChunk.Thinking}
 *       转发给 {@link #processReasoningChunk}。对应 {@link ThinkingMode#REASONING_CONTENT}</li>
 *   <li><b>内嵌通道</b>（{@code <think/>} 标签混在 text 里）：
 *       走 {@link ThinkTagParser} 流式解析。对应 {@link ThinkingMode#THINK_TAG}</li>
 * </ul>
 *
 * <p>1.x 的「从 AssistantMessage metadata / 反射 getReasoningContent 提取」逻辑随
 * Spring AI 类型移除而删除——LC4j 的思考内容在流式期就是独立回调，无需事后提取。
 *
 * @author bigchui
 */
public final class ThinkingModeProcessor {

    private final ThinkingMode thinkingMode;

    public ThinkingModeProcessor(ThinkingMode thinkingMode) {
        this.thinkingMode = thinkingMode;
    }

    public void processStreamChunk(String text, RoundState state,
                                   Sinks.Many<AgentStreamEvent> sink) {
        if (thinkingMode == ThinkingMode.THINK_TAG) {
            processThinkTagChunk(text, state, sink, true);
        } else {
            processPlainChunk(text, state, sink);
        }
    }

    public void processReasoningChunk(String reasoning, RoundState state,
                                      Sinks.Many<AgentStreamEvent> sink) {
        if (reasoning != null && !reasoning.isEmpty()) {
            state.reasoningBuffer.append(reasoning);
            sink.tryEmitNext(new AgentStreamEvent.Thinking(reasoning));
        }
    }

    public void processForceFinalChunk(String text, boolean[] inThinkHolder,
                                       StringBuilder answerBuffer,
                                       StringBuilder reasoningBuffer,
                                       Sinks.Many<AgentStreamEvent> sink) {
        if (thinkingMode == ThinkingMode.THINK_TAG) {
            processForceFinalThinkTag(text, inThinkHolder, answerBuffer, reasoningBuffer, sink, true);
        } else {
            if (text != null && !text.isEmpty()) {
                answerBuffer.append(text);
                sink.tryEmitNext(new AgentStreamEvent.Text(text));
            }
        }
    }

    public String extractThinkContent(String input) {
        if (input == null || input.isEmpty()) {
            return null;
        }
        StringBuilder think = new StringBuilder();
        ThinkTagParser.ParseResult result = ThinkTagParser.parse(input, false);
        for (ThinkTagParser.Segment seg : result.segments()) {
            if (seg.thinking()) {
                think.append(seg.content());
            }
        }
        return think.isEmpty() ? null : think.toString().trim();
    }

    /** 去除 think 标签，返回纯正文。 */
    public String stripThinkTagsIfNeeded(String answer) {
        if (thinkingMode == ThinkingMode.DISABLED && answer != null) {
            return ThinkTagParser.stripThinkTags(answer);
        }
        return answer;
    }

    private void processPlainChunk(String text, RoundState state,
                                   Sinks.Many<AgentStreamEvent> sink) {
        if (text != null && !text.isEmpty()) {
            state.textBuffer.append(text);
            sink.tryEmitNext(new AgentStreamEvent.Text(text));
        }
    }

    private void processThinkTagChunk(String text, RoundState state,
                                      Sinks.Many<AgentStreamEvent> sink,
                                      boolean emitThinkingEvents) {
        if (text == null || text.isEmpty()) {
            return;
        }
        ThinkTagParser.ParseResult result = ThinkTagParser.parse(text, state.inThink);
        state.inThink = result.inThink();
        for (ThinkTagParser.Segment seg : result.segments()) {
            if (seg.thinking()) {
                if (emitThinkingEvents) {
                    sink.tryEmitNext(new AgentStreamEvent.Thinking(seg.content()));
                }
                state.reasoningBuffer.append(seg.content());
            } else {
                state.textBuffer.append(seg.content());
                sink.tryEmitNext(new AgentStreamEvent.Text(seg.content()));
            }
        }
    }

    private void processForceFinalThinkTag(String text, boolean[] inThinkHolder,
                                           StringBuilder answerBuffer,
                                           StringBuilder reasoningBuffer,
                                           Sinks.Many<AgentStreamEvent> sink,
                                           boolean emitThinkingEvents) {
        if (text == null || text.isEmpty()) {
            return;
        }
        ThinkTagParser.ParseResult result = ThinkTagParser.parse(text, inThinkHolder[0]);
        inThinkHolder[0] = result.inThink();
        for (ThinkTagParser.Segment seg : result.segments()) {
            if (seg.thinking()) {
                if (emitThinkingEvents) {
                    sink.tryEmitNext(new AgentStreamEvent.Thinking(seg.content()));
                }
                reasoningBuffer.append(seg.content());
            } else {
                answerBuffer.append(seg.content());
                sink.tryEmitNext(new AgentStreamEvent.Text(seg.content()));
            }
        }
    }
}
