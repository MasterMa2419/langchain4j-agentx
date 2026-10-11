package com.agentx.ai.rag.parser.image;

import com.agentx.ai.rag.exception.RagErrorCode;
import com.agentx.ai.rag.exception.RagException;
import dev.langchain4j.data.image.Image;
import dev.langchain4j.data.message.ImageContent;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import lombok.extern.slf4j.Slf4j;

import java.util.List;
import java.util.Objects;

/**
 * 基于 LangChain4j ChatModel 的图片语义理解实现。
 *
 * <p>调用方传入的 ChatModel 必须支持图片输入。图片以 base64 形式传给模型，
 * 不要求图片先拥有公网 URL。
 *
 * @author bigchui
 */
@Slf4j
public final class ChatModelImageDescriber implements ImageDescriber {

    private final ChatModel chatModel;
    private final String modelName;

    public ChatModelImageDescriber(ChatModel chatModel) {
        this.chatModel = Objects.requireNonNull(chatModel, "chatModel");
        this.modelName = defaultModelName(chatModel);
    }

    @Override
    public ImageDescription describe(ImageDescriptionRequest request) {
        Objects.requireNonNull(request, "request");
        log.debug("图片语义理解开始: image={}, model={}, bytes={}",
                request.imageFileName(), modelName, request.image() == null ? 0 : request.image().length);
        try {
            String base64 = java.util.Base64.getEncoder().encodeToString(request.image());
            Image image = Image.builder()
                    .base64Data(base64)
                    .mimeType(request.mediaType())
                    .build();
            UserMessage userMessage = UserMessage.builder()
                    .addContent(new TextContent(prompt(request)))
                    .addContent(new ImageContent(image))
                    .build();

            Object response = chatModel.chat(ChatRequest.builder()
                    .messages(List.of(userMessage))
                    .build());
            String text = extractText(response);
            if (text == null || text.isBlank()) {
                throw new RagException(RagErrorCode.IMAGE_UNDERSTANDING_FAILED,
                        "图片语义描述为空: " + request.imageFileName());
            }
            log.debug("图片语义理解完成: image={}, descLen={}", request.imageFileName(), text.length());
            return new ImageDescription(normalize(text), modelName);
        } catch (RagException e) {
            log.warn("图片语义理解失败: image={}, err={}", request.imageFileName(), e.getMessage());
            throw e;
        } catch (Exception e) {
            log.warn("图片语义理解异常: image={}, err={}", request.imageFileName(), e.getMessage());
            throw new RagException(RagErrorCode.IMAGE_UNDERSTANDING_FAILED,
                    "图片语义理解失败: " + request.imageFileName(), e);
        }
    }

    private String prompt(ImageDescriptionRequest request) {
        return "你是文档图片理解引擎。请根据图片和所属文档信息生成用于知识库检索的图片描述。"
                + "要求说明图片类型、场景、对象、布局、流程、数据和图中重要文字；"
                + "结合上下文说明图片的作用；不要编造不存在的信息。"
                + "直接输出纯文本，不要换行，不要 Markdown，不要额外解释。"
                + "所属文档：" + request.documentFileName()
                + "；图片标题：" + request.caption()
                + "；图片文件：" + request.imageFileName() + "。";
    }

    private static String extractText(Object response) {
        if (response instanceof dev.langchain4j.model.chat.response.ChatResponse chatResponse
                && chatResponse.aiMessage() != null) {
            return chatResponse.aiMessage().text();
        }
        if (response instanceof String text) {
            return text;
        }
        return null;
    }

    private static String defaultModelName(ChatModel chatModel) {
        Objects.requireNonNull(chatModel, "chatModel");
        try {
            var parameters = chatModel.defaultRequestParameters();
            if (parameters != null && parameters.modelName() != null && !parameters.modelName().isBlank()) {
                return parameters.modelName();
            }
        } catch (RuntimeException ignored) {
            // 个别实现可能不暴露默认配置，此时退回到实现类名作为描述来源。
        }
        return chatModel.getClass().getName();
    }

    private String normalize(String text) {
        return text.replaceAll("[\\r\\n]+", " ").trim();
    }
}
