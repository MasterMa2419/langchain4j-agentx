package com.agentx.ai.rag.retrieve;

import com.agentx.ai.rag.common.RagDocument;
import com.agentx.ai.rag.query.Query;

import java.util.List;

/**
 * 文档检索 SPI。
 *
 * <p>对应 1.x 的 Spring AI {@code DocumentRetriever}。实现负责把 {@link Query}
 * 变成一组相关文档，典型实现：向量检索、父子检索、全文检索。
 *
 * @author bigchui
 */
@FunctionalInterface
public interface DocumentRetriever {

    List<RagDocument> retrieve(Query query);
}
