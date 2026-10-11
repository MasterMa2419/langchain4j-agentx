package com.agentx.ai.rag.retrieve.reranker;

import com.agentx.ai.rag.common.RagDocument;

import java.util.List;

/**
 * RAG 重排 SPI。
 *
 * 对检索召回的候选集做 query-document 相关度二次打分，截断数量由实现自己决定。
 * 不传实现则跳过重排，不影响主流程。
 *
 * @author bigchui
 */
public interface Reranker {

    List<RagDocument> rerank(String query, List<RagDocument> candidates);
}
