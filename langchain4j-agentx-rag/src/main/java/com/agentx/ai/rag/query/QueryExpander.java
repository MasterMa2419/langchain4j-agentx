package com.agentx.ai.rag.query;

import java.util.List;

/**
 * 查询扩展 SPI：1 输入 → N 输出。
 *
 * <p>对应 1.x 的 Spring AI {@code QueryExpander}。典型实现：多查询扩展。
 *
 * @author bigchui
 */
@FunctionalInterface
public interface QueryExpander {

    List<Query> expand(Query query);
}
