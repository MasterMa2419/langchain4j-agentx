package com.agentx.ai.rag.query;

/**
 * 查询转换 SPI：1 输入 → 1 输出。
 *
 * <p>对应 1.x 的 Spring AI {@code QueryTransformer}。典型实现：压缩、改写、HyDE。
 *
 * @author bigchui
 */
@FunctionalInterface
public interface QueryTransformer {

    Query transform(Query query);
}
