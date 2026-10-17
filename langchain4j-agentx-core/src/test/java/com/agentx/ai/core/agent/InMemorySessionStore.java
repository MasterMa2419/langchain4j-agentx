package com.agentx.ai.core.agent;

import com.agentx.ai.core.memory.store.SessionMessageStore;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import org.springframework.jdbc.core.JdbcTemplate;

import javax.sql.DataSource;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 测试用内存版会话存储：覆写 {@link SessionMessageStore} 的消息读写入口，
 * 避免测试依赖真实数据库（Data-only 侧不受影响）。
 *
 * @author bigchui
 */
class InMemorySessionStore extends SessionMessageStore {

    /** key = conversationId + "|" + stateKey */
    private final Map<String, List<ChatMessage>> data = new HashMap<>();

    InMemorySessionStore() {
        super(dummyDataSource());
    }

    /** 是否没有任何消息落库（SubAgent 模式下应为 true） */
    boolean isEmpty() {
        return data.values().stream().allMatch(List::isEmpty);
    }

    /** 已落库消息总数（跨 conversationId / stateKey 汇总） */
    int size() {
        return data.values().stream().mapToInt(List::size).sum();
    }

    /** 所有已落库消息（跨 conversationId / stateKey 汇总），用于断言消息类型 */
    List<ChatMessage> allMessages() {
        List<ChatMessage> all = new ArrayList<>();
        data.values().forEach(all::addAll);
        return all;
    }

    /**
     * JdbcTemplate 构造要求非空 DataSource，此处仅为占位；所有 JDBC 方法均不会被调用。
     */
    private static DataSource dummyDataSource() {
        return new DataSource() {
            @Override public java.sql.Connection getConnection() {
                throw new UnsupportedOperationException("in-memory test store");
            }

            @Override public java.sql.Connection getConnection(String username, String password) {
                throw new UnsupportedOperationException("in-memory test store");
            }

            @Override public java.io.PrintWriter getLogWriter() {
                return null;
            }

            @Override public void setLogWriter(java.io.PrintWriter out) {
            }

            @Override public void setLoginTimeout(int seconds) {
            }

            @Override public int getLoginTimeout() {
                return 0;
            }

            @Override public java.util.logging.Logger getParentLogger() {
                throw new UnsupportedOperationException("in-memory test store");
            }

            @Override public <T> T unwrap(Class<T> iface) {
                throw new UnsupportedOperationException("in-memory test store");
            }

            @Override public boolean isWrapperFor(Class<?> iface) {
                return false;
            }
        };
    }

    @Override
    public List<ChatMessage> getMessages(String conversationId, String stateKey) {
        return new ArrayList<>(data.getOrDefault(key(conversationId, stateKey), List.of()));
    }

    @Override
    public void appendMessages(String conversationId, long sessionId,
                               String stateKey, List<ChatMessage> messages) {
        List<ChatMessage> list = data.computeIfAbsent(key(conversationId, stateKey), x -> new ArrayList<>());
        for (ChatMessage message : messages) {
            if (!(message instanceof SystemMessage)) {
                list.add(message);
            }
        }
    }

    @Override
    public void replaceMessages(String conversationId, long sessionId,
                                String stateKey, List<ChatMessage> messages) {
        data.put(key(conversationId, stateKey), new ArrayList<>());
        appendMessages(conversationId, sessionId, stateKey, messages);
    }

    private String key(String conversationId, String stateKey) {
        return conversationId + "|" + stateKey;
    }
}
