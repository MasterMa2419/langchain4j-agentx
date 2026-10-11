package com.agentx.ai.rag.retrieve;

import com.agentx.ai.rag.common.MetadataKeys;
import com.agentx.ai.rag.common.RagDocument;
import com.agentx.ai.rag.query.Query;
import com.agentx.ai.rag.store.DocumentStore;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * 父子检索器。
 *
 * 装饰任意 DocumentRetriever，命中子块后按 parentChunkId 从 DocumentStore 回查父块，
 * 多个子块命中同一父块时自动去重。
 *
 * @author bigchui
 */
public final class ParentChildDocumentRetriever implements DocumentRetriever {

    private final DocumentRetriever baseRetriever;
    private final DocumentStore documentStore;

    public ParentChildDocumentRetriever(DocumentRetriever baseRetriever, DocumentStore documentStore) {
        this.baseRetriever = Objects.requireNonNull(baseRetriever, "baseRetriever");
        this.documentStore = Objects.requireNonNull(documentStore, "documentStore");
    }

    @Override
    public List<RagDocument> retrieve(Query query) {
        List<RagDocument> children = baseRetriever.retrieve(query);
        Set<String> seen = new HashSet<>();
        List<RagDocument> parents = new ArrayList<>();
        for (RagDocument child : children) {
            String parentId = asString(child.getMetadata().get(MetadataKeys.PARENT_CHUNK_ID));
            if (parentId == null || !seen.add(parentId)) {
                continue;
            }
            RagDocument parent = documentStore.get(parentId);
            if (parent != null) {
                parents.add(parent);
            }
        }
        return parents;
    }

    private static String asString(Object value) {
        return value == null ? null : value.toString();
    }
}
