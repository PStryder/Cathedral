import os
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from datetime import datetime
from loom.CodexGate import check_and_create_db

class VectorGate:
    """
    VectorGate handles semantic memory storage and retrieval.
    It embeds text, stores vectors in FAISS, and links to message pairs via metadata.
    """

    def __init__(self, index_path: str = "vectors/index.faiss", meta_path: str = "vectors/metadata.pkl"):
        self.index_path = index_path
        self.meta_path = meta_path
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

        # Initialize or load FAISS index
        self.dimension = 384  # MiniLM output size
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)

        # Load or initialize metadata
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            self.metadata = []

        # Get actual context pair count from SQL
        conn = check_and_create_db()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM messages WHERE messageType = 'context'")
        db_count = cursor.fetchone()[0]
        conn.close()

        index_size = self.index.ntotal
        meta_size = len(self.metadata)

        if db_count == 0:
            print("[VectorGate] No context messages found in DB. Skipping vector rebuild.")
        elif meta_size != db_count:
            print(f"[VectorGate] Warning: metadata has {meta_size} entries, but DB shows {db_count} context messages.")
            if meta_size > db_count:
                self.metadata = self.metadata[:db_count]
                print("[VectorGate] Trimming metadata to match DB.")
            elif meta_size < db_count:
                print("[VectorGate] Auto-rebuilding vector store to fix desync...")
                try:
                    self.rebuild_from_db()
                    print("[VectorGate] Recovery complete.")
                except Exception as e:
                    raise RuntimeError("Vector store rebuild failed.") from e

        # Run pruning on init to maintain memory hygiene
        self.prune_to_target(max_count=40000)

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, 'wb') as f:
            pickle.dump(self.metadata, f)

    def embed(self, text: str) -> np.ndarray:
        return self.embedder.encode([text])[0].astype(np.float32)

    def embed_batch(self, texts: list) -> np.ndarray:
        return self.embedder.encode(texts, batch_size=128, convert_to_numpy=True).astype(np.float32)

    def store(self, text: str, pair_id: str, type_: str = "context"):
        vector = self.embed(text)
        self.index.add(np.array([vector]))
        self.metadata.append({
            "pair_id": pair_id,
            "type": type_,
            "text": text,
            "created_at": datetime.utcnow(),
            "last_accessed": datetime.utcnow(),
            "access_count": 0
        })
        self.save()

    def query(self, text: str, top_k: int = 5) -> list:
        query_vec = self.embed(text)
        distances, indices = self.index.search(np.array([query_vec]), top_k)
        results = []
        now = datetime.utcnow()
        for i in indices[0]:
            if 0 <= i < len(self.metadata):
                meta = self.metadata[i]
                meta["last_accessed"] = now
                meta["access_count"] = meta.get("access_count", 0) + 1
                results.append(meta)
            else:
                print(f"[VectorGate] ⚠️ Skipped vector index {i} — no matching metadata.")
        self.save()
        return results

    def prune_to_target(self, max_count=40000, softness=0.02, max_softness=0.05):
        current_count = len(self.metadata)
        if current_count <= max_count:
            return

        excess = current_count - max_count
        softness_factor = min(max(excess / max_count, softness), max_softness)
        prune_count = int(current_count * softness_factor)

        print(f"[VectorGate] Pruning {prune_count} of {current_count} vectors to maintain target size of {max_count}.")

        # Attach indices and sort by least used + oldest
        entries = [
            {"i": i, **entry}
            for i, entry in enumerate(self.metadata)
        ]

        entries.sort(
            key=lambda x: (
                x.get("access_count", 0),
                x.get("last_accessed", x.get("created_at"))
            )
        )

        # Keep most recent N
        keep = entries[-(current_count - prune_count):]
        texts = [e["text"] for e in keep]
        new_vectors = self.embed_batch(texts)
        new_metadata = [e for e in keep]

        # Rebuild index and save
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(np.array(new_vectors, dtype=np.float32))
        self.metadata = new_metadata
        self.save()

    def delete_by_pair_id(self, pair_id: str):
        # Not implemented: FAISS does not support easy deletion
        raise NotImplementedError("Deletion from FAISS index not yet supported.")

    def rebuild_from_db(self):
        """
        Reconstructs the entire FAISS vector store and metadata from context messages in the SQL database.
        Assumes 'messageType' is 'context' and each has a unique 'messageId'.
        """
        from loom.CodexGate import check_and_create_db
        conn = check_and_create_db()
        cursor = conn.cursor()
        cursor.execute("SELECT messageId, content FROM messages WHERE messageType = 'context'")
        rows = cursor.fetchall()
        conn.close()

        print(f"[VectorGate] Rebuilding vector store from {len(rows)} context messages...")

        texts = []
        new_metadata = []

        now = datetime.utcnow()
        for message_id, content in rows:
            texts.append(content)
            new_metadata.append({
                "pair_id": message_id,
                "type": "context",
                "text": content,
                "created_at": now,
                "last_accessed": now,
                "access_count": 0
            })

        if not texts:
            print("[VectorGate] No context messages found in DB. Vector store remains empty.")
            return

        new_vectors = self.embed_batch(texts)
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(np.array(new_vectors, dtype=np.float32))
        self.metadata = new_metadata
        self.save()

        print(f"[VectorGate] Rebuild complete: {len(new_metadata)} entries embedded and stored.")

    def __repr__(self):
        return f"<VectorGate entries={len(self.metadata)}>"
