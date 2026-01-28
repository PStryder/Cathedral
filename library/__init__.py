import os
import hashlib
import pickle
from datetime import datetime

class LibraryIngestor:
    """
    Ingests text files from ./library and embeds them if they have changed.
    Stores file digests to avoid reprocessing.
    """

    def __init__(self, library_dir="library", digest_path="vectors/library_digests.pkl"):
        self.library_dir = library_dir
        self.digest_path = digest_path
        self.digests = self.load_digests()

    def load_digests(self):
        if os.path.exists(self.digest_path):
            with open(self.digest_path, 'rb') as f:
                return pickle.load(f)
        return {}

    def save_digests(self):
        with open(self.digest_path, 'wb') as f:
            pickle.dump(self.digests, f)

    def file_digest(self, path):
        hasher = hashlib.sha256()
        with open(path, 'rb') as f:
            hasher.update(f.read())
        return hasher.hexdigest()

    def should_ingest(self, filename):
        full_path = os.path.join(self.library_dir, filename)
        new_digest = self.file_digest(full_path)
        return self.digests.get(filename) != new_digest

    def ingest_file(self, filename):
        full_path = os.path.join(self.library_dir, filename)
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Save digest
        self.digests[filename] = self.file_digest(full_path)
        print(f"[Library] Ingested: {filename}")

        return {
            "text": content,
            "filename": filename,
            "ingested_at": datetime.utcnow()
        }

    def ingest_all(self):
        if not os.path.exists(self.library_dir):
            print(f"[Library] Directory {self.library_dir} does not exist.")
            return []

        records = []
        for file in os.listdir(self.library_dir):
            if file.endswith(".txt") and self.should_ingest(file):
                record = self.ingest_file(file)
                records.append(record)

        self.save_digests()
        return records
