import sqlite3
from pathlib import Path
import time
import uuid

# Function to generate a timestamped DB name
def generate_db_name():
    db_dir = Path(__file__).parent.parent / 'data'
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / 'codex.db'

def check_and_create_db():
    db_path = generate_db_name()
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Basic schema check
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='threads';")
        if cursor.fetchone() is None:
            raise ValueError("Missing schema")

        # Ensure default thread exists
        cursor.execute("SELECT threadUid FROM threads LIMIT 1;")
        if cursor.fetchone() is None:
            default_uid = str(uuid.uuid4())
            cursor.execute("INSERT INTO threads (threadUid, threadName, createdAt) VALUES (?, ?, ?);", 
                (default_uid, "default", int(time.time())))
            conn.commit()

        return conn

    except (sqlite3.DatabaseError, ValueError):
        return create_new_db()

def create_new_db():
    db_path = generate_db_name()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''CREATE TABLE IF NOT EXISTS threads (
        threadUid TEXT PRIMARY KEY,
        threadName TEXT NOT NULL UNIQUE,
        createdAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        isActive BOOLEAN DEFAULT TRUE
    )''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS messages (
        messageId TEXT PRIMARY KEY,
        threadUid TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        messageType TEXT DEFAULT 'regular',
        FOREIGN KEY (threadUid) REFERENCES threads(threadUid)
    )''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS facts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        threadUid TEXT,
        pair_id TEXT,
        fact TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS tags (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        threadUid TEXT,
        pair_id TEXT,
        tag TEXT
    )''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS user_info (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        key TEXT NOT NULL,
        value TEXT NOT NULL,
        confidence REAL DEFAULT 1.0,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')

    conn.commit()
    get_or_create_active_thread_id(conn)
    return conn

def initialize_codex_schema():
    conn = check_and_create_db()
    conn.close()

def get_or_create_active_thread_id(conn):
    cursor = conn.cursor()

    # Try to find an active thread
    cursor.execute("SELECT threadUid FROM threads WHERE isActive = TRUE LIMIT 1")
    result = cursor.fetchone()
    if result:
        return result[0]

    # Try to find an existing thread named 'default'
    cursor.execute("SELECT threadUid FROM threads WHERE threadName = ?", ("default",))
    existing = cursor.fetchone()
    if existing:
        thread_uid = existing[0]
        cursor.execute("UPDATE threads SET isActive = TRUE WHERE threadUid = ?", (thread_uid,))
        conn.commit()
        return thread_uid

    # Create new 'default' thread if it doesn't exist
    thread_uid = str(uuid.uuid4())
    thread_name = "default"
    cursor.execute(
        "INSERT INTO threads (threadUid, threadName, isActive) VALUES (?, ?, ?)",
        (thread_uid, thread_name, True)
    )
    conn.commit()
    return thread_uid


def append_to_thread(thread_uid: str, role: str, content: str):
    conn = check_and_create_db()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO messages (messageId, threadUid, role, content) VALUES (?, ?, ?, ?)",
                   (str(uuid.uuid4()), thread_uid, role, content))
    conn.commit()
    conn.close()

def get_thread_history(conn, thread_uid):
    cursor = conn.cursor()
    cursor.execute("SELECT role, content FROM messages WHERE threadUid = ? ORDER BY timestamp ASC", (thread_uid,))
    rows = cursor.fetchall()
    return [{"role": row[0], "content": row[1]} for row in rows]


def clear_thread_history(thread_uid: str):
    conn = check_and_create_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM messages WHERE threadUid = ?", (thread_uid,))
    conn.commit()
    conn.close()

def list_all_threads():
    conn = check_and_create_db()
    cursor = conn.cursor()
    cursor.execute("SELECT threadUid, threadName, createdAt, isActive FROM threads ORDER BY createdAt DESC")
    rows = cursor.fetchall()
    conn.close()
    return [
        {"thread_uid": uid, "thread_name": name, "created_at": created, "is_active": bool(active)}
        for uid, name, created, active in rows
    ]

def insert_fact(thread_uid: str, pair_id: str, fact: str):
    conn = check_and_create_db()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO facts (threadUid, pair_id, fact) VALUES (?, ?, ?)", (thread_uid, pair_id, fact))
    conn.commit()
    conn.close()

def insert_tag(thread_uid: str, pair_id: str, tag: str):
    conn = check_and_create_db()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO tags (threadUid, pair_id, tag) VALUES (?, ?, ?)", (thread_uid, pair_id, tag))
    conn.commit()
    conn.close()

def insert_or_update_user_info(user_id: str, key: str, value: str, confidence: float = 1.0):
    conn = check_and_create_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM user_info WHERE user_id = ? AND key = ?", (user_id, key))
    existing = cursor.fetchone()
    if existing:
        cursor.execute("UPDATE user_info SET value = ?, confidence = ?, last_updated = CURRENT_TIMESTAMP WHERE id = ?",
                       (value, confidence, existing[0]))
    else:
        cursor.execute("INSERT INTO user_info (user_id, key, value, confidence) VALUES (?, ?, ?, ?)",
                       (user_id, key, value, confidence))
    conn.commit()
    conn.close()

def get_user_info(user_id: str):
    conn = check_and_create_db()
    cursor = conn.cursor()
    cursor.execute("SELECT key, value, confidence FROM user_info WHERE user_id = ?", (user_id,))
    data = cursor.fetchall()
    conn.close()
    return {row[0]: {"value": row[1], "confidence": row[2]} for row in data}