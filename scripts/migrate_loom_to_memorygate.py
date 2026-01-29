#!/usr/bin/env python3
"""
Loom to MemoryGate Conversation Migration Script.

Migrates conversation data from Loom tables to MemoryGate conversation tables:
- loom_threads -> mg_conversation_threads
- loom_messages -> mg_conversation_messages
- loom_message_embeddings -> mg_conversation_embeddings
- loom_summaries -> mg_conversation_summaries

Note: loom_facts, loom_tags, and loom_user_info are not migrated as they
map to MemoryGate's observation/pattern system rather than conversations.

Usage:
    python scripts/migrate_loom_to_memorygate.py [--dry-run] [--batch-size N]

Options:
    --dry-run       Show what would be migrated without making changes
    --batch-size N  Number of records to process per batch (default: 100)
    --skip-existing Skip records that already exist in destination
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


def get_database_url() -> str:
    """Get database URL from environment."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL environment variable is required")
    return url


def migrate_threads(session, dry_run: bool = False, skip_existing: bool = False) -> int:
    """Migrate loom_threads to mg_conversation_threads."""
    print("\n[1/4] Migrating threads...")

    # Check if source table exists
    result = session.execute(text("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_name = 'loom_threads'
        )
    """))
    if not result.scalar():
        print("  Source table 'loom_threads' does not exist. Skipping.")
        return 0

    # Get count
    count_result = session.execute(text("SELECT COUNT(*) FROM loom_threads"))
    total = count_result.scalar()
    print(f"  Found {total} threads to migrate")

    if dry_run:
        return total

    # Ensure destination table exists
    session.execute(text("""
        CREATE TABLE IF NOT EXISTS mg_conversation_threads (
            id SERIAL PRIMARY KEY,
            thread_uid VARCHAR(36) UNIQUE NOT NULL,
            thread_name VARCHAR(255) NOT NULL DEFAULT 'New Thread',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT FALSE,
            personality_id VARCHAR(100),
            metadata_json TEXT
        )
    """))

    # Migrate data
    if skip_existing:
        insert_sql = """
            INSERT INTO mg_conversation_threads
                (thread_uid, thread_name, created_at, updated_at, is_active)
            SELECT thread_uid, thread_name, created_at, updated_at, is_active
            FROM loom_threads lt
            WHERE NOT EXISTS (
                SELECT 1 FROM mg_conversation_threads mct
                WHERE mct.thread_uid = lt.thread_uid
            )
        """
    else:
        insert_sql = """
            INSERT INTO mg_conversation_threads
                (thread_uid, thread_name, created_at, updated_at, is_active)
            SELECT thread_uid, thread_name, created_at, updated_at, is_active
            FROM loom_threads
            ON CONFLICT (thread_uid) DO UPDATE SET
                thread_name = EXCLUDED.thread_name,
                updated_at = EXCLUDED.updated_at,
                is_active = EXCLUDED.is_active
        """

    result = session.execute(text(insert_sql))
    migrated = result.rowcount
    print(f"  Migrated {migrated} threads")
    return migrated


def migrate_messages(session, dry_run: bool = False, skip_existing: bool = False, batch_size: int = 100) -> int:
    """Migrate loom_messages to mg_conversation_messages."""
    print("\n[2/4] Migrating messages...")

    # Check if source table exists
    result = session.execute(text("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_name = 'loom_messages'
        )
    """))
    if not result.scalar():
        print("  Source table 'loom_messages' does not exist. Skipping.")
        return 0

    # Get count
    count_result = session.execute(text("SELECT COUNT(*) FROM loom_messages"))
    total = count_result.scalar()
    print(f"  Found {total} messages to migrate")

    if dry_run:
        return total

    # Ensure destination table exists
    session.execute(text("""
        CREATE TABLE IF NOT EXISTS mg_conversation_messages (
            id SERIAL PRIMARY KEY,
            message_uid VARCHAR(36) UNIQUE NOT NULL,
            thread_uid VARCHAR(36) NOT NULL REFERENCES mg_conversation_threads(thread_uid),
            role VARCHAR(50) NOT NULL,
            content TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            message_type VARCHAR(50) DEFAULT 'regular',
            model_used VARCHAR(100),
            token_count INTEGER,
            metadata_json TEXT
        )
    """))

    # Migrate in batches
    offset = 0
    migrated = 0

    while offset < total:
        if skip_existing:
            batch_sql = f"""
                INSERT INTO mg_conversation_messages
                    (message_uid, thread_uid, role, content, timestamp, message_type)
                SELECT message_uid, thread_uid, role, content, timestamp, message_type
                FROM loom_messages lm
                WHERE NOT EXISTS (
                    SELECT 1 FROM mg_conversation_messages mcm
                    WHERE mcm.message_uid = lm.message_uid
                )
                AND lm.thread_uid IN (SELECT thread_uid FROM mg_conversation_threads)
                ORDER BY lm.id
                LIMIT {batch_size} OFFSET {offset}
            """
        else:
            batch_sql = f"""
                INSERT INTO mg_conversation_messages
                    (message_uid, thread_uid, role, content, timestamp, message_type)
                SELECT message_uid, thread_uid, role, content, timestamp, message_type
                FROM loom_messages
                WHERE thread_uid IN (SELECT thread_uid FROM mg_conversation_threads)
                ORDER BY id
                LIMIT {batch_size} OFFSET {offset}
                ON CONFLICT (message_uid) DO UPDATE SET
                    content = EXCLUDED.content,
                    message_type = EXCLUDED.message_type
            """

        result = session.execute(text(batch_sql))
        batch_migrated = result.rowcount
        migrated += batch_migrated
        offset += batch_size
        print(f"  Progress: {min(offset, total)}/{total} processed, {migrated} migrated")

    print(f"  Migrated {migrated} messages")
    return migrated


def migrate_embeddings(session, dry_run: bool = False, skip_existing: bool = False, batch_size: int = 100) -> int:
    """Migrate loom_message_embeddings to mg_conversation_embeddings."""
    print("\n[3/4] Migrating embeddings...")

    # Check if source table exists
    result = session.execute(text("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_name = 'loom_message_embeddings'
        )
    """))
    if not result.scalar():
        print("  Source table 'loom_message_embeddings' does not exist. Skipping.")
        return 0

    # Get count
    count_result = session.execute(text("SELECT COUNT(*) FROM loom_message_embeddings"))
    total = count_result.scalar()
    print(f"  Found {total} embeddings to migrate")

    if dry_run:
        return total

    # Ensure pgvector extension and destination table exist
    session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    session.execute(text("""
        CREATE TABLE IF NOT EXISTS mg_conversation_embeddings (
            id SERIAL PRIMARY KEY,
            message_uid VARCHAR(36) UNIQUE NOT NULL REFERENCES mg_conversation_messages(message_uid) ON DELETE CASCADE,
            embedding vector(1536),
            model_version VARCHAR(100) DEFAULT 'text-embedding-3-small',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """))

    # Migrate in batches
    offset = 0
    migrated = 0

    while offset < total:
        if skip_existing:
            batch_sql = f"""
                INSERT INTO mg_conversation_embeddings
                    (message_uid, embedding, model_version, created_at)
                SELECT lme.message_uid, lme.embedding, lme.model_version, lme.created_at
                FROM loom_message_embeddings lme
                WHERE NOT EXISTS (
                    SELECT 1 FROM mg_conversation_embeddings mce
                    WHERE mce.message_uid = lme.message_uid
                )
                AND lme.message_uid IN (SELECT message_uid FROM mg_conversation_messages)
                ORDER BY lme.id
                LIMIT {batch_size} OFFSET {offset}
            """
        else:
            batch_sql = f"""
                INSERT INTO mg_conversation_embeddings
                    (message_uid, embedding, model_version, created_at)
                SELECT message_uid, embedding, model_version, created_at
                FROM loom_message_embeddings
                WHERE message_uid IN (SELECT message_uid FROM mg_conversation_messages)
                ORDER BY id
                LIMIT {batch_size} OFFSET {offset}
                ON CONFLICT (message_uid) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    model_version = EXCLUDED.model_version
            """

        result = session.execute(text(batch_sql))
        batch_migrated = result.rowcount
        migrated += batch_migrated
        offset += batch_size
        print(f"  Progress: {min(offset, total)}/{total} processed, {migrated} migrated")

    print(f"  Migrated {migrated} embeddings")
    return migrated


def migrate_summaries(session, dry_run: bool = False, skip_existing: bool = False, batch_size: int = 100) -> int:
    """Migrate loom_summaries to mg_conversation_summaries."""
    print("\n[4/4] Migrating summaries...")

    # Check if source table exists
    result = session.execute(text("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_name = 'loom_summaries'
        )
    """))
    if not result.scalar():
        print("  Source table 'loom_summaries' does not exist. Skipping.")
        return 0

    # Get count
    count_result = session.execute(text("SELECT COUNT(*) FROM loom_summaries"))
    total = count_result.scalar()
    print(f"  Found {total} summaries to migrate")

    if dry_run:
        return total

    # Ensure destination table exists
    session.execute(text("""
        CREATE TABLE IF NOT EXISTS mg_conversation_summaries (
            id SERIAL PRIMARY KEY,
            thread_uid VARCHAR(36) NOT NULL REFERENCES mg_conversation_threads(thread_uid) ON DELETE CASCADE,
            summary_text TEXT NOT NULL,
            message_range_start INTEGER,
            message_range_end INTEGER,
            message_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            embedding vector(1536),
            model_version VARCHAR(100) DEFAULT 'text-embedding-3-small'
        )
    """))

    # Migrate in batches (summaries don't have a unique constraint to conflict on)
    if skip_existing:
        # For summaries, we check by thread_uid and message_range
        batch_sql = f"""
            INSERT INTO mg_conversation_summaries
                (thread_uid, summary_text, message_range_start, message_range_end, created_at, embedding)
            SELECT ls.thread_uid, ls.summary_text, ls.message_range_start, ls.message_range_end, ls.created_at, ls.embedding
            FROM loom_summaries ls
            WHERE ls.thread_uid IN (SELECT thread_uid FROM mg_conversation_threads)
            AND NOT EXISTS (
                SELECT 1 FROM mg_conversation_summaries mcs
                WHERE mcs.thread_uid = ls.thread_uid
                AND mcs.message_range_start = ls.message_range_start
                AND mcs.message_range_end = ls.message_range_end
            )
        """
    else:
        batch_sql = """
            INSERT INTO mg_conversation_summaries
                (thread_uid, summary_text, message_range_start, message_range_end, created_at, embedding)
            SELECT thread_uid, summary_text, message_range_start, message_range_end, created_at, embedding
            FROM loom_summaries
            WHERE thread_uid IN (SELECT thread_uid FROM mg_conversation_threads)
        """

    result = session.execute(text(batch_sql))
    migrated = result.rowcount
    print(f"  Migrated {migrated} summaries")
    return migrated


def create_indexes(session, dry_run: bool = False):
    """Create indexes on destination tables."""
    print("\n[Post] Creating indexes...")

    if dry_run:
        print("  Would create indexes (dry run)")
        return

    indexes = [
        ("ix_mg_threads_active", "mg_conversation_threads", "is_active"),
        ("ix_mg_threads_created", "mg_conversation_threads", "created_at"),
        ("ix_mg_threads_updated", "mg_conversation_threads", "updated_at"),
        ("ix_mg_messages_thread", "mg_conversation_messages", "thread_uid"),
        ("ix_mg_messages_timestamp", "mg_conversation_messages", "timestamp"),
        ("ix_mg_messages_role", "mg_conversation_messages", "role"),
        ("ix_mg_messages_type", "mg_conversation_messages", "message_type"),
        ("ix_mg_embeddings_message", "mg_conversation_embeddings", "message_uid"),
        ("ix_mg_summaries_thread", "mg_conversation_summaries", "thread_uid"),
        ("ix_mg_summaries_created", "mg_conversation_summaries", "created_at"),
    ]

    for idx_name, table, column in indexes:
        try:
            session.execute(text(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table} ({column})"))
            print(f"  Created index {idx_name}")
        except Exception as e:
            print(f"  Warning: Could not create index {idx_name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Migrate Loom data to MemoryGate conversation tables")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be migrated without changes")
    parser.add_argument("--batch-size", type=int, default=100, help="Records per batch (default: 100)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip records that already exist")
    args = parser.parse_args()

    print("=" * 60)
    print("Loom to MemoryGate Conversation Migration")
    print("=" * 60)

    if args.dry_run:
        print("\n*** DRY RUN MODE - No changes will be made ***\n")

    # Connect to database
    try:
        database_url = get_database_url()
        engine = create_engine(database_url)
        Session = sessionmaker(bind=engine)
        session = Session()
        print(f"Connected to database")
    except Exception as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)

    try:
        # Run migrations
        stats = {
            "threads": migrate_threads(session, args.dry_run, args.skip_existing),
            "messages": migrate_messages(session, args.dry_run, args.skip_existing, args.batch_size),
            "embeddings": migrate_embeddings(session, args.dry_run, args.skip_existing, args.batch_size),
            "summaries": migrate_summaries(session, args.dry_run, args.skip_existing, args.batch_size),
        }

        # Create indexes
        create_indexes(session, args.dry_run)

        # Commit changes
        if not args.dry_run:
            session.commit()
            print("\n*** Migration committed successfully ***")

        # Print summary
        print("\n" + "=" * 60)
        print("Migration Summary")
        print("=" * 60)
        print(f"  Threads:    {stats['threads']}")
        print(f"  Messages:   {stats['messages']}")
        print(f"  Embeddings: {stats['embeddings']}")
        print(f"  Summaries:  {stats['summaries']}")
        print("=" * 60)

        if args.dry_run:
            print("\nRun without --dry-run to execute migration.")
        else:
            print("\nMigration complete!")
            print("\nNote: Facts, Tags, and UserInfo from Loom are not migrated here.")
            print("They should be migrated to MemoryGate's observation/pattern system separately.")

    except Exception as e:
        session.rollback()
        print(f"\nError during migration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        session.close()


if __name__ == "__main__":
    main()
