#!/usr/bin/env python3
"""
Embed Slack Messages into ChromaDB for Semantic Search

Creates vector embeddings of all messages using sentence-transformers
and stores them in ChromaDB for fast semantic search.

Usage:
    python embed_messages.py              # Embed new messages only
    python embed_messages.py --rebuild    # Rebuild entire vector DB
"""

import sqlite3
import sys
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).parent
SQLITE_DB = BASE_DIR / "slack_messages.db"
CHROMA_DIR = BASE_DIR / "chroma_db"
BATCH_SIZE = 500  # Process messages in batches


def get_sqlite_messages(limit=None, offset=0):
    """Fetch messages from SQLite database."""
    conn = sqlite3.connect(SQLITE_DB)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    query = """
        SELECT id, channel, username, display_name, text, date, ts
        FROM messages
        WHERE text IS NOT NULL AND text != ''
        ORDER BY id
    """
    if limit:
        query += f" LIMIT {limit} OFFSET {offset}"

    c.execute(query)
    rows = c.fetchall()
    conn.close()
    return rows


def get_message_count():
    """Get total message count from SQLite."""
    conn = sqlite3.connect(SQLITE_DB)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM messages WHERE text IS NOT NULL AND text != ''")
    count = c.fetchone()[0]
    conn.close()
    return count


def get_embedded_ids(collection):
    """Get set of already embedded message IDs."""
    try:
        # Get all IDs from collection
        result = collection.get()
        return set(result['ids']) if result['ids'] else set()
    except Exception:
        return set()


def main():
    rebuild = '--rebuild' in sys.argv

    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Initializing ChromaDB...")
    if rebuild and CHROMA_DIR.exists():
        import shutil
        shutil.rmtree(CHROMA_DIR)
        print("Removed existing ChromaDB.")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Create or get collection
    collection = client.get_or_create_collection(
        name="slack_messages",
        metadata={"hnsw:space": "cosine"}
    )

    # Get existing embedded IDs
    existing_ids = set() if rebuild else get_embedded_ids(collection)
    print(f"Already embedded: {len(existing_ids)} messages")

    # Get total count
    total_messages = get_message_count()
    print(f"Total messages in SQLite: {total_messages:,}")

    # Process in batches
    embedded_count = 0
    skipped_count = 0
    offset = 0

    while offset < total_messages:
        messages = get_sqlite_messages(limit=BATCH_SIZE, offset=offset)
        if not messages:
            break

        # Filter out already embedded
        new_messages = []
        for msg in messages:
            msg_id = str(msg['id'])
            if msg_id not in existing_ids:
                new_messages.append(msg)
            else:
                skipped_count += 1

        if new_messages:
            # Prepare batch data
            ids = []
            texts = []
            metadatas = []

            for msg in new_messages:
                # Create searchable text combining relevant fields
                name = msg['display_name'] if msg['display_name'] else msg['username']
                search_text = msg['text']

                ids.append(str(msg['id']))
                texts.append(search_text)
                metadatas.append({
                    'channel': msg['channel'] or '',
                    'username': msg['username'] or '',
                    'display_name': msg['display_name'] or '',
                    'date': msg['date'] or '',
                    'ts': msg['ts'] or '',
                    'name': name  # For easy display
                })

            # Generate embeddings
            embeddings = model.encode(texts, show_progress_bar=False).tolist()

            # Add to ChromaDB
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )

            embedded_count += len(new_messages)

        offset += BATCH_SIZE

        # Progress update
        processed = offset if offset < total_messages else total_messages
        print(f"  Processed {processed:,}/{total_messages:,} ({embedded_count:,} new, {skipped_count:,} skipped)")

    print(f"\nDone! Embedded {embedded_count:,} new messages.")
    print(f"Total in ChromaDB: {collection.count():,}")
    print(f"ChromaDB location: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
