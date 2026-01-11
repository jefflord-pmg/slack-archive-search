#!/usr/bin/env python3
"""
Slack Export to SQLite Importer

Imports Slack export JSON files into a SQLite database for easy querying.
Supports incremental updates - run again to import only new files.

Usage:
    python import_slack.py              # Import all new files
    python import_slack.py --rebuild    # Drop and rebuild entire database
"""

import sqlite3
import json
import os
import sys
from pathlib import Path
from datetime import datetime

DB_NAME = "slack_messages.db"
BASE_DIR = Path(__file__).parent


def create_schema(conn):
    """Create database tables if they don't exist."""
    cursor = conn.cursor()

    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            username TEXT,
            display_name TEXT
        )
    """)

    # Channels table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS channels (
            channel_id TEXT PRIMARY KEY,
            channel_name TEXT
        )
    """)

    # Messages table - the main one
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            channel TEXT,
            user_id TEXT,
            username TEXT,
            display_name TEXT,
            text TEXT,
            ts TEXT,
            date TEXT,
            thread_ts TEXT,
            has_files INTEGER DEFAULT 0,
            reactions TEXT,
            source_file TEXT,
            UNIQUE(channel, ts)
        )
    """)

    # Track imported files for incremental updates
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS imported_files (
            file_path TEXT PRIMARY KEY,
            imported_at TEXT
        )
    """)

    # Create indexes for fast searching
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_text ON messages(text)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(username)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_display_name ON messages(display_name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_channel ON messages(channel)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_date ON messages(date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_ts ON messages(ts)")

    # Full-text search virtual table
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
            text,
            username,
            display_name,
            channel,
            content=messages,
            content_rowid=id
        )
    """)

    # Triggers to keep FTS in sync
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
            INSERT INTO messages_fts(rowid, text, username, display_name, channel)
            VALUES (new.id, new.text, new.username, new.display_name, new.channel);
        END
    """)

    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
            INSERT INTO messages_fts(messages_fts, rowid, text, username, display_name, channel)
            VALUES ('delete', old.id, old.text, old.username, old.display_name, old.channel);
        END
    """)

    conn.commit()


def load_lookups():
    """Load user and channel lookup tables."""
    users = {}
    channels = {}
    display_names = {}

    # Load users
    users_file = BASE_DIR / "users.json"
    if users_file.exists():
        with open(users_file, 'r', encoding='utf-8') as f:
            users = json.load(f)

    # Load channels
    channels_file = BASE_DIR / "channels.json"
    if channels_file.exists():
        with open(channels_file, 'r', encoding='utf-8') as f:
            channels = json.load(f)

    # Load display names
    display_names_file = BASE_DIR / "display_names.json"
    if display_names_file.exists():
        with open(display_names_file, 'r', encoding='utf-8') as f:
            display_names = json.load(f)

    return users, channels, display_names


def populate_lookup_tables(conn, users, channels, display_names):
    """Populate users and channels tables."""
    cursor = conn.cursor()

    # Insert users
    for user_id, username in users.items():
        display_name = display_names.get(username, "")
        cursor.execute(
            "INSERT OR REPLACE INTO users (user_id, username, display_name) VALUES (?, ?, ?)",
            (user_id, username, display_name)
        )

    # Insert channels
    for channel_id, channel_name in channels.items():
        cursor.execute(
            "INSERT OR REPLACE INTO channels (channel_id, channel_name) VALUES (?, ?)",
            (channel_id, channel_name)
        )

    conn.commit()


def get_imported_files(conn):
    """Get set of already imported file paths."""
    cursor = conn.cursor()
    cursor.execute("SELECT file_path FROM imported_files")
    return set(row[0] for row in cursor.fetchall())


def find_message_files():
    """Find all JSON message files in channel directories."""
    message_files = []

    for item in BASE_DIR.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # Skip non-channel directories
            if item.name in ['__pycache__']:
                continue

            # Look for JSON files in this directory
            for json_file in item.glob("*.json"):
                message_files.append(json_file)

    return message_files


def parse_message_file(file_path, users, display_names):
    """Parse a single message JSON file and return message data."""
    channel = file_path.parent.name

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"  Warning: Could not parse {file_path}: {e}")
        return None

    # Handle both single message and array of messages
    if isinstance(data, list):
        messages = data
    else:
        messages = [data]

    results = []
    for msg in messages:
        if msg.get('type') != 'message':
            continue

        user_id = msg.get('user', '')
        username = users.get(user_id, user_id)
        display_name = display_names.get(username, '')

        # Extract text - handle different formats
        text = msg.get('text', '')

        # Get timestamp and convert to date
        ts = msg.get('ts', '')
        date = ''
        if ts:
            try:
                dt = datetime.fromtimestamp(float(ts))
                date = dt.strftime('%Y-%m-%d')
            except (ValueError, OSError):
                pass

        # Check for files
        has_files = 1 if msg.get('files') else 0

        # Get reactions as simple string
        reactions = ''
        if msg.get('reactions'):
            reaction_names = [r.get('name', '') for r in msg['reactions']]
            reactions = ','.join(reaction_names)

        thread_ts = msg.get('thread_ts', '')

        results.append({
            'channel': channel,
            'user_id': user_id,
            'username': username,
            'display_name': display_name,
            'text': text,
            'ts': ts,
            'date': date,
            'thread_ts': thread_ts,
            'has_files': has_files,
            'reactions': reactions,
            'source_file': str(file_path.relative_to(BASE_DIR))
        })

    return results


def import_messages(conn, message_files, users, display_names, imported_files):
    """Import messages from files that haven't been imported yet."""
    cursor = conn.cursor()

    new_files = [f for f in message_files if str(f.relative_to(BASE_DIR)) not in imported_files]

    if not new_files:
        print("No new files to import.")
        return 0

    print(f"Importing {len(new_files)} new files...")

    imported_count = 0
    message_count = 0

    for file_path in new_files:
        messages = parse_message_file(file_path, users, display_names)

        if messages is None:
            continue

        for msg in messages:
            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO messages
                    (channel, user_id, username, display_name, text, ts, date, thread_ts, has_files, reactions, source_file)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    msg['channel'],
                    msg['user_id'],
                    msg['username'],
                    msg['display_name'],
                    msg['text'],
                    msg['ts'],
                    msg['date'],
                    msg['thread_ts'],
                    msg['has_files'],
                    msg['reactions'],
                    msg['source_file']
                ))
                message_count += 1
            except sqlite3.Error as e:
                print(f"  Warning: Could not insert message: {e}")

        # Mark file as imported
        rel_path = str(file_path.relative_to(BASE_DIR))
        cursor.execute(
            "INSERT OR REPLACE INTO imported_files (file_path, imported_at) VALUES (?, ?)",
            (rel_path, datetime.now().isoformat())
        )

        imported_count += 1

        # Progress indicator
        if imported_count % 500 == 0:
            print(f"  Processed {imported_count} files...")
            conn.commit()

    conn.commit()
    print(f"Imported {message_count} messages from {imported_count} files.")
    return imported_count


def rebuild_fts(conn):
    """Rebuild the FTS index."""
    cursor = conn.cursor()
    cursor.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")
    conn.commit()


def main():
    rebuild = '--rebuild' in sys.argv

    db_path = BASE_DIR / DB_NAME

    if rebuild and db_path.exists():
        print("Rebuilding database from scratch...")
        os.remove(db_path)

    conn = sqlite3.connect(db_path)

    print("Creating schema...")
    create_schema(conn)

    print("Loading lookup files...")
    users, channels, display_names = load_lookups()

    print("Populating lookup tables...")
    populate_lookup_tables(conn, users, channels, display_names)

    print("Finding message files...")
    message_files = find_message_files()
    print(f"Found {len(message_files)} message files.")

    imported_files = set() if rebuild else get_imported_files(conn)

    import_messages(conn, message_files, users, display_names, imported_files)

    # Get stats
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM messages")
    total_messages = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT channel) FROM messages")
    total_channels = cursor.fetchone()[0]

    print(f"\nDatabase ready: {db_path}")
    print(f"Total messages: {total_messages:,}")
    print(f"Total channels: {total_channels}")

    conn.close()


if __name__ == "__main__":
    main()
