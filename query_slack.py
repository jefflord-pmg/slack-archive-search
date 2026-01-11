#!/usr/bin/env python3
"""
Slack Message Query Tool

Usage:
    python query_slack.py "search term"                    # Full-text search
    python query_slack.py --user jlord "search term"       # Search by user
    python query_slack.py --channel pmg-devs "search"      # Search in channel
    python query_slack.py --sql "SELECT * FROM messages"   # Raw SQL
    python query_slack.py --stats                          # Show database stats
"""

import sqlite3
import sys
import argparse
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent / "slack_messages.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def search_messages(term, user=None, channel=None, limit=20):
    """Full-text search for messages."""
    conn = get_connection()
    c = conn.cursor()

    # Build query
    conditions = []
    params = []

    if term:
        conditions.append("messages_fts MATCH ?")
        params.append(term)

    # Join with main table for filtering
    query = """
        SELECT m.date, m.username, m.display_name, m.channel, m.text
        FROM messages m
        JOIN messages_fts fts ON m.id = fts.rowid
    """

    if user:
        conditions.append("(m.username LIKE ? OR m.display_name LIKE ?)")
        params.extend([f"%{user}%", f"%{user}%"])

    if channel:
        conditions.append("m.channel LIKE ?")
        params.append(f"%{channel}%")

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += f" ORDER BY m.date DESC LIMIT {limit}"

    results = []
    try:
        for row in c.execute(query, params):
            results.append({
                'date': row[0],
                'username': row[1],
                'display_name': row[2],
                'channel': row[3],
                'text': row[4]
            })
    except sqlite3.OperationalError as e:
        # Fall back to LIKE search if FTS fails
        query = """
            SELECT date, username, display_name, channel, text
            FROM messages
            WHERE text LIKE ?
        """
        params = [f"%{term}%"]

        if user:
            query += " AND (username LIKE ? OR display_name LIKE ?)"
            params.extend([f"%{user}%", f"%{user}%"])

        if channel:
            query += " AND channel LIKE ?"
            params.append(f"%{channel}%")

        query += f" ORDER BY date DESC LIMIT {limit}"

        for row in c.execute(query, params):
            results.append({
                'date': row[0],
                'username': row[1],
                'display_name': row[2],
                'channel': row[3],
                'text': row[4]
            })

    conn.close()
    return results


def run_sql(sql):
    """Execute raw SQL and return results."""
    conn = get_connection()
    c = conn.cursor()
    c.execute(sql)
    columns = [desc[0] for desc in c.description] if c.description else []
    rows = c.fetchall()
    conn.close()
    return columns, rows


def get_stats():
    """Get database statistics."""
    conn = get_connection()
    c = conn.cursor()

    stats = {}

    c.execute("SELECT COUNT(*) FROM messages")
    stats['total_messages'] = c.fetchone()[0]

    c.execute("SELECT COUNT(DISTINCT channel) FROM messages")
    stats['total_channels'] = c.fetchone()[0]

    c.execute("SELECT COUNT(DISTINCT username) FROM messages")
    stats['total_users'] = c.fetchone()[0]

    c.execute("SELECT MIN(date), MAX(date) FROM messages")
    row = c.fetchone()
    stats['date_range'] = f"{row[0]} to {row[1]}"

    c.execute("SELECT COUNT(*) FROM imported_files")
    stats['imported_files'] = c.fetchone()[0]

    conn.close()
    return stats


def main():
    parser = argparse.ArgumentParser(description="Query Slack messages database")
    parser.add_argument("search", nargs="?", help="Search term")
    parser.add_argument("--user", "-u", help="Filter by username or display name")
    parser.add_argument("--channel", "-c", help="Filter by channel")
    parser.add_argument("--limit", "-l", type=int, default=20, help="Max results (default 20)")
    parser.add_argument("--sql", help="Run raw SQL query")
    parser.add_argument("--stats", action="store_true", help="Show database stats")

    args = parser.parse_args()

    if args.stats:
        stats = get_stats()
        print("Database Statistics:")
        print(f"  Total messages: {stats['total_messages']:,}")
        print(f"  Total channels: {stats['total_channels']}")
        print(f"  Total users: {stats['total_users']}")
        print(f"  Date range: {stats['date_range']}")
        print(f"  Imported files: {stats['imported_files']:,}")
        return

    if args.sql:
        columns, rows = run_sql(args.sql)
        if columns:
            print(" | ".join(columns))
            print("-" * 60)
        for row in rows[:50]:  # Limit output
            print(" | ".join(str(x)[:50] for x in row))
        if len(rows) > 50:
            print(f"... and {len(rows) - 50} more rows")
        return

    if not args.search and not args.user and not args.channel:
        parser.print_help()
        return

    results = search_messages(
        args.search or "",
        user=args.user,
        channel=args.channel,
        limit=args.limit
    )

    if not results:
        print("No messages found.")
        return

    for msg in results:
        name = msg['display_name'] or msg['username']
        print(f"[{msg['date']}] #{msg['channel']} - {name}:")
        print(f"  {msg['text'][:200]}{'...' if len(msg['text']) > 200 else ''}")
        print()


if __name__ == "__main__":
    main()
