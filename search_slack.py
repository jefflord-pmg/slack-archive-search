#!/usr/bin/env python3
"""
Semantic Search for Slack Messages

Uses ChromaDB vector embeddings for natural language search.

Usage:
    python search_slack.py "when did chris ask about booking travel"
    python search_slack.py "API authentication issues" --user jlord
    python search_slack.py "project deadline" --channel pmg-devs
    python search_slack.py "deployment problems" --after 2024-01-01
"""

import argparse
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).parent
CHROMA_DIR = BASE_DIR / "chroma_db"

# Cache model globally for faster repeated queries
_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model


def search(query, n_results=10, user=None, channel=None, after=None, before=None):
    """
    Semantic search for messages.

    Args:
        query: Natural language search query
        n_results: Number of results to return
        user: Filter by username or display name
        channel: Filter by channel name
        after: Filter messages after this date (YYYY-MM-DD)
        before: Filter messages before this date (YYYY-MM-DD)
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection("slack_messages")

    model = get_model()
    query_embedding = model.encode(query).tolist()

    # Build where filter
    where_filter = None
    where_conditions = []

    if user:
        # Match either username or display_name (case insensitive via multiple conditions)
        where_conditions.append({
            "$or": [
                {"username": {"$eq": user}},
                {"username": {"$eq": user.lower()}},
                {"display_name": {"$eq": user}},
            ]
        })

    if channel:
        where_conditions.append({"channel": {"$eq": channel}})

    if after:
        where_conditions.append({"date": {"$gte": after}})

    if before:
        where_conditions.append({"date": {"$lte": before}})

    if len(where_conditions) == 1:
        where_filter = where_conditions[0]
    elif len(where_conditions) > 1:
        where_filter = {"$and": where_conditions}

    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )

    return results


def format_results(results):
    """Format search results for display."""
    if not results['ids'][0]:
        return "No results found."

    output = []
    for i, (doc, meta, dist) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        # Convert distance to similarity score (cosine distance)
        similarity = 1 - dist
        name = meta.get('display_name') or meta.get('username') or 'Unknown'
        channel = meta.get('channel', 'unknown')
        date = meta.get('date', 'unknown')

        output.append(f"[{i+1}] {similarity:.0%} match - {date} | {name} in #{channel}")

        # Truncate long messages
        text = doc[:500] + "..." if len(doc) > 500 else doc
        output.append(f"    {text}")
        output.append("")

    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(description="Semantic search for Slack messages")
    parser.add_argument("query", help="Natural language search query")
    parser.add_argument("-n", "--num", type=int, default=10, help="Number of results (default 10)")
    parser.add_argument("-u", "--user", help="Filter by username or display name")
    parser.add_argument("-c", "--channel", help="Filter by channel")
    parser.add_argument("--after", help="Messages after date (YYYY-MM-DD)")
    parser.add_argument("--before", help="Messages before date (YYYY-MM-DD)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    if not CHROMA_DIR.exists():
        print("Error: ChromaDB not found. Run 'python embed_messages.py' first.")
        sys.exit(1)

    results = search(
        args.query,
        n_results=args.num,
        user=args.user,
        channel=args.channel,
        after=args.after,
        before=args.before
    )

    if args.json:
        import json
        output = []
        for doc, meta, dist in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            output.append({
                "text": doc,
                "similarity": 1 - dist,
                **meta
            })
        print(json.dumps(output, indent=2))
    else:
        print(format_results(results))


if __name__ == "__main__":
    main()
