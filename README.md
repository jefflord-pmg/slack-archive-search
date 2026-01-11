# Slack Archive Search

A RAG (Retrieval Augmented Generation) system for searching Slack message exports using semantic search and SQLite.

## Features

- **Semantic Search** - Find messages by meaning, not just keywords (ChromaDB + sentence-transformers)
- **SQL Queries** - Filter by user, channel, date; run aggregations
- **Incremental Updates** - Import new Slack exports without reprocessing everything

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Place your Slack export in this folder (channel folders, users.json, channels.json)

3. Import messages to SQLite:
   ```bash
   python import_slack.py
   ```

4. Create vector embeddings:
   ```bash
   python embed_messages.py
   ```

## Usage

### Semantic Search
```bash
python search_slack.py "deployment issues"
python search_slack.py "API problems" --user jlord
python search_slack.py "bug fix" --channel dev-team
python search_slack.py "release" --after 2024-01-01
```

### SQL Queries (via Python)
```python
import sqlite3
conn = sqlite3.connect('slack_messages.db')
c = conn.cursor()
c.execute("SELECT * FROM messages WHERE text LIKE '%search term%'")
```

## Files

| File | Purpose |
|------|---------|
| `import_slack.py` | Import Slack JSON exports to SQLite |
| `embed_messages.py` | Create vector embeddings in ChromaDB |
| `search_slack.py` | Semantic search CLI |
| `query_slack.py` | SQL-based search CLI |
| `display_names.json` | Map usernames to display names |
| `CLAUDE.md` | Instructions for Claude Code |
| `HOW_IT_WORKS.md` | Detailed system documentation |

## Updating

When you have new Slack exports:
```bash
python import_slack.py      # Import new messages
python embed_messages.py    # Embed new messages
```

Both scripts are incremental - they only process new data.

## Data Not Included

This repo contains only the tooling. Your actual Slack data (messages, databases) should be added locally and is excluded via `.gitignore`.
