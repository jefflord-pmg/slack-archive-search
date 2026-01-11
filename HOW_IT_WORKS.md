# How the Slack Message Search System Works

## TL;DR

When you ask a question like "what is our health insurance company?", I:

1. **Semantic Search (ChromaDB)** - Search for messages with similar *meaning*, not just exact keywords. "health insurance company" finds messages about "Cigna", "ACI", "self-funded", etc.
2. **SQL Queries (SQLite)** - Filter/refine by user, date, channel, or do aggregations (counts, rankings).
3. **Combine & Synthesize** - Read the relevant messages and synthesize an answer.

**The key insight:** ChromaDB finds *what* you're looking for (semantic meaning), SQLite helps filter *who/when/where*.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Your Question                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Determine Search Strategy                       │
│  - Factual lookup? → Semantic search first                  │
│  - Aggregation? → SQL query                                 │
│  - Person-specific? → Filter by user                        │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│      ChromaDB            │    │        SQLite            │
│   (Semantic Search)      │    │    (Structured Data)     │
│                          │    │                          │
│ - Vector embeddings      │    │ - Exact keyword search   │
│ - Finds similar meaning  │    │ - Date/user/channel      │
│ - Natural language       │    │ - Aggregations (COUNT)   │
│ - Fuzzy matching         │    │ - Rankings, stats        │
└──────────────────────────┘    └──────────────────────────┘
              │                               │
              └───────────────┬───────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Synthesize Answer                          │
│  - Read relevant messages                                   │
│  - Extract key facts                                        │
│  - Present with context (who said it, when)                 │
└─────────────────────────────────────────────────────────────┘
```

---

## The Two Search Systems

### 1. ChromaDB (Semantic/Vector Search)

**What it does:** Finds messages with similar *meaning*, even if they use different words.

**How it works:**
- Every message was converted to a 384-dimensional vector using the `all-MiniLM-L6-v2` model
- Your query is converted to the same vector space
- ChromaDB finds messages whose vectors are closest (cosine similarity)

**When to use it:**
- Natural language questions
- When you don't know the exact wording
- Finding conceptually related messages

**Example:**
```bash
python search_slack.py "health insurance company provider"
```
This finds messages about "Cigna", "ACI", "self-funded", "PHCS network" - none of which contain "health insurance company" exactly.

### 2. SQLite (Structured/Keyword Search)

**What it does:** Exact keyword matching, filtering, and aggregations.

**When to use it:**
- Counting things ("who talks about X the most?")
- Filtering by user, date, or channel
- Exact phrase matching
- Statistical queries

**Example:**
```sql
SELECT username, COUNT(*)
FROM messages
WHERE LOWER(text) LIKE '%typescript%'
GROUP BY username
ORDER BY COUNT(*) DESC
```

---

## Question Types & How I Handle Them

### Type 1: Factual Lookup
**"What is our health insurance company?"**

1. Run semantic search: `"health insurance company provider benefits"`
2. Scan results for relevant info
3. If needed, do follow-up SQL searches for specific terms found
4. Synthesize answer from multiple messages

### Type 2: Person's Opinion/Activity
**"What does Robert think about AI?"**

1. Semantic search with user filter: `--user rcastles "AI opinions concerns excited"`
2. Get multiple messages showing their perspective
3. Categorize into themes (positive, negative, neutral)
4. Summarize their overall stance

### Type 3: Aggregation/Ranking
**"Who talks about TypeScript the most?"**

1. SQL query with COUNT and GROUP BY
2. Optionally calculate ratios (mentions / total messages)
3. Present as ranked list

### Type 4: Specific Event Lookup
**"When did Chris ask about plane tickets?"**

1. Semantic search: `"booking flights plane tickets travel"`
2. Filter by user if specified
3. Return date and context

### Type 5: Recent Activity
**"What did Ben say was wrong with select overlay?"**

1. Semantic search: `"select overlay problem issue bug"`
2. Filter by user: `--user balexander`
3. Results naturally sorted by relevance
4. Can add date filter if needed: `--after 2025-12-01`

---

## Detailed Example: Health Insurance Question

**Question:** "What do I say if I am asked what our health insurance company is?"

### Step 1: Initial Semantic Search
```bash
python search_slack.py "health insurance company provider benefits coverage"
```

This returns messages mentioning insurance, but results might have encoding issues or be too broad.

### Step 2: SQL Keyword Search
```python
SELECT date, display_name, channel, text
FROM messages
WHERE LOWER(text) LIKE '%health insurance%'
   OR LOWER(text) LIKE '%cigna%'
   OR LOWER(text) LIKE '%aetna%'
   -- ... other insurance companies
ORDER BY date DESC
LIMIT 15
```

Results show Cigna was mentioned in early 2024, but more recent messages discuss a change.

### Step 3: Narrow Down with Date Filter
```python
SELECT date, display_name, channel, text
FROM messages
WHERE (LOWER(text) LIKE '%insurance%'
   OR LOWER(text) LIKE '%aci%'
   OR LOWER(text) LIKE '%phcs%')
  AND date >= '2025-07-01'
ORDER BY date DESC
```

This reveals the recent discussion about the new self-funded plan.

### Step 4: Find the Definitive Answer
Found Jeff's message from Aug 9, 2025:
> "You can say 'My insurance is self-funded through my employer. The network is PHCS for VDHP, and the claims administrator is ACI'."

### Step 5: Synthesize Response
Combine findings:
- Current: Self-funded, PHCS network, ACI claims administrator
- Previous: Cigna (through early 2024)
- Dental: Guardian

---

## The Scripts

### `search_slack.py` - Semantic Search
```bash
# Basic search
python search_slack.py "deployment issues"

# Filter by user
python search_slack.py "API problems" --user balexander

# Filter by channel
python search_slack.py "bug fix" --channel pmg-devs

# Filter by date
python search_slack.py "release" --after 2025-01-01 --before 2025-06-01

# More results
python search_slack.py "authentication" -n 20
```

### Direct SQLite Queries (via Python)
```python
import sqlite3
conn = sqlite3.connect('slack_messages.db')
c = conn.cursor()

# Keyword search
c.execute("""
    SELECT date, username, channel, text
    FROM messages
    WHERE LOWER(text) LIKE '%search term%'
    ORDER BY date DESC
    LIMIT 10
""")

# Count by user
c.execute("""
    SELECT username, COUNT(*)
    FROM messages
    WHERE LOWER(text) LIKE '%topic%'
    GROUP BY username
    ORDER BY COUNT(*) DESC
""")

# Date range
c.execute("""
    SELECT * FROM messages
    WHERE date BETWEEN '2025-01-01' AND '2025-03-31'
    AND channel = 'pmg-devs'
""")
```

---

## Database Schema

### SQLite Tables

**messages** (main table)
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| channel | TEXT | Channel name |
| user_id | TEXT | Slack user ID |
| username | TEXT | Screen name (jlord) |
| display_name | TEXT | Full name (Jeff Lord) |
| text | TEXT | Message content |
| ts | TEXT | Slack timestamp |
| date | TEXT | YYYY-MM-DD |
| thread_ts | TEXT | Thread parent timestamp |
| has_files | INTEGER | 1 if has attachments |
| reactions | TEXT | Comma-separated emoji names |
| source_file | TEXT | Original JSON file |

**users**
| Column | Type |
|--------|------|
| user_id | TEXT |
| username | TEXT |
| display_name | TEXT |

**channels**
| Column | Type |
|--------|------|
| channel_id | TEXT |
| channel_name | TEXT |

**imported_files** (for incremental updates)
| Column | Type |
|--------|------|
| file_path | TEXT |
| imported_at | TEXT |

### ChromaDB Collection

**slack_messages** collection with metadata:
- `channel` - Channel name
- `username` - Screen name
- `display_name` - Full name
- `date` - YYYY-MM-DD
- `ts` - Timestamp
- `name` - Display name or username (for convenience)

---

## Updating the System

When new Slack export files are added:

```bash
# 1. Import new messages to SQLite
python import_slack.py

# 2. Embed new messages to ChromaDB
python embed_messages.py
```

Both scripts are incremental - they track what's already been processed and only add new content.

To rebuild from scratch:
```bash
python import_slack.py --rebuild
python embed_messages.py --rebuild
```

---

## Limitations & Tips

### What Works Well
- Finding specific discussions/decisions
- Understanding someone's opinions over time
- Finding when something was discussed
- Aggregations and statistics

### What Doesn't Work Well
- Messages with only images/files (no text to search)
- Very short messages without context
- Sarcasm/irony detection
- Messages in threads (they're there, but context can be fragmented)

### Tips for Better Results
1. **Try multiple phrasings** - If first search doesn't find it, try synonyms
2. **Use user filters** - Narrow down when you know who said it
3. **Check recent first** - Add date filters for current info
4. **Combine approaches** - Use semantic search to find topics, SQL to get counts
