#!/usr/bin/env python3
"""
Detect conversational micro-threads in Slack messages.

Given a target message (channel + timestamp), analyzes surrounding messages
and groups them by detected conversation topics using embeddings and LLM classification.
"""

import argparse
import json
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
import numpy as np
from openai import OpenAI

# Paths
BASE_DIR = Path(__file__).parent
SQLITE_DB = BASE_DIR / "slack_messages.db"
CHROMA_DIR = BASE_DIR / "chroma_db"

# Defaults
DEFAULT_LLM_URL = "http://10.0.0.177:1234/v1"
DEFAULT_MODEL = "local-model"
DEFAULT_WINDOW = 15


def get_db_connection():
    """Get SQLite database connection."""
    if not SQLITE_DB.exists():
        print(f"Error: Database not found at {SQLITE_DB}")
        print("Run 'python import_slack.py' first to create the database.")
        sys.exit(1)
    conn = sqlite3.connect(SQLITE_DB)
    conn.row_factory = sqlite3.Row
    return conn


def get_chroma_collection():
    """Get ChromaDB collection."""
    if not CHROMA_DIR.exists():
        print(f"Error: ChromaDB not found at {CHROMA_DIR}")
        print("Run 'python embed_messages.py' first to create embeddings.")
        sys.exit(1)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection("slack_messages")


def get_llm_client(base_url: str) -> OpenAI:
    """Get OpenAI client configured for local LLM."""
    return OpenAI(
        base_url=base_url,
        api_key="not-needed"
    )


def fetch_context_window(channel: str, ts: str, window: int = 15) -> Tuple[List[dict], Optional[dict]]:
    """
    Get N messages before and after target timestamp.
    Returns (messages, target_message) where target_message is None if not found.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Get messages before target
    before = cursor.execute("""
        SELECT id, ts, username, display_name, text, thread_ts, user_id
        FROM messages
        WHERE channel = ? AND ts < ?
        ORDER BY ts DESC
        LIMIT ?
    """, (channel, ts, window)).fetchall()

    # Get target and messages after
    after = cursor.execute("""
        SELECT id, ts, username, display_name, text, thread_ts, user_id
        FROM messages
        WHERE channel = ? AND ts >= ?
        ORDER BY ts ASC
        LIMIT ?
    """, (channel, ts, window + 1)).fetchall()

    conn.close()

    # Convert to dicts
    before = [dict(row) for row in before]
    after = [dict(row) for row in after]

    # Find target message
    target_msg = None
    for msg in after:
        if msg['ts'] == ts:
            target_msg = msg
            break

    # Combine and sort chronologically
    all_messages = sorted(before + after, key=lambda m: m['ts'])

    return all_messages, target_msg


def get_embeddings_for_messages(message_ids: List[int]) -> Dict[str, List[float]]:
    """Retrieve existing embeddings from ChromaDB."""
    collection = get_chroma_collection()
    str_ids = [str(id) for id in message_ids]

    try:
        results = collection.get(
            ids=str_ids,
            include=["embeddings"]
        )
        if results['embeddings'] is not None and len(results['embeddings']) > 0:
            return dict(zip(results['ids'], results['embeddings']))
    except Exception as e:
        print(f"Warning: Could not retrieve embeddings: {e}")

    return {}


def compute_similarity_matrix(embeddings: Dict[str, List[float]], message_ids: List[int]) -> Tuple[List[str], np.ndarray]:
    """Compute cosine similarity between all message pairs."""
    # Order by message_ids to maintain consistency
    ordered_ids = [str(id) for id in message_ids if str(id) in embeddings]

    if len(ordered_ids) < 2:
        return ordered_ids, np.array([[1.0]])

    vectors = np.array([embeddings[id] for id in ordered_ids])

    # Handle zero vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero

    normalized = vectors / norms
    similarity = np.dot(normalized, normalized.T)

    return ordered_ids, similarity


def get_similarity_hints(ids: List[str], similarity: np.ndarray, messages: List[dict], threshold: float = 0.6) -> str:
    """Build human-readable similarity hints for LLM."""
    hints = []
    id_to_idx = {msg['id']: i for i, msg in enumerate(messages)}

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            sim = similarity[i, j]
            if sim >= threshold:
                # Map back to message indices (1-indexed for LLM)
                msg_i = id_to_idx.get(int(ids[i]))
                msg_j = id_to_idx.get(int(ids[j]))
                if msg_i is not None and msg_j is not None:
                    hints.append(f"Messages {msg_i + 1} and {msg_j + 1}: {sim:.0%} similar")

    if not hints:
        return "No strong semantic similarities detected above threshold."

    return "\n".join(hints[:15])  # Limit to top 15


def extract_features(messages: List[dict]) -> dict:
    """Extract signals that indicate conversational relationships."""
    features = {
        'mentions': {},
        'participants': {},
        'time_gaps': {},
        'participant_sequence': []
    }

    for i, msg in enumerate(messages):
        msg_id = msg['id']
        text = msg.get('text', '') or ''

        # Extract @mentions (Slack format: <@U123ABC>)
        slack_mentions = re.findall(r'<@(\w+)>', text)
        # Also check for plain @username mentions
        plain_mentions = re.findall(r'@(\w+)', text)
        features['mentions'][msg_id] = slack_mentions + plain_mentions

        # Track participant
        features['participants'][msg_id] = msg.get('username', '')
        features['participant_sequence'].append(msg.get('username', ''))

        # Compute time gap from previous message
        if i > 0:
            try:
                prev_ts = float(messages[i - 1]['ts'])
                curr_ts = float(msg['ts'])
                features['time_gaps'][msg_id] = curr_ts - prev_ts
            except (ValueError, TypeError):
                features['time_gaps'][msg_id] = 0
        else:
            features['time_gaps'][msg_id] = 0

    return features


def classify_with_llm(messages: List[dict], features: dict, similarity_hints: str,
                      llm_url: str, model: str) -> List[dict]:
    """Use local LLM to identify micro-threads."""

    # Format messages for LLM (1-indexed)
    msg_lines = []
    for i, m in enumerate(messages):
        name = m.get('display_name') or m.get('username') or 'Unknown'
        text = (m.get('text') or '')[:200]
        # Show last 6 chars of timestamp for reference
        ts_short = m['ts'][-6:] if m.get('ts') else ''
        msg_lines.append(f"[{i + 1}] {name} ({ts_short}): {text}")

    msg_text = "\n".join(msg_lines)

    prompt = f"""Analyze these Slack messages and identify distinct conversation threads.

Messages (chronological order):
{msg_text}

Semantic similarity hints:
{similarity_hints}

Instructions:
1. Group messages by conversation topic/thread
2. Each message can only belong to ONE thread
3. Consider: who's talking to whom, topic continuity, @mentions, question/answer patterns
4. Name each thread briefly (2-5 words describing the topic)
5. Assign a confidence score (0.0-1.0) based on how certain you are about the grouping

Return ONLY valid JSON (no other text):
{{
  "threads": [
    {{
      "name": "Thread topic name",
      "message_indices": [1, 3, 5],
      "participants": ["user1", "user2"],
      "confidence": 0.85
    }}
  ]
}}"""

    try:
        client = get_llm_client(llm_url)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.3
        )

        content = response.choices[0].message.content

        # Try to extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            result = json.loads(json_match.group())
            return result.get('threads', [])

    except json.JSONDecodeError as e:
        print(f"Warning: Could not parse LLM response as JSON: {e}")
    except Exception as e:
        print(f"Warning: LLM classification failed: {e}")

    return []


def classify_with_embeddings_only(messages: List[dict], ids: List[str],
                                   similarity: np.ndarray, threshold: float = 0.5) -> List[dict]:
    """
    Fallback: cluster messages using only embeddings (no LLM).
    Uses a simple greedy clustering approach.
    """
    n = len(ids)
    if n == 0:
        return []

    id_to_msg_idx = {int(ids[i]): i for i in range(n)}
    msg_id_to_idx = {msg['id']: i for i, msg in enumerate(messages)}

    # Track which messages are assigned
    assigned = set()
    threads = []

    # For each message, find similar messages and form a cluster
    for i, msg in enumerate(messages):
        if msg['id'] in assigned:
            continue

        cluster_indices = [i + 1]  # 1-indexed
        assigned.add(msg['id'])

        # Find similar messages
        if msg['id'] in id_to_msg_idx:
            emb_idx = list(ids).index(str(msg['id'])) if str(msg['id']) in ids else -1
            if emb_idx >= 0:
                for j in range(n):
                    if i != j and similarity[emb_idx, j] >= threshold:
                        other_id = int(ids[j])
                        other_msg_idx = msg_id_to_idx.get(other_id)
                        if other_msg_idx is not None and other_id not in assigned:
                            cluster_indices.append(other_msg_idx + 1)
                            assigned.add(other_id)

        # Get participants
        participants = list(set(
            messages[idx - 1].get('username', '')
            for idx in cluster_indices
            if messages[idx - 1].get('username')
        ))

        # Generate thread name from first message
        first_text = messages[cluster_indices[0] - 1].get('text', '')[:50]
        thread_name = first_text if first_text else "Unnamed thread"

        threads.append({
            'name': thread_name,
            'message_indices': sorted(cluster_indices),
            'participants': participants,
            'confidence': 0.5  # Lower confidence for embeddings-only
        })

    return threads


def format_output(threads: List[dict], messages: List[dict], target_ts: str) -> str:
    """Format threads for display, highlighting target message."""
    if not threads:
        return "No threads detected."

    output = []

    for thread in threads:
        output.append(f"\n{'=' * 60}")
        confidence = thread.get('confidence', 0)
        output.append(f"THREAD: {thread['name']} ({confidence:.0%} confidence)")
        participants = thread.get('participants', [])
        if participants:
            output.append(f"Participants: {', '.join(participants)}")
        output.append('=' * 60)

        for idx in thread.get('message_indices', []):
            if idx < 1 or idx > len(messages):
                continue

            msg = messages[idx - 1]
            is_target = msg.get('ts') == target_ts
            marker = ">>>" if is_target else "   "

            name = msg.get('display_name') or msg.get('username') or 'Unknown'
            try:
                time_str = datetime.fromtimestamp(float(msg['ts'])).strftime('%H:%M')
            except (ValueError, TypeError):
                time_str = '??:??'

            text = msg.get('text', '') or ''
            text = text[:100] + ('...' if len(text) > 100 else '')
            # Clean up newlines for display
            text = text.replace('\n', ' ')

            output.append(f"{marker} [{time_str}] {name}: {text}")

    return "\n".join(output)


def format_json_output(threads: List[dict], messages: List[dict], target_ts: str) -> str:
    """Format threads as JSON."""
    result = {
        'target_ts': target_ts,
        'threads': []
    }

    for thread in threads:
        thread_data = {
            'name': thread.get('name', ''),
            'confidence': thread.get('confidence', 0),
            'participants': thread.get('participants', []),
            'messages': []
        }

        for idx in thread.get('message_indices', []):
            if idx < 1 or idx > len(messages):
                continue

            msg = messages[idx - 1]
            thread_data['messages'].append({
                'index': idx,
                'ts': msg.get('ts', ''),
                'username': msg.get('username', ''),
                'display_name': msg.get('display_name', ''),
                'text': msg.get('text', ''),
                'is_target': msg.get('ts') == target_ts
            })

        result['threads'].append(thread_data)

    return json.dumps(result, indent=2)


def format_html_output(threads: List[dict], messages: List[dict], target_ts: str, channel: str, theme: str = "light") -> str:
    """Format threads as HTML with tabbed interface."""
    import html

    def format_message_html(msg: dict, is_target: bool = False) -> str:
        """Format a single message as HTML."""
        name = html.escape(msg.get('display_name') or msg.get('username') or 'Unknown')
        text = html.escape(msg.get('text', '') or '')
        # Convert newlines to <br> and preserve some formatting
        text = text.replace('\n', '<br>')

        try:
            ts = float(msg['ts'])
            time_str = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        except (ValueError, TypeError):
            time_str = 'Unknown time'

        target_class = ' target-message' if is_target else ''
        target_badge = '<span class="target-badge">TARGET</span>' if is_target else ''

        return f'''
        <div class="message{target_class}">
            <div class="message-header">
                <span class="message-author">{name}</span>
                <span class="message-time">{time_str}</span>
                {target_badge}
            </div>
            <div class="message-text">{text}</div>
        </div>'''

    # Build thread tabs HTML
    thread_tabs = []
    thread_contents = []

    # "All Messages" tab
    thread_tabs.append('<button class="tab-button active" onclick="openTab(event, \'all-messages\')">All Messages</button>')

    all_messages_html = ''.join(
        format_message_html(msg, msg.get('ts') == target_ts)
        for msg in messages
    )
    thread_contents.append(f'''
    <div id="all-messages" class="tab-content active">
        <div class="thread-info">
            <strong>All {len(messages)} messages</strong> in context window
        </div>
        {all_messages_html}
    </div>''')

    # Thread tabs
    for i, thread in enumerate(threads):
        tab_id = f'thread-{i}'
        thread_name = html.escape(thread.get('name', f'Thread {i+1}')[:30])
        confidence = thread.get('confidence', 0)

        thread_tabs.append(
            f'<button class="tab-button" onclick="openTab(event, \'{tab_id}\')">{thread_name}</button>'
        )

        participants = ', '.join(html.escape(p) for p in thread.get('participants', []))
        messages_html = ''

        for idx in thread.get('message_indices', []):
            if idx < 1 or idx > len(messages):
                continue
            msg = messages[idx - 1]
            messages_html += format_message_html(msg, msg.get('ts') == target_ts)

        thread_contents.append(f'''
    <div id="{tab_id}" class="tab-content">
        <div class="thread-info">
            <strong>{html.escape(thread.get('name', 'Unnamed'))}</strong>
            <span class="confidence">({confidence:.0%} confidence)</span>
            <br><span class="participants">Participants: {participants}</span>
        </div>
        {messages_html}
    </div>''')

    tabs_html = '\n        '.join(thread_tabs)
    contents_html = '\n    '.join(thread_contents)

    # Theme-specific CSS
    if theme == "dark":
        theme_css = '''
        body {
            background: #1a1a2e;
            color: #eee;
        }
        h1 {
            color: #fff;
        }
        .subtitle {
            color: #888;
        }
        .tab-container {
            border-bottom: 2px solid #333;
        }
        .tab-button {
            background: #2d2d44;
            color: #aaa;
        }
        .tab-button:hover {
            background: #3d3d5c;
            color: #fff;
        }
        .tab-button.active {
            background: #4a4a6a;
            color: #fff;
        }
        .thread-info {
            background: #2d2d44;
        }
        .thread-info strong {
            color: #7c7cff;
        }
        .confidence {
            color: #888;
        }
        .participants {
            color: #aaa;
        }
        .message {
            background: #252538;
            border-left: 3px solid #444;
        }
        .message:hover {
            background: #2a2a40;
        }
        .message.target-message {
            border-left: 3px solid #ff6b6b;
            background: #2d2535;
        }
        .message-author {
            color: #7c7cff;
        }
        .message-time {
            color: #666;
        }
        .message-text {
            color: #ddd;
        }
        .message-text a {
            color: #7c7cff;
        }'''
    else:  # light theme
        theme_css = '''
        body {
            background: #f5f5f5;
            color: #333;
        }
        h1 {
            color: #222;
        }
        .subtitle {
            color: #666;
        }
        .tab-container {
            border-bottom: 2px solid #ddd;
        }
        .tab-button {
            background: #e0e0e0;
            color: #555;
        }
        .tab-button:hover {
            background: #d0d0d0;
            color: #333;
        }
        .tab-button.active {
            background: #4a90d9;
            color: #fff;
        }
        .thread-info {
            background: #e8e8e8;
        }
        .thread-info strong {
            color: #2563eb;
        }
        .confidence {
            color: #666;
        }
        .participants {
            color: #777;
        }
        .message {
            background: #fff;
            border-left: 3px solid #ddd;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .message:hover {
            background: #fafafa;
        }
        .message.target-message {
            border-left: 3px solid #e53e3e;
            background: #fff5f5;
        }
        .message-author {
            color: #2563eb;
        }
        .message-time {
            color: #999;
        }
        .message-text {
            color: #444;
        }
        .message-text a {
            color: #2563eb;
        }'''

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Micro-Threads: #{html.escape(channel)}</title>
    <style>
        * {{
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            margin: 0;
            padding: 20px;
            line-height: 1.5;
        }}
        h1 {{
            margin-bottom: 5px;
        }}
        .subtitle {{
            margin-bottom: 20px;
        }}
        .tab-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            margin-bottom: 20px;
            padding-bottom: 10px;
        }}
        .tab-button {{
            padding: 10px 20px;
            border: none;
            cursor: pointer;
            border-radius: 8px 8px 0 0;
            font-size: 14px;
            transition: all 0.2s;
            max-width: 200px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        .tab-button.active {{
            font-weight: bold;
        }}
        .tab-content {{
            display: none;
            animation: fadeIn 0.3s;
        }}
        .tab-content.active {{
            display: block;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}
        .thread-info {{
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
        }}
        .confidence {{
            font-size: 14px;
        }}
        .participants {{
            font-size: 13px;
        }}
        .message {{
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 10px;
        }}
        .message-header {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 8px;
        }}
        .message-author {{
            font-weight: bold;
        }}
        .message-time {{
            font-size: 12px;
        }}
        .target-badge {{
            background: #e53e3e;
            color: #fff;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: bold;
        }}
        .message-text {{
            word-wrap: break-word;
        }}
        {theme_css}
    </style>
</head>
<body>
    <h1>Micro-Thread Detection</h1>
    <p class="subtitle">Channel: #{html.escape(channel)} | {len(threads)} threads detected | {len(messages)} messages analyzed</p>

    <div class="tab-container">
        {tabs_html}
    </div>

    {contents_html}

    <script>
        function openTab(evt, tabId) {{
            // Hide all tab contents
            const contents = document.querySelectorAll('.tab-content');
            contents.forEach(c => c.classList.remove('active'));

            // Deactivate all tab buttons
            const buttons = document.querySelectorAll('.tab-button');
            buttons.forEach(b => b.classList.remove('active'));

            // Show selected tab and activate button
            document.getElementById(tabId).classList.add('active');
            evt.currentTarget.classList.add('active');
        }}
    </script>
</body>
</html>'''


def main():
    parser = argparse.ArgumentParser(
        description="Detect conversational micro-threads in Slack messages"
    )
    parser.add_argument("-c", "--channel", required=True,
                        help="Channel name")
    parser.add_argument("-t", "--ts", required=True,
                        help="Target message timestamp")
    parser.add_argument("-w", "--window", type=int, default=DEFAULT_WINDOW,
                        help=f"Number of messages before/after target (default: {DEFAULT_WINDOW})")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL,
                        help=f"LLM model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--url", default=DEFAULT_LLM_URL,
                        help=f"LLM server URL (default: {DEFAULT_LLM_URL})")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")
    parser.add_argument("--html", action="store_true",
                        help="Output as HTML file with tabbed interface")
    parser.add_argument("--theme", choices=["light", "dark"], default="light",
                        help="HTML color theme (default: light)")
    parser.add_argument("-o", "--output",
                        help="Output file path (for --html, defaults to threads_<channel>.html)")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM, use embeddings-only clustering")

    args = parser.parse_args()

    # Fetch context window
    print(f"Fetching messages from #{args.channel} around timestamp {args.ts}...")
    messages, target_msg = fetch_context_window(args.channel, args.ts, args.window)

    if not messages:
        print(f"Error: No messages found in channel '{args.channel}'")
        sys.exit(1)

    if target_msg is None:
        print(f"Warning: Target message with ts={args.ts} not found in results.")
        print(f"Found {len(messages)} messages in the window.")

    print(f"Found {len(messages)} messages in context window.")

    # Get embeddings
    message_ids = [msg['id'] for msg in messages]
    embeddings = get_embeddings_for_messages(message_ids)
    print(f"Retrieved {len(embeddings)} embeddings from ChromaDB.")

    # Compute similarity matrix
    ids, similarity = compute_similarity_matrix(embeddings, message_ids)

    # Extract features
    features = extract_features(messages)

    # Get similarity hints
    similarity_hints = get_similarity_hints(ids, similarity, messages)

    # Classify threads
    if args.no_llm:
        print("Using embeddings-only clustering...")
        threads = classify_with_embeddings_only(messages, ids, similarity)
    else:
        print(f"Classifying with LLM ({args.model} at {args.url})...")
        threads = classify_with_llm(messages, features, similarity_hints,
                                     args.url, args.model)

        # Fallback to embeddings if LLM fails
        if not threads:
            print("LLM classification returned no results, falling back to embeddings...")
            threads = classify_with_embeddings_only(messages, ids, similarity)

    # Output results
    if args.html:
        html_content = format_html_output(threads, messages, args.ts, args.channel, args.theme)
        output_file = args.output or f"threads_{args.channel}.html"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"HTML output written to: {output_file}")
    elif args.json:
        print(format_json_output(threads, messages, args.ts))
    else:
        print(format_output(threads, messages, args.ts))


if __name__ == "__main__":
    main()
