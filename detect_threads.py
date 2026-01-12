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


def get_llm_client(base_url: str, api_key: Optional[str] = None) -> OpenAI:
    """Get OpenAI client configured for local or remote LLM."""
    return OpenAI(
        base_url=base_url,
        api_key=api_key or "not-needed"
    )


def load_user_mappings() -> Dict[str, str]:
    """
    Load user ID to display name mapping.
    Combines users.json (ID -> username) and display_names.json (username -> display name).
    Returns dict of user_id -> display_name (or username if no display name).
    """
    user_id_to_name = {}

    # Load users.json (user_id -> username)
    users_file = BASE_DIR / "users.json"
    if users_file.exists():
        with open(users_file, 'r', encoding='utf-8') as f:
            users = json.load(f)
    else:
        users = {}

    # Load display_names.json (username -> display_name)
    display_names_file = BASE_DIR / "display_names.json"
    if display_names_file.exists():
        with open(display_names_file, 'r', encoding='utf-8') as f:
            display_names = json.load(f)
    else:
        display_names = {}

    # Build user_id -> best_name mapping
    for user_id, username in users.items():
        display_name = display_names.get(username, '')
        user_id_to_name[user_id] = display_name if display_name else username

    return user_id_to_name


def replace_slack_links(text: str, html_format: bool = False) -> str:
    """
    Replace Slack URL links like <https://example.com> or <https://example.com|text> with proper links.

    Args:
        text: The message text
        html_format: If True, return HTML anchor tags; otherwise return plain text
    """
    def replace_link(match):
        url = match.group(1)
        display_text = match.group(2) if match.group(2) else url
        if html_format:
            return f'<a href="{url}" target="_blank" rel="noopener">{display_text}</a>'
        else:
            return display_text if display_text != url else url

    # Replace <URL> and <URL|display_text> patterns (but not user mentions <@...> or channel refs <#...>)
    text = re.sub(r'<(https?://[^|>]+)(?:\|([^>]+))?>', replace_link, text)
    return text


def replace_user_mentions(text: str, user_mappings: Dict[str, str], html_format: bool = False) -> str:
    """
    Replace Slack user mentions like <@U04EPRRJD> with @displayname.

    Args:
        text: The message text
        user_mappings: Dict of user_id -> display_name
        html_format: If True, wrap mentions in styled span tags
    """
    def replace_mention(match):
        user_id = match.group(1)
        name = user_mappings.get(user_id, user_id)
        if html_format:
            return f'<span class="user-mention">@{name}</span>'
        else:
            return f'@{name}'

    # Replace <@USER_ID> and <@USER_ID|display_name> patterns
    text = re.sub(r'<@([A-Z0-9]+)(?:\|[^>]*)?>', replace_mention, text)
    return text


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
                      llm_url: str, model: str, api_key: Optional[str] = None,
                      max_tokens: int = 4000) -> List[dict]:
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
4. Name each thread with a BROAD, GENERAL category (1-2 words max):
   - Use general topics, not specific details
   - Examples: "Weather" not "Weather forecast", "Movies" not "Movies/Tron discussion", "Hardware" not "RAM upgrade issue"
   - Think: what section of a newspaper or forum would this go in?
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

    debug_file = BASE_DIR / "llm_debug.json"

    try:
        client = get_llm_client(llm_url, api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.3
        )

        # Immediately save raw response info before any processing
        debug_data = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "response_type": type(response).__name__,
            "response_repr": repr(response)[:2000],  # Truncate if huge
        }

        # If response is a string, save it directly
        if isinstance(response, str):
            debug_data["raw_string_response"] = response[:5000]
            with open(debug_file, 'w', encoding='utf-8') as f:
                json.dump(debug_data, f, indent=2, default=str)
            print(f"Debug: LLM response saved to {debug_file}")
            print(f"Warning: Response was a raw string, not an OpenAI response object")
            # Try to extract JSON from the string response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    return result.get('threads', [])
                except json.JSONDecodeError:
                    pass
            return []

        # Add structured response info
        debug_data["choices_count"] = len(response.choices) if hasattr(response, 'choices') and response.choices else 0

        # Extract content - handle different response structures
        content = None
        message = None

        if hasattr(response, 'choices') and response.choices:
            message = response.choices[0].message if hasattr(response.choices[0], 'message') else None

        if message:
            debug_data["message_role"] = getattr(message, 'role', None)
            debug_data["message_keys"] = [k for k in dir(message) if not k.startswith('_')]

            # Standard content field
            content = getattr(message, 'content', None)
            debug_data["content"] = content

            # Check for thinking model fields
            reasoning = getattr(message, 'reasoning_content', None)
            if reasoning:
                debug_data["reasoning_content"] = reasoning

            # Check for tool calls or other fields
            tool_calls = getattr(message, 'tool_calls', None)
            if tool_calls:
                debug_data["tool_calls"] = str(tool_calls)

            # Some models put content in different places
            if not content:
                # Try to find content in other common locations
                for attr in ['text', 'output', 'response', 'answer']:
                    alt_content = getattr(message, attr, None)
                    if alt_content:
                        content = alt_content
                        debug_data[f"alt_content_{attr}"] = alt_content
                        break

        # Save debug info to file
        with open(debug_file, 'w', encoding='utf-8') as f:
            json.dump(debug_data, f, indent=2, default=str)
        print(f"Debug: LLM response saved to {debug_file}")

        # Handle empty content - try reasoning_content as fallback for thinking models
        if not content or (isinstance(content, str) and len(content.strip()) == 0):
            print("Warning: LLM returned empty content field")
            # Check if reasoning_content has the answer (common with thinking models)
            if reasoning and isinstance(reasoning, str):
                print("  Found reasoning_content, checking for JSON in reasoning...")
                # Sometimes the JSON is at the end of the reasoning
                json_in_reasoning = re.search(r'\{\s*"threads"\s*:\s*\[[\s\S]*\]\s*\}', reasoning)
                if json_in_reasoning:
                    print("  Found JSON in reasoning_content, using that")
                    content = json_in_reasoning.group()
                else:
                    print(f"  No JSON found in reasoning_content ({len(reasoning)} chars)")
                    print(f"  Check {debug_file} for full response")
                    return []
            else:
                print(f"  Response object had {len(debug_data.get('message_keys', []))} message attributes")
                return []

        # For thinking models, strip out <think>...</think> or similar tags
        if isinstance(content, str):
            # Remove common thinking tags
            content_cleaned = re.sub(r'<think>[\s\S]*?</think>', '', content, flags=re.IGNORECASE)
            content_cleaned = re.sub(r'<thinking>[\s\S]*?</thinking>', '', content_cleaned, flags=re.IGNORECASE)
            content_cleaned = re.sub(r'<reasoning>[\s\S]*?</reasoning>', '', content_cleaned, flags=re.IGNORECASE)
            if content_cleaned.strip():
                content = content_cleaned

        # Try to extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            try:
                result = json.loads(json_match.group())
                threads = result.get('threads', [])
                if not threads:
                    print("Warning: LLM response parsed but 'threads' array is empty")
                    print(f"  Parsed JSON keys: {list(result.keys())}")
                return threads
            except json.JSONDecodeError as e:
                print(f"Warning: Found JSON-like content but failed to parse: {e}")
                print(f"  Check {debug_file} for full response")
                return []
        else:
            print("Warning: No JSON found in LLM response")
            print(f"  Check {debug_file} for full response")
            return []

    except json.JSONDecodeError as e:
        print(f"Warning: Could not parse LLM response as JSON: {e}")
    except Exception as e:
        print(f"Warning: LLM classification failed: {e}")
        import traceback
        traceback.print_exc()

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


def format_output(threads: List[dict], messages: List[dict], target_ts: str, user_mappings: Optional[Dict[str, str]] = None) -> str:
    """Format threads for display, highlighting target message."""
    if not threads:
        return "No threads detected."

    if user_mappings is None:
        user_mappings = {}

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
            text = replace_slack_links(text, html_format=False)
            text = replace_user_mentions(text, user_mappings, html_format=False)
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


def format_html_output(threads: List[dict], messages: List[dict], target_ts: str, channel: str, theme: str = "light", user_mappings: Optional[Dict[str, str]] = None) -> str:
    """Format threads as HTML with tabbed interface."""
    import html as html_module

    if user_mappings is None:
        user_mappings = {}

    # Color palette for threads (works well in both light and dark themes)
    thread_colors = [
        '#3b82f6',  # blue
        '#10b981',  # green
        '#f59e0b',  # amber
        '#ef4444',  # red
        '#8b5cf6',  # purple
        '#06b6d4',  # cyan
        '#f97316',  # orange
        '#ec4899',  # pink
        '#14b8a6',  # teal
        '#6366f1',  # indigo
    ]

    # Build message index to thread mapping
    msg_to_thread = {}  # msg_index (1-based) -> thread_index
    thread_names = {}  # thread_index -> thread_name
    for thread_idx, thread in enumerate(threads):
        thread_names[thread_idx] = thread.get('name', f'Thread {thread_idx + 1}')
        for msg_idx in thread.get('message_indices', []):
            msg_to_thread[msg_idx] = thread_idx

    def get_thread_color(thread_idx: int) -> str:
        return thread_colors[thread_idx % len(thread_colors)]

    # Avatar colors for users
    avatar_colors = [
        '#e91e63', '#9c27b0', '#673ab7', '#3f51b5', '#2196f3',
        '#03a9f4', '#00bcd4', '#009688', '#4caf50', '#8bc34a',
        '#ff9800', '#ff5722', '#795548', '#607d8b',
    ]

    def get_avatar_color(username: str) -> str:
        """Get a consistent color for a username."""
        hash_val = sum(ord(c) for c in (username or 'unknown'))
        return avatar_colors[hash_val % len(avatar_colors)]

    def get_initials(name: str) -> str:
        """Get initials from a name (up to 2 chars)."""
        if not name:
            return '?'
        parts = name.split()
        if len(parts) >= 2:
            return (parts[0][0] + parts[-1][0]).upper()
        return name[:2].upper()

    def format_message_html(msg: dict, msg_index: int, is_target: bool = False,
                            thread_idx: Optional[int] = None, thread_name: Optional[str] = None,
                            clickable: bool = False) -> str:
        """Format a single message as HTML."""
        display_name = msg.get('display_name') or msg.get('username') or 'Unknown'
        name = html_module.escape(display_name)
        text = msg.get('text', '') or ''

        # Avatar
        initials = get_initials(display_name)
        avatar_color = get_avatar_color(msg.get('username') or display_name)

        # Replace Slack links with HTML anchor tags
        text = replace_slack_links(text, html_format=True)

        # Replace user mentions with HTML spans
        text = replace_user_mentions(text, user_mappings, html_format=True)

        # Extract HTML elements and replace with indexed placeholders (before HTML escaping)
        html_elements = []
        def extract_html(match):
            html_elements.append(match.group(0))
            return f'__HTML_{len(html_elements)-1}__'
        # Extract anchor tags
        text = re.sub(r'<a [^>]+>[^<]*</a>', extract_html, text)
        # Extract user mention spans
        text = re.sub(r'<span class="user-mention">@[^<]+</span>', extract_html, text)

        # Now HTML escape the text (placeholders are safe)
        text = html_module.escape(text)

        # Restore the original HTML elements (unescaped)
        for i, element in enumerate(html_elements):
            text = text.replace(f'__HTML_{i}__', element)

        text = text.replace('\n', '<br>')

        try:
            ts = float(msg['ts'])
            time_str = datetime.fromtimestamp(ts).strftime('%I:%M %p')
        except (ValueError, TypeError):
            time_str = ''

        target_class = ' target-message' if is_target else ''
        target_badge = '<span class="target-badge">TARGET</span>' if is_target else ''

        # Thread badge
        thread_badge = ''
        click_attr = ''
        clickable_class = ''

        if thread_idx is not None:
            color = get_thread_color(thread_idx)
            badge_label = html_module.escape(thread_name[:20]) if thread_name else f'Thread {thread_idx + 1}'
            thread_badge = f'<span class="thread-badge" style="background: {color};">{badge_label}</span>'
            if clickable:
                escaped_name = html_module.escape(thread_name or f'Thread {thread_idx + 1}').replace("'", "\\'")
                click_attr = f'onclick="openTab(null, \'thread-{thread_idx}\', \'{escaped_name}\', \'{color}\')"'
                clickable_class = ' clickable'

        return f'''
        <div class="message{target_class}{clickable_class}" {click_attr}>
            <div class="avatar" style="border-color: {avatar_color}; color: {avatar_color};">{initials}</div>
            <div class="message-body">
                <div class="message-header">
                    <span class="message-author">{name}</span>
                    <span class="message-time">{time_str}</span>
                    {thread_badge}
                    {target_badge}
                </div>
                <div class="message-text">{text}</div>
            </div>
        </div>'''

    # Build thread tabs HTML (sidebar items)
    thread_tabs = []
    thread_contents = []

    # "All Messages" tab
    thread_tabs.append('<button class="thread-item active" data-tab="all-messages" onclick="openTab(event, \'all-messages\', \'All Messages\', \'\')">All Messages</button>')

    # Build all messages with thread coloring
    all_messages_parts = []
    for i, msg in enumerate(messages):
        msg_index = i + 1  # 1-based
        thread_idx = msg_to_thread.get(msg_index)
        thread_name = thread_names.get(thread_idx) if thread_idx is not None else None
        is_target = msg.get('ts') == target_ts
        all_messages_parts.append(
            format_message_html(msg, msg_index, is_target, thread_idx, thread_name, clickable=True)
        )
    all_messages_html = ''.join(all_messages_parts)

    thread_contents.append(f'''
    <div id="all-messages" class="tab-content active">
        {all_messages_html}
    </div>''')

    # Thread tabs with color indicators
    for i, thread in enumerate(threads):
        tab_id = f'thread-{i}'
        thread_name_raw = thread.get('name', f'Thread {i+1}')
        thread_name = html_module.escape(thread_name_raw[:30])
        confidence = thread.get('confidence', 0)
        color = get_thread_color(i)

        thread_tabs.append(
            f'<button class="thread-item" data-tab="{tab_id}" onclick="openTab(event, \'{tab_id}\', \'{thread_name}\', \'{color}\')">'
            f'<span class="thread-dot" style="background: {color};"></span>'
            f'<span class="thread-item-text">{thread_name}</span>'
            f'</button>'
        )

        participants = ', '.join(html_module.escape(p) for p in thread.get('participants', []))
        messages_html = ''

        for idx in thread.get('message_indices', []):
            if idx < 1 or idx > len(messages):
                continue
            msg = messages[idx - 1]
            messages_html += format_message_html(msg, idx, msg.get('ts') == target_ts, i, thread_names.get(i), clickable=False)

        thread_contents.append(f'''
    <div id="{tab_id}" class="tab-content">
        <div class="thread-info">
            <strong>{len(thread.get('message_indices', []))} messages</strong> &middot; {confidence:.0%} confidence &middot; {participants}
        </div>
        {messages_html}
    </div>''')

    tabs_html = '\n        '.join(thread_tabs)
    contents_html = '\n    '.join(thread_contents)

    # Theme-specific CSS
    if theme == "dark":
        theme_css = '''
        :root {
            --bg-primary: #1a1d21;
            --bg-secondary: #222529;
            --bg-hover: #2c2f33;
            --bg-active: #1164a3;
            --text-primary: #d1d2d3;
            --text-secondary: #ababad;
            --text-muted: #616061;
            --border-color: #393a3d;
            --author-color: #e8e8e9;
            --link-color: #1d9bd1;
            --target-bg: #3a2a2a;
            --target-border: #e53e3e;
        }'''
    else:
        theme_css = '''
        :root {
            --bg-primary: #fff;
            --bg-secondary: #f8f8f8;
            --bg-hover: #f0f0f0;
            --bg-active: #1164a3;
            --text-primary: #1d1c1d;
            --text-secondary: #616061;
            --text-muted: #868686;
            --border-color: #e0e0e0;
            --author-color: #1d1c1d;
            --link-color: #1264a3;
            --target-bg: #fff5f5;
            --target-border: #e53e3e;
        }'''

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>#{html_module.escape(channel)}</title>
    <style>
        {theme_css}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: Slack-Lato, Lato, appleLogo, sans-serif;
            font-size: 15px;
            line-height: 1.46668;
            background: var(--bg-primary);
            color: var(--text-primary);
            height: 100vh;
            display: flex;
            flex-direction: column;
        }}
        .app-container {{
            display: flex;
            flex: 1;
            overflow: hidden;
        }}
        /* Sidebar */
        .sidebar {{
            width: 240px;
            background: var(--bg-secondary);
            border-right: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            flex-shrink: 0;
        }}
        .sidebar-header {{
            padding: 12px 16px;
            border-bottom: 1px solid var(--border-color);
            font-weight: 700;
            font-size: 15px;
        }}
        .sidebar-header small {{
            font-weight: 400;
            color: var(--text-muted);
            font-size: 12px;
            display: block;
            margin-top: 2px;
        }}
        .thread-list {{
            flex: 1;
            overflow-y: auto;
            padding: 8px 0;
        }}
        .thread-item {{
            padding: 6px 16px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 8px;
            color: var(--text-secondary);
            font-size: 14px;
            border: none;
            background: none;
            width: 100%;
            text-align: left;
        }}
        .thread-item:hover {{
            background: var(--bg-hover);
        }}
        .thread-item.active {{
            background: var(--bg-active);
            color: #fff;
        }}
        .thread-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            flex-shrink: 0;
        }}
        .thread-item-text {{
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        /* Main content */
        .main-content {{
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}
        .channel-header {{
            padding: 10px 20px;
            border-bottom: 1px solid var(--border-color);
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .channel-header .thread-dot {{
            width: 10px;
            height: 10px;
        }}
        .channel-header small {{
            font-weight: 400;
            color: var(--text-muted);
            font-size: 13px;
        }}
        .messages-container {{
            flex: 1;
            overflow-y: auto;
            padding: 8px 0;
        }}
        .tab-content {{
            display: none;
        }}
        .tab-content.active {{
            display: block;
        }}
        /* Messages */
        .message {{
            padding: 4px 20px;
            display: flex;
            gap: 8px;
        }}
        .message:hover {{
            background: var(--bg-hover);
        }}
        .message.target-message {{
            background: var(--target-bg);
            border-left: 3px solid var(--target-border);
            padding-left: 17px;
        }}
        .message.clickable {{
            cursor: pointer;
        }}
        .avatar {{
            width: 36px;
            height: 36px;
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 13px;
            background: var(--bg-primary);
            border: 2px solid;
            flex-shrink: 0;
        }}
        .message-body {{
            flex: 1;
            min-width: 0;
        }}
        .message-header {{
            display: flex;
            align-items: baseline;
            gap: 8px;
            flex-wrap: wrap;
        }}
        .message-author {{
            font-weight: 700;
            color: var(--author-color);
        }}
        .message-time {{
            font-size: 12px;
            color: var(--text-muted);
        }}
        .thread-badge {{
            font-size: 11px;
            padding: 1px 6px;
            border-radius: 3px;
            color: #fff;
            font-weight: 600;
        }}
        .target-badge {{
            font-size: 11px;
            padding: 1px 6px;
            border-radius: 3px;
            background: var(--target-border);
            color: #fff;
            font-weight: 600;
        }}
        .message-text {{
            word-wrap: break-word;
            margin-top: 2px;
        }}
        .message-text a {{
            color: var(--link-color);
            text-decoration: none;
        }}
        .message-text a:hover {{
            text-decoration: underline;
        }}
        .user-mention {{
            background: #e8f5fa;
            color: #1264a3;
            padding: 0 3px;
            border-radius: 3px;
            font-weight: 500;
        }}
        .thread-info {{
            padding: 8px 20px;
            color: var(--text-muted);
            font-size: 13px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .thread-info strong {{
            color: var(--text-primary);
        }}
    </style>
</head>
<body>
    <div class="app-container">
        <div class="sidebar">
            <div class="sidebar-header">
                #{html_module.escape(channel)}
                <small>{len(messages)} messages &middot; {len(threads)} threads</small>
            </div>
            <div class="thread-list">
                {tabs_html}
            </div>
        </div>
        <div class="main-content">
            <div class="channel-header" id="content-header">
                <span>All Messages</span>
                <small>Click a message to view its thread</small>
            </div>
            <div class="messages-container">
                {contents_html}
            </div>
        </div>
    </div>
    <script>
        function openTab(evt, tabId, threadName, threadColor) {{
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.querySelectorAll('.thread-item').forEach(b => b.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            if (evt && evt.currentTarget) {{
                evt.currentTarget.classList.add('active');
            }} else {{
                document.querySelector('[data-tab="' + tabId + '"]').classList.add('active');
            }}
            const header = document.getElementById('content-header');
            if (tabId === 'all-messages') {{
                header.innerHTML = '<span>All Messages</span><small>Click a message to view its thread</small>';
            }} else {{
                header.innerHTML = '<span class="thread-dot" style="background:' + threadColor + '"></span>' + threadName;
            }}
        }}
    </script>
</body>
</html>'''


def load_channels_mapping() -> Dict[str, str]:
    """Load channel ID to name mapping from channels.json."""
    channels_file = BASE_DIR / "channels.json"
    if channels_file.exists():
        with open(channels_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def parse_slack_url(url: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse a Slack message URL to extract channel ID and timestamp.

    URL format: https://workspace.slack.com/archives/CHANNEL_ID/pTIMESTAMP
    Example: https://pmgnet.slack.com/archives/G2510ESB0/p1768001778553349

    Returns: (channel_id, timestamp) or (None, None) if parsing fails
    """
    # Match pattern: /archives/CHANNEL_ID/pTIMESTAMP
    match = re.search(r'/archives/([A-Z0-9]+)/p(\d+)', url)
    if match:
        channel_id = match.group(1)
        raw_ts = match.group(2)
        # Convert p1768001778553349 -> 1768001778.553349 (insert dot before last 6 digits)
        if len(raw_ts) > 6:
            timestamp = f"{raw_ts[:-6]}.{raw_ts[-6:]}"
        else:
            timestamp = raw_ts
        return channel_id, timestamp
    return None, None


def main():
    parser = argparse.ArgumentParser(
        description="Detect conversational micro-threads in Slack messages"
    )
    parser.add_argument("-c", "--channel",
                        help="Channel name (or use --slack-url)")
    parser.add_argument("-t", "--ts",
                        help="Target message timestamp (or use --slack-url)")
    parser.add_argument("--slack-url",
                        help="Slack message URL (e.g., https://workspace.slack.com/archives/G2510ESB0/p1768001778553349)")
    parser.add_argument("-w", "--window", type=int, default=DEFAULT_WINDOW,
                        help=f"Number of messages before/after target (default: {DEFAULT_WINDOW})")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL,
                        help=f"LLM model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--llm-url", default=DEFAULT_LLM_URL,
                        help=f"LLM server URL (default: {DEFAULT_LLM_URL})")
    parser.add_argument("--api-key",
                        help="API key for LLM service (optional, for remote APIs)")
    parser.add_argument("--max-tokens", type=int, default=4000,
                        help="Max tokens for LLM response (default: 4000, increase for thinking models)")
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

    # Handle Slack URL parsing
    channel = args.channel
    ts = args.ts

    if args.slack_url:
        channel_id, parsed_ts = parse_slack_url(args.slack_url)
        if not channel_id or not parsed_ts:
            print(f"Error: Could not parse Slack URL: {args.slack_url}")
            print("Expected format: https://workspace.slack.com/archives/CHANNEL_ID/pTIMESTAMP")
            sys.exit(1)

        # Look up channel name from ID
        channels_map = load_channels_mapping()
        if channel_id in channels_map:
            channel = channels_map[channel_id]
            print(f"Resolved channel ID '{channel_id}' to '{channel}'")
        else:
            print(f"Warning: Channel ID '{channel_id}' not found in channels.json, using ID as name")
            channel = channel_id

        ts = parsed_ts
        print(f"Parsed timestamp: {ts}")

    # Validate we have required arguments
    if not channel or not ts:
        print("Error: Must provide either --slack-url or both -c/--channel and -t/--ts")
        sys.exit(1)

    # Fetch context window
    print(f"Fetching messages from #{channel} around timestamp {ts}...")
    messages, target_msg = fetch_context_window(channel, ts, args.window)

    if not messages:
        print(f"Error: No messages found in channel '{channel}'")
        sys.exit(1)

    if target_msg is None:
        print(f"Warning: Target message with ts={ts} not found in results.")
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
        print(f"Classifying with LLM ({args.model} at {args.llm_url})...")
        threads = classify_with_llm(messages, features, similarity_hints,
                                     args.llm_url, args.model, args.api_key, args.max_tokens)

        # Fallback to embeddings if LLM fails
        if not threads:
            print("LLM classification returned no results, falling back to embeddings...")
            threads = classify_with_embeddings_only(messages, ids, similarity)

    # Load user mappings for display name resolution
    user_mappings = load_user_mappings()

    # Output results
    if args.html:
        html_content = format_html_output(threads, messages, ts, channel, args.theme, user_mappings)
        output_file = args.output or f"threads_{channel}.html"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"HTML output written to: {output_file}")
    elif args.json:
        print(format_json_output(threads, messages, ts))
    else:
        print(format_output(threads, messages, ts, user_mappings))


if __name__ == "__main__":
    main()
