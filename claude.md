# Slack Export Data

This folder contains exported Slack workspace data.

## Structure

- **Channel folders**: Each subfolder represents a Slack channel or direct message conversation. Contains JSON files with messages organized by date (e.g., `2024-01-15.json`).
- **channels.json**: Maps channel IDs to channel names
- **users.json**: Maps user IDs to screen names
- **display_names.json**: Maps screen names to friendly display names (user-maintained)

## Key Files

### channels.json
Maps Slack channel IDs to readable names. Channel ID prefixes indicate type:
- `C` prefix: Public channels (e.g., `C04E1TA8E` → `general`)
- `G` prefix: Private groups (e.g., `G2510ESB0` → `pmg-devs`)
- `D` prefix: Direct messages (e.g., `D04EQSD5Z` → `rcastles`)

### users.json
Maps Slack user IDs to screen names. Examples:
- `U04E1TA3L` → `rcastles`
- `U04EQSD4R` → `jlord`
- `U04E1U0F6` → `gpirela`
- `U04EPRRJD` → `balexander`
- `U0GE6M99R` → `dbenge`
- `U0GE8FPTK` → `jnester`

Bot/integration users are also included (e.g., `USLACKBOT` → `slackbot`, `U021Y5STYRX` → `zapier`).

### display_names.json
Maps screen names to human-friendly display names. Edit this file to add real names:
- `"rcastles": "Robert Castles"`
- `"jlord": "Jeff Lord"`

Empty values (`""`) mean no display name has been set yet. When querying by friendly name (e.g., "Robert Castles"), check this file to resolve to screen name ("rcastles"), then use users.json to get the user ID.

## Message Format

Messages in channel folders are JSON files named by date. Each message typically contains:
- `user`: User ID (lookup in users.json for name)
- `text`: Message content
- `ts`: Timestamp
- `type`: Usually "message"
- `reactions`: Any emoji reactions
- `thread_ts`: Parent message timestamp if in a thread
- `replies`: Thread replies if applicable

## Common Tasks

### Find messages from a user
1. Look up user ID in users.json
2. Search channel JSON files for that user ID

### Search for keywords
Search across all channel folders for text content in JSON files.

### Analyze channel activity
Count messages per channel, find most active users, etc.
