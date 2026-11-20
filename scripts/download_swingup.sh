#!/usr/bin/env bash
# Download only the swing-up agent archive into data/agents and extract it.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
dest="$root/data/agents"
mkdir -p "$dest"
archive="$dest/swingup.tar.gz"
url="https://huggingface.co/nzolman/sindy-rl_data/resolve/main/agents/swingup.tar.gz"
trial="$dest/swingup/dyna"

if [[ ! -f "$archive" ]]; then
  curl -fL --retry 3 -o "$archive" "$url"
fi

if [[ ! -d "$trial/checkpoint_001150" ]]; then
  tar -xzf "$archive" -C "$dest"
fi

if [[ ! -f "$trial/params.json" || ! -d "$trial/checkpoint_001150" ]]; then
  echo "Expected $trial with params.json and checkpoint_001150." >&2
  echo "Extracted tree:" >&2
  find "$dest" -maxdepth 5 \( -type d -o -name 'params.json' -o -name 'checkpoint_*' \) | sort >&2
  exit 1
fi

echo "Swing-up trial is at $trial"
