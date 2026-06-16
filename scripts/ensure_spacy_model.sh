#!/usr/bin/env bash
# One-time install of spaCy en_core_web_sm (reward context F1).
# Use when GitHub download times out during training startup.
set -euo pipefail

cd "$(dirname "$0")/.."

if python -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null; then
  echo "en_core_web_sm already installed."
  exit 0
fi

echo "Installing en_core_web_sm via pip (PyPI wheel, avoids GitHub release assets)..."
pip install "en-core-web-sm==3.8.0" || python -m spacy download en_core_web_sm

python -c "import spacy; spacy.load('en_core_web_sm'); print('OK: en_core_web_sm')"
