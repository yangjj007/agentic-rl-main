#!/usr/bin/env bash
# One-time install of spaCy en_core_web_sm (reward context F1).
# Model is NOT on PyPI as en-core-web-sm — use GitHub release wheel or spacy download.
set -euo pipefail

cd "$(dirname "$0")/.."

if python -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null; then
  echo "en_core_web_sm already installed."
  exit 0
fi

WHEEL_URL="https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
MIRROR_URL="https://ghproxy.com/${WHEEL_URL}"
LOCAL_WHEEL="${HOME}/.cache/dyme_en_core_web_sm-3.8.0-py3-none-any.whl"

install_wheel() {
  local url="$1"
  echo "Trying: pip install ${url}"
  pip install "${url}"
}

mkdir -p "$(dirname "${LOCAL_WHEEL}")"

if ! install_wheel "${MIRROR_URL}" 2>/dev/null; then
  if ! install_wheel "${WHEEL_URL}" 2>/dev/null; then
    echo "pip wheel failed; trying wget + local install..."
    if command -v wget >/dev/null 2>&1; then
      wget -c -O "${LOCAL_WHEEL}" "${MIRROR_URL}" || wget -c -O "${LOCAL_WHEEL}" "${WHEEL_URL}"
      pip install "${LOCAL_WHEEL}"
    else
      echo "Falling back to: python -m spacy download en_core_web_sm"
      python -m spacy download en_core_web_sm
    fi
  fi
fi

python -c "import spacy; spacy.load('en_core_web_sm'); print('OK: en_core_web_sm')"
