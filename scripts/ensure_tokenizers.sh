#!/usr/bin/env bash
# Install Hugging Face tokenizers (hard dependency of transformers>=4.57).
set -euo pipefail

cd "$(dirname "$0")/.."

if python -c "import tokenizers" 2>/dev/null; then
  python -c "import tokenizers; print(f'tokenizers already installed: {tokenizers.__version__}')"
  exit 0
fi

echo "Installing tokenizers (required by transformers)..."
pip install 'tokenizers>=0.21.0,<0.23.0'
python -c "import tokenizers; print(f'OK: tokenizers {tokenizers.__version__}')"
