#!/bin/bash
# nanoGPT Installation Script
# Downloads the OFFICIAL nanoGPT model.py from karpathy/nanoGPT repository
# This is REQUIRED before using the nanogpt model

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
NANOGPT_DIR="$PROJECT_ROOT/models/nanogpt"
TARGET_FILE="$NANOGPT_DIR/model.py"

echo "============================================"
echo "nanoGPT Installation Script"
echo "============================================"
echo "Downloading official nanoGPT from:"
echo "  https://github.com/karpathy/nanoGPT"
echo ""

# Create nanogpt directory if it doesn't exist
mkdir -p "$NANOGPT_DIR"

# Download model.py from nanoGPT repository
NANOGPT_URL="https://raw.githubusercontent.com/karpathy/nanoGPT/master/model.py"

echo "Downloading nanoGPT model.py..."
if command -v curl &> /dev/null; then
    curl -fsSL "$NANOGPT_URL" -o "$TARGET_FILE"
elif command -v wget &> /dev/null; then
    wget -q "$NANOGPT_URL" -O "$TARGET_FILE"
else
    echo "Error: Neither curl nor wget is installed."
    exit 1
fi

if [ -f "$TARGET_FILE" ]; then
    echo "✓ Successfully downloaded model.py"
    echo ""
    echo "Location: $TARGET_FILE"
else
    echo "✗ Failed to download model.py"
    exit 1
fi

echo ""
echo "============================================"
echo "Installation complete!"
echo "============================================"
echo ""
echo "You can now use the nanogpt model:"
echo "  python rng_ml_pipeline.py --model_name nanogpt --hardware <your_hardware> ..."
echo ""
