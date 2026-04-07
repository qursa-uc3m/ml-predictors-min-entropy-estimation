#!/bin/bash
# gbarp_gen Installation Script
# Initializes the gbarp_gen submodule and builds the C backend

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
GBARP_DIR="$PROJECT_ROOT/gbarp_gen"

echo "============================================"
echo "gbarp_gen Installation Script"
echo "============================================"

# Initialize and update the submodule
if [ ! -f "$GBARP_DIR/build.sh" ]; then
    echo "Initializing gbarp_gen submodule..."
    cd "$PROJECT_ROOT"
    git submodule update --init --recursive gbarp_gen
fi

# Build the C backend (optional, for faster generation)
echo "Building gbarp_gen C backend..."
cd "$GBARP_DIR"
./build.sh

echo ""
echo "============================================"
echo "Installation complete!"
echo "============================================"
echo ""
echo "The Python API is available via:"
echo "  from gbarp_gen.python import gbAR, constant_alpha"
echo ""
echo "The C backend (optional) is at:"
echo "  $GBARP_DIR/build/gbarp_gen"
echo ""
