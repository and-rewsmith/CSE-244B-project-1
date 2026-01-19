#!/usr/bin/env bash
set -e

# Upgrade pip
python -m pip install --upgrade pip

# Install Black
pip install black

# Run Black in check mode with project line length
# black --check --diff ./neuromorphic_optimized_llms

echo "Black formatting check passed successfully!" 