#!/bin/bash
set -e

echo "=== Dana Development Setup ==="

# Check Python
python3 --version || { echo "Python 3.12+ required"; exit 1; }

# Install dana-common
echo "Installing dana-common..."
pip install -e packages/dana-common[test]

# Install all services
for svc in services/*/; do
    if [ -f "$svc/pyproject.toml" ]; then
        echo "Installing $(basename $svc)..."
        pip install -e "$svc[test]"
    fi
done

# Install SDK
echo "Installing dana-sdk..."
pip install -e packages/dana-sdk[test]

echo "=== Setup complete! Run 'make up' to start services ==="
