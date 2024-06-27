#!/bin/bash

# Ensure we exit immediately if any command fails
set -e

# Clean up old distribution files
echo "Cleaning up old distribution files..."
rm -rf dist/*

# Create source distribution and wheel
echo "Building source distribution and wheel..."
python -m build

# Optional: Run checks
echo "Running twine check..."
twine check dist/*


echo "Build complete. Distribution files are in the 'dist/' directory."
echo "To upload to test PyPI, run: twine upload --repository testpypi dist/*"
echo "Test installation from test PyPi, run: pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ml2sql"

echo "To upload to PyPI, run: twine upload dist/*"
