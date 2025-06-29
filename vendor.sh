#!/bin/bash

# vendor.sh - Download and vendor CVXOPT C headers

set -e  # Exit on any error

echo "Vendoring CVXOPT C headers..."

# Create the extras/cvxopt/include directory
mkdir -p extras/cvxopt/include

# Download cvxopt.h
echo "Downloading cvxopt.h..."
curl -s -o extras/cvxopt/include/cvxopt.h https://raw.githubusercontent.com/cvxopt/cvxopt/master/src/C/cvxopt.h

# Download blas_redefines.h
echo "Downloading blas_redefines.h..."
curl -s -o extras/cvxopt/include/blas_redefines.h https://raw.githubusercontent.com/cvxopt/cvxopt/master/src/C/blas_redefines.h

# Verify files were downloaded
if [ -f "extras/cvxopt/include/cvxopt.h" ] && [ -f "extras/cvxopt/include/blas_redefines.h" ]; then
    echo "✓ CVXOPT headers successfully vendored to extras/cvxopt/include/"
    echo "✓ cvxopt.h ($(wc -c < extras/cvxopt/include/cvxopt.h) bytes)"
    echo "✓ blas_redefines.h ($(wc -c < extras/cvxopt/include/blas_redefines.h) bytes)"
else
    echo "✗ Failed to download CVXOPT headers"
    exit 1
fi

echo "Done! You can now build with CVXOPT support."
