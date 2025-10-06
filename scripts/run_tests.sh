#!/bin/bash
#
# Test execution script
#
# PURPOSE: Run all tests with proper configuration and reporting

set -e  # Exit on error

echo "=========================================="
echo "DAPGD Test Suite"
echo "=========================================="

# Set Python path
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Run different test suites
echo ""
echo "1. Running PG guidance tests..."
echo "----------------------------------------"
pytest tests/test_pg_guidance.py -v --tb=short --cov=dapgd/guidance

echo ""
echo "2. Running sampler tests..."
echo "----------------------------------------"
pytest tests/test_sampling_simple.py -v --tb=short

echo ""
echo "3. Running all DAPGD tests with coverage..."
echo "----------------------------------------"
pytest tests/test_pg_guidance.py tests/test_sampling_simple.py -v \
    --cov=dapgd/guidance --cov=dapgd/sampling \
    --cov-report=html --cov-report=term

echo ""
echo "=========================================="
echo "All tests passed!"
echo "Coverage report: htmlcov/index.html"
echo "=========================================="
