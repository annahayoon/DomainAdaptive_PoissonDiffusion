#!/bin/bash
# Wrapper script to run photography parameter sweep with Python 3.10+
# Photography models require Python 3.10+ due to numpy 2.0+ compatibility

echo "Running photography parameter sweep with Python 3.10..."
echo ""

# Use Python 3.10 or 3.11
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
elif command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
else
    echo "ERROR: Python 3.10 or 3.11 not found!"
    echo "Available:"
    command -v python python3 python3.8 python3.9 python3.10 python3.11 2>/dev/null
    exit 1
fi

echo "Using: $PYTHON_CMD"
$PYTHON_CMD --version
echo ""

# Run the parameter sweep
cd /home/jilab/DAPGD
$PYTHON_CMD sample/fine_parameter_sweep.py --domains photography "$@"

