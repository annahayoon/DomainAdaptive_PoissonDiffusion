#!/bin/bash
# Verification script for stratified evaluation integration

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     STRATIFIED EVALUATION INTEGRATION - VERIFICATION           ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check files exist
echo "📁 Checking files..."
echo ""

files_to_check=(
    "analysis/stratified_evaluation.py"
    "analysis/__init__.py"
    "STRATIFIED_EVALUATION_GUIDE.md"
    "INTEGRATION_COMPLETE.md"
    "IMPLEMENTATION_STATUS.md"
    "QUICK_START.md"
    "README_STRATIFIED_EVALUATION.md"
    "test_stratified_integration.py"
    "test_stratified_full_integration.py"
)

all_files_exist=true
for file in "${files_to_check[@]}"; do
    if [ -f "/home/jilab/Jae/$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file"
        all_files_exist=false
    fi
done

echo ""
echo "📝 Checking modifications..."
echo ""

# Check if sampling code was modified
if grep -q "from analysis.stratified_evaluation import StratifiedEvaluator" /home/jilab/Jae/sample/sample_noisy_pt_lle_PGguidance.py; then
    echo "✅ StratifiedEvaluator import added to sampling code"
else
    echo "❌ StratifiedEvaluator import NOT found in sampling code"
fi

if grep -q "--evaluate_stratified" /home/jilab/Jae/sample/sample_noisy_pt_lle_PGguidance.py; then
    echo "✅ --evaluate_stratified argument added"
else
    echo "❌ --evaluate_stratified argument NOT found"
fi

if grep -q "stratified_eval = StratifiedEvaluator" /home/jilab/Jae/sample/sample_noisy_pt_lle_PGguidance.py; then
    echo "✅ Stratified evaluation loop integrated"
else
    echo "❌ Stratified evaluation loop NOT found"
fi

echo ""
echo "🧪 Running tests..."
echo ""

cd /home/jilab/Jae

# Run integration test
if python test_stratified_integration.py > /tmp/test_integration.log 2>&1; then
    echo "✅ Integration tests passed"
    passed_integration=true
else
    echo "❌ Integration tests failed"
    passed_integration=false
fi

# Run full workflow test
if python test_stratified_full_integration.py > /tmp/test_full.log 2>&1; then
    echo "✅ Full workflow tests passed"
    passed_full=true
else
    echo "❌ Full workflow tests failed"
    passed_full=false
fi

echo ""
echo "📊 Summary"
echo ""

if [ "$all_files_exist" = true ] && [ "$passed_integration" = true ] && [ "$passed_full" = true ]; then
    echo "✅ ALL CHECKS PASSED"
    echo ""
    echo "Status: PRODUCTION READY"
    echo ""
    echo "Next steps:"
    echo "  1. Read QUICK_START.md (5 min)"
    echo "  2. Prepare test data with clean references"
    echo "  3. Run sampling code with --evaluate_stratified flag"
    echo "  4. Fill in Table 1 with results"
    exit 0
else
    echo "❌ SOME CHECKS FAILED"
    echo ""
    if [ "$passed_integration" = false ]; then
        echo "Integration test output:"
        tail -20 /tmp/test_integration.log
    fi
    if [ "$passed_full" = false ]; then
        echo "Full workflow test output:"
        tail -20 /tmp/test_full.log
    fi
    exit 1
fi
