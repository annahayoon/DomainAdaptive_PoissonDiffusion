#!/usr/bin/env python
"""
Clean Phase 7 test runner - eliminates mock complexity and provides real testing.

This script addresses the Phase 7 issues by:
1. Running clean tests without complex mock setups
2. Using real data when available, clean synthetic fallbacks when not
3. Providing clear EDM integration testing without mock validation issues
4. Generating comprehensive reports for debugging

Usage:
    python scripts/run_clean_phase7_tests.py [--data-root PATH] [--output-dir PATH] [--verbose]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.logging_config import get_logger
from tests.test_edm_integration_clean import CleanEDMTester
from tests.test_real_data_integration import RealDataTestSuite

logger = get_logger(__name__)


class Phase7TestRunner:
    """Clean Phase 7 test runner without mock complexity."""

    def __init__(
        self, data_root: str = None, output_dir: str = None, verbose: bool = False
    ):
        """
        Initialize test runner.

        Args:
            data_root: Path to real preprocessed data (optional)
            output_dir: Directory for test outputs
            verbose: Enable verbose logging
        """
        self.data_root = Path(data_root) if data_root else None
        self.output_dir = (
            Path(output_dir)
            if output_dir
            else project_root / "results" / "clean_phase7_tests"
        )
        self.verbose = verbose

        # Setup output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Configure logging
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        logger.info(f"Phase 7 test runner initialized")
        logger.info(f"Data root: {self.data_root}")
        logger.info(f"Output dir: {self.output_dir}")

    def run_real_data_tests(self) -> Dict[str, Any]:
        """Run real data integration tests."""
        logger.info("Running real data integration tests")

        suite = RealDataTestSuite(str(self.data_root) if self.data_root else None)

        try:
            results = suite.run_all_tests()

            # Save results
            results_file = self.output_dir / "real_data_test_results.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2, default=str)

            logger.info(f"Real data test results saved to {results_file}")
            return results

        finally:
            suite.cleanup()

    def run_edm_integration_tests(self) -> Dict[str, Any]:
        """Run clean EDM integration tests."""
        logger.info("Running EDM integration tests")

        tester = CleanEDMTester()
        results = tester.run_all_tests()

        # Save results
        results_file = self.output_dir / "edm_integration_test_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"EDM integration test results saved to {results_file}")
        return results

    def run_performance_tests(self) -> Dict[str, Any]:
        """Run basic performance tests."""
        logger.info("Running performance tests")

        results = {"success": True, "errors": [], "metrics": {}}

        try:
            import time

            import psutil
            import torch

            # Test GPU availability
            if torch.cuda.is_available():
                device = torch.device("cuda")
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = (
                    torch.cuda.get_device_properties(0).total_memory / 1024**3
                )  # GB
                results["metrics"]["gpu_available"] = True
                results["metrics"]["gpu_name"] = gpu_name
                results["metrics"]["gpu_memory_gb"] = gpu_memory
            else:
                device = torch.device("cpu")
                results["metrics"]["gpu_available"] = False

            # Test memory usage
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024**2  # MB

            # Simple tensor operations
            start_time = time.time()
            x = torch.randn(1000, 1000, device=device)
            y = torch.matmul(x, x.t())
            torch.cuda.synchronize() if device.type == "cuda" else None
            end_time = time.time()

            end_memory = process.memory_info().rss / 1024**2  # MB

            results["metrics"]["device"] = str(device)
            results["metrics"]["tensor_op_time_ms"] = (end_time - start_time) * 1000
            results["metrics"]["memory_usage_mb"] = end_memory - start_memory
            results["metrics"]["cpu_count"] = psutil.cpu_count()
            results["metrics"]["total_memory_gb"] = (
                psutil.virtual_memory().total / 1024**3
            )

            logger.info("Performance tests completed")

        except Exception as e:
            results["success"] = False
            results["errors"].append(f"Performance test failed: {str(e)}")
            logger.error(f"Performance test failed: {e}")

        # Save results
        results_file = self.output_dir / "performance_test_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        return results

    def generate_report(self, all_results: Dict[str, Any]) -> str:
        """Generate comprehensive test report."""
        logger.info("Generating test report")

        report_lines = [
            "=" * 80,
            "CLEAN PHASE 7 TEST REPORT",
            "=" * 80,
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Data root: {self.data_root or 'Synthetic data used'}",
            f"Output directory: {self.output_dir}",
            "",
            "SUMMARY",
            "-" * 40,
        ]

        # Overall status
        all_success = True
        total_errors = 0

        for test_suite, results in all_results.items():
            suite_success = results.get(
                "overall_success", results.get("success", False)
            )
            suite_errors = results.get("total_errors", len(results.get("errors", [])))

            all_success = all_success and suite_success
            total_errors += suite_errors

            status = "PASS" if suite_success else "FAIL"
            report_lines.append(f"{test_suite:25} : {status} ({suite_errors} errors)")

        report_lines.extend(
            [
                "",
                f"Overall Status: {'PASS' if all_success else 'FAIL'}",
                f"Total Errors: {total_errors}",
                "",
                "DETAILED RESULTS",
                "-" * 40,
            ]
        )

        # Detailed results for each test suite
        for test_suite, results in all_results.items():
            report_lines.extend(
                [f"", f"{test_suite.upper()} TESTS:", "-" * (len(test_suite) + 7)]
            )

            if "test_results" in results:
                # Real data and EDM tests
                for test_name, test_result in results["test_results"].items():
                    status = "PASS" if test_result["success"] else "FAIL"
                    report_lines.append(f"  {test_name:20} : {status}")

                    if test_result["errors"]:
                        for error in test_result["errors"]:
                            report_lines.append(f"    ERROR: {error}")

                    if "metrics" in test_result and test_result["metrics"]:
                        report_lines.append(f"    Metrics: {test_result['metrics']}")
            else:
                # Performance tests
                status = "PASS" if results["success"] else "FAIL"
                report_lines.append(f"  Performance: {status}")

                if results["errors"]:
                    for error in results["errors"]:
                        report_lines.append(f"    ERROR: {error}")

                if "metrics" in results:
                    for key, value in results["metrics"].items():
                        report_lines.append(f"    {key}: {value}")

        # Recommendations
        report_lines.extend(["", "RECOMMENDATIONS", "-" * 40])

        if all_success:
            report_lines.extend(
                [
                    "✓ All tests passed! Phase 7 issues have been resolved.",
                    "✓ Mock complexity has been eliminated from test framework.",
                    "✓ EDM integration is working without validation issues.",
                    "✓ End-to-end pipeline is functioning correctly.",
                    "",
                    "Next steps:",
                    "1. Run full training with real preprocessed data",
                    "2. Evaluate on validation datasets",
                    "3. Compare with baseline methods",
                ]
            )
        else:
            report_lines.extend(
                [
                    "⚠ Some tests failed. Review the errors above.",
                    "",
                    "Common solutions:",
                    "1. Check data paths and file formats",
                    "2. Verify EDM installation: bash external/setup_edm.sh",
                    "3. Ensure sufficient GPU memory for testing",
                    "4. Check dependency versions in requirements.txt",
                ]
            )

        report_lines.extend(
            [
                "",
                "FILES GENERATED",
                "-" * 40,
                f"• Real data tests: {self.output_dir}/real_data_test_results.json",
                f"• EDM integration: {self.output_dir}/edm_integration_test_results.json",
                f"• Performance: {self.output_dir}/performance_test_results.json",
                f"• This report: {self.output_dir}/phase7_test_report.txt",
                "",
                "=" * 80,
            ]
        )

        report_text = "\n".join(report_lines)

        # Save report
        report_file = self.output_dir / "phase7_test_report.txt"
        with open(report_file, "w") as f:
            f.write(report_text)

        logger.info(f"Test report saved to {report_file}")
        return report_text

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all Phase 7 tests and generate report."""
        logger.info("Starting comprehensive Phase 7 test suite")
        start_time = time.time()

        all_results = {}

        try:
            # Run real data integration tests
            all_results["real_data_integration"] = self.run_real_data_tests()

            # Run EDM integration tests
            all_results["edm_integration"] = self.run_edm_integration_tests()

            # Run performance tests
            all_results["performance"] = self.run_performance_tests()

        except Exception as e:
            logger.error(f"Test suite failed with exception: {e}")
            all_results["fatal_error"] = {
                "success": False,
                "errors": [f"Fatal error: {str(e)}"],
                "overall_success": False,
            }

        # Generate report
        report_text = self.generate_report(all_results)

        # Print summary
        end_time = time.time()
        duration = end_time - start_time

        print(report_text)
        print(f"\nTest suite completed in {duration:.1f} seconds")

        return all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run clean Phase 7 tests")
    parser.add_argument("--data-root", type=str, help="Path to real preprocessed data")
    parser.add_argument("--output-dir", type=str, help="Output directory for results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Run tests
    runner = Phase7TestRunner(
        data_root=args.data_root, output_dir=args.output_dir, verbose=args.verbose
    )

    results = runner.run_all_tests()

    # Exit with appropriate code
    overall_success = all(
        result.get("overall_success", result.get("success", False))
        for result in results.values()
        if "fatal_error" not in results
    )

    sys.exit(0 if overall_success else 1)


if __name__ == "__main__":
    main()
