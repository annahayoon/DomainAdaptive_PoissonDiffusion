#!/usr/bin/env python3
"""
Phase 7 Validation & Testing Orchestrator

This script runs the complete Phase 7 validation suite including:
- End-to-end integration testing (Task 7.1)
- Scientific validation (Task 7.2)
- Edge case handling (Task 7.3)

Generates comprehensive reports and validates system readiness.

Usage:
    python scripts/run_phase7_validation.py [--output-dir results/] [--verbose]
"""

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.logging_config import get_logger, setup_project_logging

logger = get_logger(__name__)


class Phase7ValidationOrchestrator:
    """Orchestrates all Phase 7 validation tests."""

    def __init__(self, output_dir: Path, verbose: bool = False):
        self.output_dir = output_dir
        self.verbose = verbose
        self.results = {}
        self.start_time = None

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        log_level = "DEBUG" if verbose else "INFO"
        setup_project_logging(level=log_level)

    def run_test_suite(self, test_module: str, suite_name: str) -> Dict[str, Any]:
        """Run a specific test suite and capture results."""
        logger.info(f"Running {suite_name}...")

        start_time = time.time()

        try:
            # Run pytest on the specific module
            cmd = [
                sys.executable,
                "-m",
                "pytest",
                f"tests/{test_module}",
                "-v",
                "--tb=short",
            ]

            if not self.verbose:
                cmd.append("-q")

            result = subprocess.run(
                cmd,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )

            duration = time.time() - start_time

            # Parse results from stdout (simple parsing)
            stdout_lines = result.stdout.split("\n")

            # Extract test counts from pytest output
            passed = failed = skipped = errors = 0
            total = 0

            for line in stdout_lines:
                if "passed" in line and (
                    "failed" in line or "error" in line or "skipped" in line
                ):
                    # Parse summary line like "1 passed, 2 failed in 1.23s"
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "passed" and i > 0:
                            passed = int(parts[i - 1])
                        elif part == "failed" and i > 0:
                            failed = int(parts[i - 1])
                        elif part == "skipped" and i > 0:
                            skipped = int(parts[i - 1])
                        elif part == "error" and i > 0:
                            errors = int(parts[i - 1])
                elif line.strip().endswith("passed"):
                    # Simple case: "3 passed"
                    parts = line.split()
                    if len(parts) >= 2 and parts[-1] == "passed":
                        passed = int(parts[-2])

            total = passed + failed + skipped + errors
            success_rate = (passed / max(total, 1)) * 100

            suite_results = {
                "suite_name": suite_name,
                "module": test_module,
                "duration_seconds": duration,
                "total_tests": total,
                "passed": passed,
                "failed": failed,
                "skipped": skipped,
                "errors": errors,
                "success_rate": success_rate,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "status": "PASSED" if result.returncode == 0 else "FAILED",
            }

            if result.returncode == 0:
                logger.info(f"✅ {suite_name} completed successfully")
            else:
                logger.error(
                    f"❌ {suite_name} failed with exit code {result.returncode}"
                )
                if self.verbose:
                    logger.error(f"STDERR: {result.stderr}")

            return suite_results

        except subprocess.TimeoutExpired:
            logger.error(f"❌ {suite_name} timed out after 10 minutes")
            return {
                "suite_name": suite_name,
                "module": test_module,
                "duration_seconds": 600,
                "status": "TIMEOUT",
                "error": "Test suite timed out",
            }

        except Exception as e:
            logger.error(f"❌ {suite_name} failed with exception: {e}")
            return {
                "suite_name": suite_name,
                "module": test_module,
                "status": "ERROR",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

    def run_integration_tests(self) -> Dict[str, Any]:
        """Run Task 7.1: Integration Testing."""
        logger.info("=" * 60)
        logger.info("TASK 7.1: INTEGRATION TESTING")
        logger.info("=" * 60)

        return self.run_test_suite(
            "test_end_to_end_integration.py", "integration_tests"
        )

    def run_scientific_validation(self) -> Dict[str, Any]:
        """Run Task 7.2: Scientific Validation."""
        logger.info("=" * 60)
        logger.info("TASK 7.2: SCIENTIFIC VALIDATION")
        logger.info("=" * 60)

        return self.run_test_suite(
            "test_scientific_validation.py", "scientific_validation"
        )

    def run_edge_case_tests(self) -> Dict[str, Any]:
        """Run Task 7.3: Edge Case Handling."""
        logger.info("=" * 60)
        logger.info("TASK 7.3: EDGE CASE HANDLING")
        logger.info("=" * 60)

        return self.run_test_suite("test_edge_case_handling.py", "edge_case_tests")

    def run_existing_test_validation(self) -> Dict[str, Any]:
        """Run validation on existing test suites to ensure no regressions."""
        logger.info("=" * 60)
        logger.info("REGRESSION TESTING: EXISTING TEST SUITES")
        logger.info("=" * 60)

        # Run key existing test suites
        existing_suites = [
            ("test_transforms.py", "transforms_regression"),
            ("test_calibration.py", "calibration_regression"),
            ("test_poisson_guidance.py", "guidance_regression"),
            ("test_metrics.py", "metrics_regression"),
            ("test_patch_processing.py", "patch_processing_regression"),
        ]

        regression_results = {}

        for test_module, suite_name in existing_suites:
            logger.info(f"Running regression test: {suite_name}")
            result = self.run_test_suite(test_module, suite_name)
            regression_results[suite_name] = result

        # Summarize regression results
        total_suites = len(regression_results)
        passed_suites = sum(
            1 for r in regression_results.values() if r.get("status") == "PASSED"
        )

        summary = {
            "total_regression_suites": total_suites,
            "passed_regression_suites": passed_suites,
            "regression_success_rate": (passed_suites / total_suites) * 100
            if total_suites > 0
            else 0,
            "individual_results": regression_results,
        }

        if passed_suites == total_suites:
            logger.info(f"✅ All {total_suites} regression tests passed")
        else:
            logger.warning(
                f"⚠️  {total_suites - passed_suites} regression tests failed"
            )

        return summary

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        logger.info("=" * 60)
        logger.info("GENERATING COMPREHENSIVE REPORT")
        logger.info("=" * 60)

        total_duration = time.time() - self.start_time

        # Calculate overall statistics
        all_results = [
            self.results.get("integration_tests", {}),
            self.results.get("scientific_validation", {}),
            self.results.get("edge_case_tests", {}),
        ]

        # Filter out regression results for main statistics
        main_results = [
            r
            for r in all_results
            if r.get("status") in ["PASSED", "FAILED", "TIMEOUT", "ERROR"]
        ]

        total_tests = sum(r.get("total_tests", 0) for r in main_results)
        total_passed = sum(r.get("passed", 0) for r in main_results)
        total_failed = sum(r.get("failed", 0) for r in main_results)
        total_errors = sum(r.get("errors", 0) for r in main_results)

        passed_suites = sum(1 for r in main_results if r.get("status") == "PASSED")
        total_suites = len(main_results)

        # Overall status
        if passed_suites == total_suites and total_suites > 0:
            overall_status = "PASSED"
        elif passed_suites > 0:
            overall_status = "PARTIAL"
        else:
            overall_status = "FAILED"

        # Regression test summary
        regression_results = self.results.get("regression_tests", {})
        regression_success_rate = regression_results.get("regression_success_rate", 0)

        report = {
            "phase": "Phase 7: Validation & Testing",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_duration_seconds": total_duration,
            "overall_status": overall_status,
            # Main test statistics
            "main_test_summary": {
                "total_test_suites": total_suites,
                "passed_test_suites": passed_suites,
                "suite_success_rate": (passed_suites / total_suites) * 100
                if total_suites > 0
                else 0,
                "total_individual_tests": total_tests,
                "passed_individual_tests": total_passed,
                "failed_individual_tests": total_failed,
                "error_individual_tests": total_errors,
                "individual_test_success_rate": (total_passed / total_tests) * 100
                if total_tests > 0
                else 0,
            },
            # Regression test summary
            "regression_test_summary": {
                "regression_success_rate": regression_success_rate,
                "regression_status": "PASSED"
                if regression_success_rate >= 90
                else "FAILED",
            },
            # Detailed results
            "detailed_results": self.results,
            # Requirements compliance
            "requirements_compliance": {
                "task_7_1_integration": self.results.get("integration_tests", {}).get(
                    "status"
                )
                == "PASSED",
                "task_7_2_scientific": self.results.get(
                    "scientific_validation", {}
                ).get("status")
                == "PASSED",
                "task_7_3_edge_cases": self.results.get("edge_case_tests", {}).get(
                    "status"
                )
                == "PASSED",
                "no_regressions": regression_success_rate >= 90,
            },
            # Recommendations
            "recommendations": self._generate_recommendations(
                overall_status, regression_success_rate
            ),
        }

        return report

    def _generate_recommendations(
        self, overall_status: str, regression_rate: float
    ) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []

        if overall_status == "PASSED":
            recommendations.append(
                "✅ All Phase 7 validation tests passed - system is ready for deployment"
            )

            if regression_rate >= 95:
                recommendations.append(
                    "✅ Excellent regression test performance - no significant regressions detected"
                )
            elif regression_rate >= 90:
                recommendations.append(
                    "⚠️  Minor regression issues detected - review failed tests"
                )
            else:
                recommendations.append(
                    "❌ Significant regressions detected - address before proceeding"
                )

        elif overall_status == "PARTIAL":
            recommendations.append(
                "⚠️  Some validation tests failed - review and fix issues before deployment"
            )
            recommendations.append(
                "🔍 Focus on failed test suites to identify root causes"
            )

        else:
            recommendations.append(
                "❌ Major validation failures - system not ready for deployment"
            )
            recommendations.append(
                "🚨 Critical issues detected - comprehensive review required"
            )

        # Specific recommendations based on individual test results
        if self.results.get("integration_tests", {}).get("status") != "PASSED":
            recommendations.append(
                "🔧 Fix integration test failures - end-to-end pipeline issues detected"
            )

        if self.results.get("scientific_validation", {}).get("status") != "PASSED":
            recommendations.append(
                "🧪 Address scientific validation failures - physics implementation issues"
            )

        if self.results.get("edge_case_tests", {}).get("status") != "PASSED":
            recommendations.append(
                "🛡️  Improve edge case handling - robustness issues detected"
            )

        return recommendations

    def save_report(self, report: Dict[str, Any]):
        """Save comprehensive report to files."""
        # Save JSON report
        json_file = self.output_dir / "phase7_validation_report.json"
        with open(json_file, "w") as f:
            json.dump(report, f, indent=2)

        # Save human-readable report
        text_file = self.output_dir / "phase7_validation_report.txt"
        with open(text_file, "w") as f:
            self._write_text_report(f, report)

        logger.info(f"Reports saved to:")
        logger.info(f"  JSON: {json_file}")
        logger.info(f"  Text: {text_file}")

    def _write_text_report(self, f, report: Dict[str, Any]):
        """Write human-readable text report."""
        f.write("=" * 80 + "\n")
        f.write("PHASE 7: VALIDATION & TESTING - COMPREHENSIVE REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {report['timestamp']}\n")
        f.write(f"Duration: {report['total_duration_seconds']:.1f} seconds\n")
        f.write(f"Overall Status: {report['overall_status']}\n")
        f.write("\n")

        # Main test summary
        main_summary = report["main_test_summary"]
        f.write("MAIN TEST SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(
            f"Test Suites: {main_summary['passed_test_suites']}/{main_summary['total_test_suites']} passed "
        )
        f.write(f"({main_summary['suite_success_rate']:.1f}%)\n")
        f.write(
            f"Individual Tests: {main_summary['passed_individual_tests']}/{main_summary['total_individual_tests']} passed "
        )
        f.write(f"({main_summary['individual_test_success_rate']:.1f}%)\n")
        f.write(f"Failed Tests: {main_summary['failed_individual_tests']}\n")
        f.write(f"Error Tests: {main_summary['error_individual_tests']}\n")
        f.write("\n")

        # Regression summary
        regression_summary = report["regression_test_summary"]
        f.write("REGRESSION TEST SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(
            f"Regression Success Rate: {regression_summary['regression_success_rate']:.1f}%\n"
        )
        f.write(f"Regression Status: {regression_summary['regression_status']}\n")
        f.write("\n")

        # Requirements compliance
        compliance = report["requirements_compliance"]
        f.write("REQUIREMENTS COMPLIANCE\n")
        f.write("-" * 40 + "\n")
        f.write(
            f"Task 7.1 (Integration): {'✅ PASSED' if compliance['task_7_1_integration'] else '❌ FAILED'}\n"
        )
        f.write(
            f"Task 7.2 (Scientific): {'✅ PASSED' if compliance['task_7_2_scientific'] else '❌ FAILED'}\n"
        )
        f.write(
            f"Task 7.3 (Edge Cases): {'✅ PASSED' if compliance['task_7_3_edge_cases'] else '❌ FAILED'}\n"
        )
        f.write(
            f"No Regressions: {'✅ PASSED' if compliance['no_regressions'] else '❌ FAILED'}\n"
        )
        f.write("\n")

        # Detailed results
        f.write("DETAILED RESULTS\n")
        f.write("-" * 40 + "\n")
        for suite_name, result in report["detailed_results"].items():
            if isinstance(result, dict) and "status" in result:
                status_icon = "✅" if result["status"] == "PASSED" else "❌"
                f.write(f"{status_icon} {suite_name}: {result['status']}")
                if "duration_seconds" in result:
                    f.write(f" ({result['duration_seconds']:.1f}s)")
                f.write("\n")
        f.write("\n")

        # Recommendations
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 40 + "\n")
        for rec in report["recommendations"]:
            f.write(f"{rec}\n")
        f.write("\n")

        f.write("=" * 80 + "\n")

    def run_complete_validation(self) -> Dict[str, Any]:
        """Run the complete Phase 7 validation suite."""
        logger.info("🚀 Starting Phase 7 Complete Validation Suite")
        logger.info("=" * 80)

        self.start_time = time.time()

        # Run all test suites
        self.results["integration_tests"] = self.run_integration_tests()
        self.results["scientific_validation"] = self.run_scientific_validation()
        self.results["edge_case_tests"] = self.run_edge_case_tests()
        self.results["regression_tests"] = self.run_existing_test_validation()

        # Generate comprehensive report
        report = self.generate_comprehensive_report()

        # Save report
        self.save_report(report)

        # Print summary
        self._print_final_summary(report)

        return report

    def _print_final_summary(self, report: Dict[str, Any]):
        """Print final summary to console."""
        logger.info("=" * 80)
        logger.info("PHASE 7 VALIDATION COMPLETE")
        logger.info("=" * 80)

        status = report["overall_status"]
        if status == "PASSED":
            logger.info("🎉 ALL VALIDATION TESTS PASSED!")
            logger.info("✅ System is ready for deployment")
        elif status == "PARTIAL":
            logger.warning("⚠️  PARTIAL SUCCESS - Some tests failed")
            logger.warning("🔍 Review failed tests before proceeding")
        else:
            logger.error("❌ VALIDATION FAILED")
            logger.error("🚨 Critical issues detected - system not ready")

        main_summary = report["main_test_summary"]
        logger.info(
            f"📊 Test Suites: {main_summary['passed_test_suites']}/{main_summary['total_test_suites']} passed"
        )
        logger.info(
            f"📊 Individual Tests: {main_summary['passed_individual_tests']}/{main_summary['total_individual_tests']} passed"
        )

        regression_rate = report["regression_test_summary"]["regression_success_rate"]
        logger.info(f"📊 Regression Rate: {regression_rate:.1f}%")

        logger.info(
            f"⏱️  Total Duration: {report['total_duration_seconds']:.1f} seconds"
        )
        logger.info("=" * 80)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run Phase 7 validation suite")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/phase7_validation",
        help="Output directory for reports",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Run validation
    orchestrator = Phase7ValidationOrchestrator(output_dir, args.verbose)
    report = orchestrator.run_complete_validation()

    # Exit with appropriate code
    if report["overall_status"] == "PASSED":
        sys.exit(0)
    elif report["overall_status"] == "PARTIAL":
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()
